"""Bounded learning integration check, launched through a site's run_module.py.

All outputs stay below the site's runs/<run>/smoke directory. These tiny runs
check execution and checkpoint compatibility, not learned policy performance.
"""
from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import torch


def check_site_dynamics():
    from prepare_new_baselines import generation_config
    from RL.traffic_env import EnvConfig
    from new_baselines.data import inverse_controls
    from new_baselines.models import bicycle_rollout
    from utility_model import kinematic_bicycle_rollout

    cfg = generation_config()
    sim = EnvConfig().sim_config
    for field, key in (("dt", "dt"), ("length", "vehicle_length"),
                       ("width", "vehicle_width"), ("wheelbase", "wheelbase"),
                       ("max_speed", "max_agent_speed"), ("max_accel", "max_accel"),
                       ("perception_radius", "perception_radius"),
                       ("min_steer_speed", "min_steer_speed")):
        np.testing.assert_allclose(getattr(cfg, field), sim[key], err_msg=key)
    assert cfg.steer_from_rest == sim.get("steer_from_rest", False)
    initial = np.array([0., 0., 0., 0., 0., cfg.length, cfg.width, 1.], np.float32)
    actions = np.array([[2., .2], [1., -.1]], np.float32)
    states = [initial]
    for accel, steer in actions:
        old = states[-1]
        move = kinematic_bicycle_rollout(old[:2], old[4], np.linalg.norm(old[2:4]),
                                         accel, steer, cfg.dt, sim)
        states.append(np.r_[move["pos"], move["vel"], move["heading"], cfg.length, cfg.width, 1.])
    states = np.asarray(states, np.float32)[None]
    inverse, _, _ = inverse_controls(states, cfg)
    np.testing.assert_allclose(inverse[0, :-1], actions, atol=2e-6)
    pos, heading = bicycle_rollout(torch.tensor(initial)[None, None], torch.tensor(actions)[None, None], cfg)
    np.testing.assert_allclose(pos[0, 0].numpy(), states[0, 1:, :2], atol=2e-6)
    np.testing.assert_allclose(heading[0, 0].numpy(), states[0, 1:, 4], atol=2e-6)


def main():
    import config as site
    import prepare_new_baselines as preparation
    from Baselines.registry import build_controller, controller_kwargs
    from Baselines.runner import rollout
    from Baselines.scenario import build_scenario
    from new_baselines.training import train, select_checkpoints
    from RL.experiment_protocol import write_json

    torch.set_num_threads(1)
    check_site_dynamics()
    output = site.RUN_ROOT / "smoke"
    output.mkdir(parents=True, exist_ok=True)
    cfg = replace(preparation.generation_config(), hidden=32, layers=1, decoder_layers=1,
                  max_agents=4, context=4, history=3, horizon=4, diffusion_steps=4, return_bins=32)
    prepare = getattr(preparation, "prepare_new_baselines", None)
    if prepare is None:
        prepare = preparation.prepare_tgsim_new_baselines
    prepare(output / "data", site.TRAJECTORIES_CSV, cfg, train_scenes=2, val_scenes=2)
    scenario = build_scenario(810000, num_agents=3, max_steps=3)
    rows = {}
    for name in site.NEW_BASELINE_MODELS:
        raw = output / (name + "_raw.pt")
        payload = train(name, output / "data/train.npz", output / "data/validation.npz", raw, cfg,
                        steps=2, batch_size=2, eval_every=2, device="cpu", smoke=True)
        selected = output / (name + "_selected.pt")
        args = SimpleNamespace(num_agents=3, max_steps=3, val_episodes=1, test_episodes=1,
                               validation_seed_start=910000, test_seed_start=810000,
                               run_id=0, lane_kf=0, calibration=site.CALIBRATION, dense_spawn=False)
        select_checkpoints(name, sorted(raw.with_suffix(".candidates").glob("step_*.pt")),
                           selected, args, allow_smoke=True)
        controller = build_controller(name, checkpoint=selected, device="cpu", allow_smoke=True)
        result = rollout(scenario, controller)
        assert result.steps == 3 and np.isfinite(result.positions).all()
        try:
            build_controller(name, checkpoint=selected, device="cpu")
        except ValueError as exc:
            assert "Smoke checkpoint" in str(exc)
        else:
            raise AssertionError("Smoke checkpoint accepted by normal benchmark")
        rows[name] = dict(optimizer_steps=2, rollout_steps=result.steps,
                          validation_loss=payload["metadata"]["best_validation_loss"],
                          selected_checkpoint=str(selected))
        print(name + ": trained, selected, reloaded and rolled out", flush=True)

    trainers = {
        "residual_marl": ("RL.train_ppo", ["--residual-mode", "candidate_logits"]),
        "residual_param": ("RL.train_ppo", ["--residual-mode", "param_delta"]),
        "direct_discrete_rl": ("Baselines.train_direct_discrete_rl", ["--minibatch-size", "512"]),
        "mappo": ("Baselines.train_marl", ["--algo", "mappo"]),
        "pure_rl": ("Baselines.train_pure_rl", []),
        "happo": ("Baselines.train_marl", ["--algo", "happo"]),
        "hatrpo": ("Baselines.train_marl", ["--algo", "hatrpo"]),
    }
    for name in site.TRAIN_MODELS:
        module, extra = trainers[name]
        checkpoint = output / (name + ".pt")
        command = [sys.executable, str(site.CASE_ROOT / "run_module.py"), module, *extra,
                   "--updates", "1", "--episodes-per-update", "1", "--total-env-steps", "3",
                   "--num-agents", "3", "--max-steps", "3", "--hidden-dim", "32",
                   "--val-episodes", "1", "--test-episodes", "1", "--skip-test",
                   "--calibration", str(site.CALIBRATION), "--save", str(checkpoint)]
        with (output / (name + ".log")).open("w", encoding="utf-8") as log:
            subprocess.run(command, cwd=site.REPO_ROOT, stdout=log, stderr=subprocess.STDOUT,
                           check=True, timeout=300)
        summary = json.loads(checkpoint.with_suffix(".summary.json").read_text())
        assert summary["budget"]["environment_steps"] == 3
        controller = build_controller(name, **controller_kwargs(
            name, checkpoint_override=checkpoint, calibration=site.CALIBRATION))
        result = rollout(scenario, controller)
        assert result.steps == 3 and np.isfinite(result.positions).all()
        rows[name] = dict(environment_steps=3, rollout_steps=result.steps)
        print(name + ": trained, reloaded and rolled out", flush=True)
    write_json(output / "summary.json", dict(smoke_only=True, models=rows,
               note="Execution checks only; these checkpoints are not performance benchmarks."))


if __name__ == "__main__":
    main()
