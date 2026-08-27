"""Diagnose how much the learned residual can move the utility prior.

Two questions the benchmark cannot answer on its own:

`--mode residual` — does the trained policy actually emit a meaningful ΔΘ, or
has it collapsed onto the prior? Reports the distribution of every residual
component over a rollout, in absolute terms and relative to its allowed range.

`--mode sweep` — could *any* ΔΘ within the allowed range do better? Samples
parameter sets around Θ_base and reports the achievable spread of collisions and
arrival. If the spread is narrow, the ceiling is the functional form of the
utility rather than the parameter values, and no residual policy can fix it.

    python -m Baselines.diagnose_residual --mode residual --scenarios 2
    python -m Baselines.diagnose_residual --mode sweep --samples 12 --scenarios 2
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

import Baselines._paths  # noqa: F401
from Baselines.dynamics import apply_control, observation, observation_dim
from Baselines.metrics import rollout_metrics
from Baselines.residual_marl import DEFAULT_CHECKPOINT, load_residual_policy
from Baselines.runner import rollout
from Baselines.scenario import build_scenario
from Baselines.utility_prior import UtilityPriorController
from RL.calibration_io import (
    DEFAULT_RESIDUAL_SCALES,
    PARAM_GAUGE,
    RESIDUAL_PARAM_KEYS,
    apply_residual,
    load_base_params,
)
from RL.param_gauge import AMPLITUDE_KEYS, amplitude_proportions, base_logits
from RL.corridor import DEFAULT_LANE_KF, DEFAULT_RUN_ID


def report_residuals(args: argparse.Namespace) -> None:
    base = load_base_params(args.calibration, prefer="robust")
    scenario = build_scenario(
        seed=args.seed,
        num_agents=args.num_agents,
        max_steps=args.max_steps,
        run_id=args.run_id,
        lane_kf=args.lane_kf,
    )
    policy = load_residual_policy(args.checkpoint, observation_dim(scenario))

    samples: list[np.ndarray] = []
    for offset in range(args.scenarios):
        scenario = build_scenario(
            seed=args.seed + offset,
            num_agents=args.num_agents,
            max_steps=args.max_steps,
            run_id=args.run_id,
            lane_kf=args.lane_kf,
        )
        agents = scenario.spawn_agents()
        controller = UtilityPriorController(calibration=args.calibration)
        controller.reset(scenario)
        for _ in range(args.probe_steps):
            for i, agent in enumerate(agents):
                if agent.reached_destination:
                    continue
                delta, _ = policy.act(
                    np.asarray(observation(agents, i, scenario), dtype=np.float32), 0.0
                )
                samples.append(np.array([float(delta[k]) for k in RESIDUAL_PARAM_KEYS]))
            controls = controller.compute_controls(agents, scenario, 0)
            for i, control in enumerate(controls):
                if not agents[i].reached_destination:
                    apply_control(agents[i], control, scenario)

    stack = np.asarray(samples)
    rows = []
    for j, key in enumerate(RESIDUAL_PARAM_KEYS):
        scale = float(DEFAULT_RESIDUAL_SCALES[key])
        column = stack[:, j]
        if key.startswith("z_"):
            base_val = float(base_logits(base)[j])
            ref_label = "z_base"
        else:
            base_val = float(base.get(key, float("nan")))
            ref_label = "theta_base"
        rows.append(
            {
                "param": key,
                ref_label: base_val,
                "delta_mean": float(np.mean(column)),
                "delta_std": float(np.std(column)),
                "delta_absmax": float(np.max(np.abs(column))),
                "allowed": scale,
                "saturation": float(np.mean(np.abs(column)) / scale),
            }
        )
    frame = pd.DataFrame(rows).set_index("param")
    props = amplitude_proportions(base)
    print(f"\nGauge: {PARAM_GAUGE} | base amplitude proportions (sum=1):")
    for k in AMPLITUDE_KEYS:
        print(f"  {k}: {props[k]:.4f}")
    pd.set_option("display.width", 200)
    print(f"\nResidual usage over {stack.shape[0]} agent-steps ({args.checkpoint}):\n")
    print(frame.round(4).to_string())
    print(
        "\nsaturation = mean|delta| / allowed range; a value near 0 means the policy "
        "has collapsed onto the prior, near 1 means it is pinned at the bound."
    )


def report_sweep(args: argparse.Namespace) -> None:
    base = load_base_params(args.calibration, prefer="robust")
    rng = np.random.default_rng(args.seed)

    candidates: list[tuple[str, dict[str, float]]] = [("theta_base", dict(base))]
    for k in range(args.samples):
        delta = {
            key: float(rng.uniform(-DEFAULT_RESIDUAL_SCALES[key], DEFAULT_RESIDUAL_SCALES[key]))
            for key in RESIDUAL_PARAM_KEYS
        }
        candidates.append((f"sample_{k:02d}", apply_residual(base, delta)))

    rows = []
    for label, params in candidates:
        per_scenario = []
        for offset in range(args.scenarios):
            scenario = build_scenario(
                seed=args.seed + offset,
                num_agents=args.num_agents,
                max_steps=args.max_steps,
                run_id=args.run_id,
                lane_kf=args.lane_kf,
            )
            controller = UtilityPriorController(params=params)
            per_scenario.append(rollout_metrics(rollout(scenario, controller)))
        rows.append(
            {
                "setting": label,
                "collision_events": float(np.mean([m["collision_events"] for m in per_scenario])),
                "arrival_rate": float(np.mean([m["arrival_rate"] for m in per_scenario])),
                "mean_speed": float(np.mean([m["mean_speed_mps"] for m in per_scenario])),
                "rms_jerk": float(np.mean([m["rms_jerk"] for m in per_scenario])),
            }
        )
        print(f"  {rows[-1]['setting']:12s} collisions={rows[-1]['collision_events']:6.2f} "
              f"arrival={rows[-1]['arrival_rate']:5.2f}")

    frame = pd.DataFrame(rows).set_index("setting")
    print(f"\nAchievable spread over {args.samples} parameter draws:\n")
    print(frame.round(3).to_string())
    sweep = frame.drop(index="theta_base")
    print(
        f"\ncollisions: base={frame.loc['theta_base', 'collision_events']:.2f}, "
        f"sampled best={sweep['collision_events'].min():.2f}, "
        f"worst={sweep['collision_events'].max():.2f}\n"
        f"arrival:    base={frame.loc['theta_base', 'arrival_rate']:.2f}, "
        f"sampled best={sweep['arrival_rate'].max():.2f}, "
        f"worst={sweep['arrival_rate'].min():.2f}"
    )
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(args.output)
        print(f"Wrote {args.output}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Diagnose the residual policy against its prior")
    parser.add_argument("--mode", choices=["residual", "sweep"], default="residual")
    parser.add_argument("--scenarios", type=int, default=2)
    parser.add_argument("--samples", type=int, default=12)
    parser.add_argument("--num-agents", type=int, default=10)
    parser.add_argument("--max-steps", type=int, default=300)
    parser.add_argument("--probe-steps", type=int, default=40)
    parser.add_argument("--run-id", type=int, default=DEFAULT_RUN_ID)
    parser.add_argument("--lane-kf", type=int, default=DEFAULT_LANE_KF)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--calibration", type=Path, default=None)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    if args.mode == "residual":
        report_residuals(args)
    else:
        report_sweep(args)


if __name__ == "__main__":
    main()
