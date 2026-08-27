#!/usr/bin/env python
"""Evaluate an RLlib checkpoint on TrafficMAR-v0."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np

import RL._paths  # noqa: F401
from RL.gym_env import TrafficMARLEnv

try:
    import torch
    from ray.rllib.core.columns import Columns
except ImportError as exc:
    raise SystemExit(
        "RLlib/torch required for this evaluator. "
        "For the primary PPO checkpoints use visualize_simulation.py or RL.demo."
    ) from exc


def find_latest_checkpoint(checkpoint_dir: Path) -> Path:
    candidates = sorted(checkpoint_dir.glob("checkpoint_*"))
    if not candidates:
        raise FileNotFoundError(f"No checkpoints found in {checkpoint_dir}")
    return candidates[-1].resolve()


def load_algorithm(checkpoint_path: str | Path):
    from ray.rllib.algorithms.algorithm import Algorithm

    return Algorithm.from_checkpoint(str(Path(checkpoint_path).resolve()))


def compute_residual_action(
    algo,
    obs: np.ndarray,
    policy_id: str = "shared_policy",
    explore: bool = False,
) -> np.ndarray:
    module = algo.get_module(policy_id)
    obs_batch = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
    fwd = module.forward_inference({Columns.OBS: obs_batch})
    action_dist = module.get_inference_action_dist_cls().from_logits(fwd["action_dist_inputs"])
    if explore:
        action = action_dist.sample()
    else:
        action = action_dist.to_deterministic().sample()
    return action.squeeze(0).detach().cpu().numpy()


def record_rollout(env: TrafficMARLEnv, algo, explore: bool = False) -> dict[str, Any]:
    obs, _ = env.reset()
    agent_list = sorted(env.agents)
    n_agents = len(agent_list)
    positions: list[list[np.ndarray]] = [[] for _ in range(n_agents)]
    velocities: list[list[np.ndarray]] = [[] for _ in range(n_agents)]
    residuals: list[list[np.ndarray]] = [[] for _ in range(n_agents)]
    controls: list[list[dict[str, float]]] = [[] for _ in range(n_agents)]

    for i, aid in enumerate(agent_list):
        idx = int(aid.split("_")[1])
        positions[i].append(env._env.agents[idx].pos.copy())
        velocities[i].append(env._env.agents[idx].vel.copy())
        residuals[i].append(np.zeros(env.action_spaces[aid].shape))
        controls[i].append({"accel": 0.0, "steering": 0.0})

    done = False
    while not done:
        actions = {}
        for aid, o in obs.items():
            actions[aid] = compute_residual_action(
                algo, o, policy_id="shared_policy", explore=explore
            ).astype(np.float32)
        obs, _, terminateds, truncateds, _ = env.step(actions)
        done = terminateds["__all__"] or truncateds["__all__"]
        for i, aid in enumerate(agent_list):
            idx = int(aid.split("_")[1])
            positions[i].append(env._env.agents[idx].pos.copy())
            velocities[i].append(env._env.agents[idx].vel.copy())
            residuals[i].append(actions[aid].copy())
            controls[i].append(dict(env._env.agents[idx].prev_control))

    return {
        "positions": positions,
        "velocities": velocities,
        "residuals": residuals,
        "controls": controls,
        "destinations": [a.dest.copy() for a in env._env.agents],
        "metric": env.rollout_metric(),
        "collisions": env._env.collision_count,
        "steps": env._env.step_count,
        "road_y": (
            env._cfg.sim_config["road_y_min"],
            env._cfg.sim_config["road_y_max"],
        ),
        "highway_length": env._cfg.highway_length,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate RLlib TrafficMAR checkpoint")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("RL/checkpoints/rllib_ppo/checkpoint_final"),
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-steps", type=int, default=240)
    parser.add_argument("--num-agents", type=int, default=10)
    args = parser.parse_args()

    checkpoint = args.checkpoint.resolve()
    if checkpoint.is_dir() and not (checkpoint / "rllib_checkpoint.json").exists():
        checkpoint = find_latest_checkpoint(checkpoint)

    algo = load_algorithm(checkpoint)
    algo_env_config = dict(getattr(algo.config, "env_config", {}) or {})
    algo_env_config.update(
        {
            "seed": args.seed,
            "max_steps": args.max_steps,
            "num_agents": args.num_agents,
        }
    )
    env = TrafficMARLEnv(algo_env_config)
    rollout = record_rollout(env, algo, explore=False)
    print("RLlib residual-policy rollout:")
    print(
        {
            "metric": rollout["metric"],
            "collisions": rollout["collisions"],
            "steps": rollout["steps"],
        }
    )


if __name__ == "__main__":
    main()
