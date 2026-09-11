#!/usr/bin/env python
"""Compare utility-only baseline vs residual-modulated rollout."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

import RL._paths  # noqa: F401
from RL.calibration_io import DEFAULT_CALIBRATION_PATH, load_base_params
from RL.traffic_env import EnvConfig, MultiAgentTrafficEnv

try:
    import torch
    from RL.train_ppo import TorchResidualPolicy, action_space_from_blob, residual_mode_from_blob
except ImportError:
    torch = None
    TorchResidualPolicy = None


def run_rollout(env: MultiAgentTrafficEnv, policy=None, explore_std: float = 0.0) -> dict:
    if policy is not None:
        env.config.residual_mode = policy.residual_mode
    obs_list = env.reset()
    done = False
    rewards = []
    while not done:
        if policy is None:
            residual_actions = None
        else:
            residual_actions = []
            for obs in obs_list:
                action, _ = policy.act(obs, explore_std=explore_std)
                residual_actions.append(action)
        obs_list, rewards, done, info = env.step(residual_actions)

    return {
        "metric": env.rollout_metric(),
        "collisions": info["collision_count"],
        "steps": info["steps"],
        "destinations_reached": info["destinations_reached"],
        "mean_step_reward": float(np.mean(rewards)) if rewards else 0.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--calibration", type=Path, default=DEFAULT_CALIBRATION_PATH)
    parser.add_argument("--prefer-params", choices=("robust", "best"), default="robust")
    parser.add_argument("--checkpoint", type=Path, default=Path("RL/checkpoints/v2/residual_policy.pt"))
    args = parser.parse_args()

    base_params = None
    if args.calibration.exists():
        base_params = load_base_params(args.calibration, prefer=args.prefer_params)
        print(f"Theta_base = {args.prefer_params} from {args.calibration}")

    env_cfg = EnvConfig(base_params=base_params)
    baseline_env = MultiAgentTrafficEnv(env_cfg, seed=args.seed)
    baseline = run_rollout(baseline_env, policy=None)
    print("=== Utility-only baseline ===")
    print(baseline)

    if torch is None or not args.checkpoint.exists():
        print("\nNo trained checkpoint found. Run: python -m RL.train_ppo")
        return

    try:
        payload = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    except TypeError:
        payload = torch.load(args.checkpoint, map_location="cpu")
    from Baselines.residual_marl import load_residual_policy
    policy = load_residual_policy(args.checkpoint, payload["obs_dim"], allow_legacy=True)

    residual_env = MultiAgentTrafficEnv(env_cfg, seed=args.seed)
    residual = run_rollout(residual_env, policy=policy, explore_std=0.0)
    print("\n=== Residual policy rollout ===")
    print(residual)


if __name__ == "__main__":
    main()
