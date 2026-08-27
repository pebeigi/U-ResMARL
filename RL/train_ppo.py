#!/usr/bin/env python
"""Train residual utility policy with custom shared-policy PPO (primary trainer).

Run from repo root:
  python -m RL.train_ppo --calibration Calibration/utility_calibration.json

RLlib alternative (optional, needs ray[rllib] + dm_tree):
  python -m RL.train_rllib
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np

import RL._paths  # noqa: F401
from RL.calibration_io import (
    DEFAULT_CALIBRATION_PATH,
    DEFAULT_RESIDUAL_SCALES,
    LEGACY_RESIDUAL_SCALES,
    PARAM_GAUGE,
    RESIDUAL_PARAM_KEYS,
    load_base_params,
    residual_scales_for_checkpoint,
)
from RL.param_gauge import LEGACY_RESIDUAL_PARAM_KEYS
from RL.traffic_env import EnvConfig, MultiAgentTrafficEnv

try:
    import torch
    import torch.nn as nn
except ImportError as exc:
    raise SystemExit("PyTorch is required for training. Install with: pip install torch") from exc


def normalize_obs(obs: torch.Tensor, highway_length: float = 500.0) -> torch.Tensor:
    """Scale Frenet / body-frame features into a PPO-friendly range."""
    obs = obs.clone()
    length = max(float(highway_length), 1.0)
    obs[..., 0] = obs[..., 0] / length
    obs[..., 1] = obs[..., 1] / 8.0
    obs[..., 2] = obs[..., 2] / 16.0
    obs[..., 3] = obs[..., 3] / np.pi
    obs[..., 4] = obs[..., 4] / np.pi
    obs[..., 5] = obs[..., 5] / 12.0
    obs[..., 6] = obs[..., 6] / 12.0

    for start in range(7, obs.shape[-1], 4):
        obs[..., start] = obs[..., start] / 60.0
        obs[..., start + 1] = obs[..., start + 1] / 12.0
        obs[..., start + 2] = obs[..., start + 2] / 16.0
        obs[..., start + 3] = obs[..., start + 3] / 16.0
    return obs


class TorchResidualPolicy(nn.Module):
    """Shared actor-critic pi(o)->Delta with tanh-bounded Gaussian actor.

    Default gauge (``logit_simplex``): five amplitude logits ``z_*`` plus four
    additive shape deltas (``xi_i``, ``gamma``, ``sigma_long``, ``sigma_lat``).
    """

    def __init__(
        self,
        obs_dim: int,
        hidden_dim: int = 128,
        residual_scales: dict[str, float] | None = None,
        highway_length: float = 500.0,
        *,
        param_gauge: str = PARAM_GAUGE,
    ):
        super().__init__()
        self.obs_dim = obs_dim
        self.hidden_dim = hidden_dim
        self.param_gauge = str(param_gauge)
        self._residual_keys = (
            LEGACY_RESIDUAL_PARAM_KEYS if self.param_gauge == "additive" else RESIDUAL_PARAM_KEYS
        )
        scales = residual_scales or (
            LEGACY_RESIDUAL_SCALES if self.param_gauge == "additive" else DEFAULT_RESIDUAL_SCALES
        )
        scale_vec = torch.tensor([float(scales[k]) for k in self._residual_keys], dtype=torch.float32)
        self.register_buffer("residual_scales", scale_vec)
        self.register_buffer("highway_length", torch.tensor(float(highway_length)))
        # Kept for older checkpoint / visualize_simulation compatibility.
        self.residual_scale = float(scale_vec.mean().item())

        self.actor = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, len(self._residual_keys)),
            nn.Tanh(),
        )
        self.critic = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )
        self.log_std = nn.Parameter(torch.full((len(self._residual_keys),), -1.2))

    def forward(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        obs_n = normalize_obs(obs, float(self.highway_length.item()))
        mean = self.actor(obs_n) * self.residual_scales
        value = self.critic(obs_n).squeeze(-1)
        return mean, value

    def distribution(self, obs: torch.Tensor) -> tuple[torch.distributions.Normal, torch.Tensor]:
        mean, value = self.forward(obs)
        std = torch.exp(self.log_std).expand_as(mean)
        return torch.distributions.Normal(mean, std), value

    def action_to_dict(self, action: np.ndarray) -> dict[str, float]:
        scales = self.residual_scales.detach().cpu().numpy()
        clipped = np.clip(action, -scales, scales)
        return dict(zip(self._residual_keys, clipped.astype(float)))

    def act(self, obs: np.ndarray, explore_std: float = 0.0) -> tuple[dict[str, float], torch.Tensor]:
        """Compatibility method used by visualization/demo scripts."""
        obs_t = torch.as_tensor(obs, dtype=torch.float32)
        with torch.no_grad():
            mean, _ = self.forward(obs_t)
            if explore_std > 0:
                std = torch.full_like(mean, explore_std)
                action = torch.distributions.Normal(mean, std).sample()
                log_prob = torch.distributions.Normal(mean, std).log_prob(action).sum()
            else:
                action = mean
                log_prob = torch.zeros(())
        return self.action_to_dict(action.cpu().numpy()), log_prob

    def sample_action(self, obs: np.ndarray) -> tuple[dict[str, float], np.ndarray, float, float]:
        obs_t = torch.as_tensor(obs, dtype=torch.float32)
        with torch.no_grad():
            dist, value = self.distribution(obs_t)
            action_t = dist.sample()
            log_prob_t = dist.log_prob(action_t).sum()
        action_np = action_t.cpu().numpy()
        return self.action_to_dict(action_np), action_np, float(log_prob_t), float(value)


@dataclass
class PPOMemory:
    observations: list[np.ndarray]
    actions: list[np.ndarray]
    log_probs: list[float]
    values: list[float]
    rewards: list[float]
    dones: list[float]


def collect_rollouts(
    env: MultiAgentTrafficEnv,
    policy: TorchResidualPolicy,
    episodes_per_update: int,
) -> tuple[PPOMemory, list[float], list[int], list[float]]:
    memory = PPOMemory([], [], [], [], [], [])
    metrics: list[float] = []
    collisions: list[int] = []
    realism: list[float] = []

    for _ in range(episodes_per_update):
        obs_list = env.reset()
        done = False
        while not done:
            residual_actions = []
            step_transition_indices: list[int] = []
            for obs in obs_list:
                action_dict, action_np, log_prob, value = policy.sample_action(obs)
                residual_actions.append(action_dict)
                memory.observations.append(obs)
                memory.actions.append(action_np)
                memory.log_probs.append(log_prob)
                memory.values.append(value)
                step_transition_indices.append(len(memory.rewards))

            obs_list, rewards, done, info = env.step(residual_actions)
            for reward, _ in zip(rewards, step_transition_indices):
                memory.rewards.append(float(reward))
                memory.dones.append(float(done))

        metrics.append(env.rollout_metric())
        collisions.append(env.collision_count)
        realism.append(float(info.get("realism_distance", float("nan"))))

    return memory, metrics, collisions, realism


def compute_gae(
    rewards: np.ndarray,
    values: np.ndarray,
    dones: np.ndarray,
    gamma: float,
    gae_lambda: float,
) -> tuple[np.ndarray, np.ndarray]:
    advantages = np.zeros_like(rewards, dtype=np.float32)
    last_gae = 0.0
    next_value = 0.0
    for t in reversed(range(len(rewards))):
        nonterminal = 1.0 - dones[t]
        delta = rewards[t] + gamma * next_value * nonterminal - values[t]
        last_gae = delta + gamma * gae_lambda * nonterminal * last_gae
        advantages[t] = last_gae
        next_value = values[t]
    returns = advantages + values
    return advantages, returns


def ppo_update(
    policy: TorchResidualPolicy,
    optimizer: torch.optim.Optimizer,
    memory: PPOMemory,
    gamma: float,
    gae_lambda: float,
    clip_coef: float,
    value_coef: float,
    entropy_coef: float,
    epochs: int,
    minibatch_size: int,
) -> dict[str, float]:
    obs = torch.as_tensor(np.array(memory.observations), dtype=torch.float32)
    actions = torch.as_tensor(np.array(memory.actions), dtype=torch.float32)
    old_log_probs = torch.as_tensor(np.array(memory.log_probs), dtype=torch.float32)
    values_np = np.array(memory.values, dtype=np.float32)
    rewards_np = np.array(memory.rewards, dtype=np.float32)
    dones_np = np.array(memory.dones, dtype=np.float32)

    advantages_np, returns_np = compute_gae(rewards_np, values_np, dones_np, gamma, gae_lambda)
    advantages = torch.as_tensor(advantages_np, dtype=torch.float32)
    returns = torch.as_tensor(returns_np, dtype=torch.float32)
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    n = obs.shape[0]
    indices = np.arange(n)
    last_stats = {"loss": 0.0, "policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0}
    for _ in range(epochs):
        np.random.shuffle(indices)
        for start in range(0, n, minibatch_size):
            batch_idx = indices[start : start + minibatch_size]
            dist, value = policy.distribution(obs[batch_idx])
            new_log_probs = dist.log_prob(actions[batch_idx]).sum(dim=-1)
            entropy = dist.entropy().sum(dim=-1).mean()

            log_ratio = new_log_probs - old_log_probs[batch_idx]
            ratio = torch.exp(log_ratio)
            unclipped = ratio * advantages[batch_idx]
            clipped = torch.clamp(ratio, 1 - clip_coef, 1 + clip_coef) * advantages[batch_idx]
            policy_loss = -torch.min(unclipped, clipped).mean()
            value_loss = 0.5 * (returns[batch_idx] - value).pow(2).mean()
            loss = policy_loss + value_coef * value_loss - entropy_coef * entropy

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(policy.parameters(), max_norm=0.5)
            optimizer.step()

            last_stats = {
                "loss": float(loss.detach()),
                "policy_loss": float(policy_loss.detach()),
                "value_loss": float(value_loss.detach()),
                "entropy": float(entropy.detach()),
            }
    return last_stats


def make_env(args: argparse.Namespace, seed: int) -> MultiAgentTrafficEnv:
    base_params = None
    if args.calibration is not None:
        base_params = load_base_params(args.calibration, prefer=args.prefer_params)
    cfg_kwargs = dict(
        max_steps=args.max_steps,
        num_agents=args.num_agents,
        base_params=base_params,
        collision_penalty=float(getattr(args, "collision_penalty", 0.0)),
        behavior_coef=float(getattr(args, "behavior_coef", 0.0)),
        behavior_shaping_coef=float(getattr(args, "behavior_shaping_coef", 0.0)),
    )
    # Optional denser packing so collision-aware residuals see crowded traffic.
    if getattr(args, "dense_spawn", False):
        cfg_kwargs["spawn_s_range"] = (20.0, 80.0)
        cfg_kwargs["spawn_lateral_frac"] = 0.55
        cfg_kwargs["min_initial_spacing"] = 5.0
    cfg = EnvConfig(**cfg_kwargs)
    return MultiAgentTrafficEnv(cfg, seed=seed)


def baseline_metric(args: argparse.Namespace) -> float:
    metrics = []
    for i in range(max(1, args.baseline_episodes)):
        env = make_env(args, seed=args.seed + i)
        env.reset()
        done = False
        while not done:
            _, _, done, _ = env.step(None)
        metrics.append(env.rollout_metric())
    return float(np.mean(metrics))


def train(args: argparse.Namespace) -> None:
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    probe_env = make_env(args, seed=args.seed)
    policy = TorchResidualPolicy(
        probe_env.obs_dim,
        hidden_dim=args.hidden_dim,
        residual_scales=DEFAULT_RESIDUAL_SCALES,
        highway_length=float(probe_env.config.highway_length),
    )
    optimizer = torch.optim.Adam(policy.parameters(), lr=args.lr)

    if args.calibration is not None:
        print(f"Using {args.prefer_params} utility params from {args.calibration}")
    if args.collision_penalty > 0.0:
        print(f"OBB collision penalty = {args.collision_penalty:.2f} per colliding agent-step")
    if args.behavior_coef > 0.0 or args.behavior_shaping_coef > 0.0:
        print(
            f"Behavioral data term: band={args.behavior_coef:.2f}, "
            f"shaping={args.behavior_shaping_coef:.2f}"
        )
    base = baseline_metric(args)
    print(f"Utility-only baseline metric over {args.baseline_episodes} episodes: {base:.3f}")

    best_metric = float("inf")
    best_collisions = float("inf")
    best_state = None
    for update in range(1, args.updates + 1):
        env = make_env(args, seed=args.seed + update * 1000)
        memory, metrics, collisions, realism = collect_rollouts(
            env, policy, args.episodes_per_update
        )
        stats = ppo_update(
            policy,
            optimizer,
            memory,
            gamma=args.gamma,
            gae_lambda=args.gae_lambda,
            clip_coef=args.clip_coef,
            value_coef=args.value_coef,
            entropy_coef=args.entropy_coef,
            epochs=args.ppo_epochs,
            minibatch_size=args.minibatch_size,
        )

        mean_metric = float(np.mean(metrics))
        mean_collisions = float(np.mean(collisions))
        if mean_collisions < best_collisions - 1e-9 or (
            abs(mean_collisions - best_collisions) < 1e-9 and mean_metric < best_metric
        ):
            best_collisions = mean_collisions
            best_state = {k: v.detach().cpu().clone() for k, v in policy.state_dict().items()}
        best_metric = min(best_metric, mean_metric)
        log_every = args.log_every if args.log_every > 0 else max(min(args.updates // 10, 10), 1)
        if update == 1 or update % log_every == 0:
            realism_note = ""
            if np.any(np.isfinite(realism)):
                realism_note = f" | realism={np.nanmean(realism):5.3f}"
            print(
                f"Update {update:4d}/{args.updates} | metric={mean_metric:8.3f} | "
                f"best={best_metric:8.3f} | collisions={np.mean(collisions):5.2f}{realism_note} | "
                f"loss={stats['loss']:8.3f} | entropy={stats['entropy']:6.3f}"
            )

    if args.save is not None:
        args.save.parent.mkdir(parents=True, exist_ok=True)
        if best_state is not None:
            policy.load_state_dict(best_state)
        blob = {
            "state_dict": policy.state_dict(),
            "obs_dim": probe_env.obs_dim,
            "hidden_dim": args.hidden_dim,
            "residual_scale": policy.residual_scale,
            "residual_scales": {k: float(DEFAULT_RESIDUAL_SCALES[k]) for k in RESIDUAL_PARAM_KEYS},
            "param_gauge": PARAM_GAUGE,
            "highway_length": float(probe_env.config.highway_length),
            "algo": "ppo",
            "prefer_params": args.prefer_params,
            "calibration": str(args.calibration) if args.calibration else None,
            "collision_penalty": float(args.collision_penalty),
            "behavior_coef": float(args.behavior_coef),
            "behavior_shaping_coef": float(args.behavior_shaping_coef),
            "updates": int(args.updates),
            "best_collisions": float(best_collisions),
        }
        torch.save(blob, args.save)
        print(
            f"Saved PPO residual policy to {args.save} "
            f"(best train collisions={best_collisions:.2f})"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Train residual MARL policy with PPO")
    parser.add_argument("--updates", type=int, default=100)
    parser.add_argument("--episodes-per-update", type=int, default=4)
    parser.add_argument("--num-agents", type=int, default=10)
    parser.add_argument("--max-steps", type=int, default=240)
    parser.add_argument("--baseline-episodes", type=int, default=5)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--clip-coef", type=float, default=0.2)
    parser.add_argument("--value-coef", type=float, default=0.5)
    parser.add_argument("--entropy-coef", type=float, default=0.01)
    parser.add_argument("--ppo-epochs", type=int, default=4)
    parser.add_argument("--minibatch-size", type=int, default=512)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--calibration",
        type=Path,
        default=DEFAULT_CALIBRATION_PATH,
        help="Calibration JSON; uses robust_params by default",
    )
    parser.add_argument(
        "--prefer-params",
        choices=("robust", "best"),
        default="robust",
        help="Which calibrated parameter set to freeze as Θ_base",
    )
    parser.add_argument(
        "--save",
        type=Path,
        default=Path("RL/checkpoints/residual_policy.pt"),
    )
    parser.add_argument(
        "--collision-penalty",
        type=float,
        default=8.0,
        help="Per-step reward penalty for each agent involved in an OBB collision",
    )
    parser.add_argument(
        "--dense-spawn",
        action="store_true",
        help="Pack agents into a shorter spawn window (stress-like training distribution)",
    )
    parser.add_argument(
        "--behavior-coef",
        type=float,
        default=0.0,
        help="Weight of the dense out-of-band penalty on measured speed/accel/lateral marginals",
    )
    parser.add_argument(
        "--behavior-shaping-coef",
        type=float,
        default=0.0,
        help="Weight of potential-based shaping on the distance to the measured marginals",
    )
    parser.add_argument(
        "--log-every",
        type=int,
        default=0,
        help="Print every N updates (0 = auto: min(10, updates/10))",
    )
    args = parser.parse_args()
    if args.calibration is not None and not args.calibration.exists():
        print(f"Warning: calibration file missing ({args.calibration}); using DEFAULT_BASE_PARAMS")
        args.calibration = None
    train(args)


if __name__ == "__main__":
    main()
