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
    """Scale Frenet / body-frame features into a PPO-friendly range.

    Supports both the legacy 7-ego layout and the current 8-ego layout (with
    remaining station).
    """
    from RL.obs import ego_feature_count

    obs = obs.clone()
    length = max(float(highway_length), 1.0)
    ego_dim = ego_feature_count(int(obs.shape[-1]))
    obs[..., 0] = obs[..., 0] / length
    obs[..., 1] = obs[..., 1] / 8.0
    obs[..., 2] = obs[..., 2] / 16.0
    obs[..., 3] = obs[..., 3] / np.pi
    obs[..., 4] = obs[..., 4] / np.pi
    obs[..., 5] = obs[..., 5] / 12.0
    obs[..., 6] = obs[..., 6] / 12.0
    if ego_dim >= 8:
        obs[..., 7] = obs[..., 7] / length
        neighbor_start = 8
    else:
        neighbor_start = 7

    for start in range(neighbor_start, obs.shape[-1], 4):
        obs[..., start] = obs[..., start] / 60.0
        obs[..., start + 1] = obs[..., start + 1] / 12.0
        obs[..., start + 2] = obs[..., start + 2] / 16.0
        obs[..., start + 3] = obs[..., start + 3] / 16.0
    return obs


#: Action-space variants, recorded in the checkpoint so older policies replay
#: exactly as they were trained.
#:   ``legacy``          -- Gaussian directly in physical Delta units, clipped by
#:                          the environment but scored unclipped.
#:   ``normalized_tanh`` -- Gaussian in [-1, 1]^d, sample clipped then scored.
#:   ``squashed_tanh``   -- Gaussian in R^d squashed by tanh into [-1, 1]^d with
#:                          the change-of-variables correction (current default).
LEGACY_ACTION_SPACE = "legacy"
NORMALIZED_ACTION_SPACE = "normalized_tanh"
SQUASHED_ACTION_SPACE = "squashed_tanh"
DEFAULT_ACTION_SPACE = SQUASHED_ACTION_SPACE

#: Residual interfaces.
#:   ``candidate_logits`` -- additive residual on the discrete utility grid (default).
#:   ``param_delta``      -- residual on utility parameters Θ (ablation / legacy).
RESIDUAL_MODE_CANDIDATE = "candidate_logits"
RESIDUAL_MODE_PARAM = "param_delta"
DEFAULT_RESIDUAL_MODE = RESIDUAL_MODE_CANDIDATE
CANDIDATE_LOGIT_SCALE = 2.0

#: Floor inside log(1 - tanh(u)^2) so the squash correction stays finite.
_SQUASH_EPS = 1e-6


def action_space_from_blob(blob: dict) -> str:
    """Action-space variant a checkpoint was trained with."""
    return str(blob.get("action_space", LEGACY_ACTION_SPACE))


def residual_mode_from_blob(blob: dict) -> str:
    return str(blob.get("residual_mode", RESIDUAL_MODE_PARAM))


from RL.value_normalization import ValueNormalizer


class TorchResidualPolicy(ValueNormalizer):
    """Shared actor-critic with a bounded Gaussian actor.

    Default residual interface (``candidate_logits``): the actor emits an additive
    residual over the discrete utility grid so PPO credit assignment is continuous
    in the executed control.  The legacy ``param_delta`` interface edits Θ instead.
    """

    def __init__(
        self,
        obs_dim: int,
        hidden_dim: int = 128,
        residual_scales: dict[str, float] | None = None,
        highway_length: float = 500.0,
        *,
        param_gauge: str = PARAM_GAUGE,
        action_space: str = DEFAULT_ACTION_SPACE,
        residual_mode: str = DEFAULT_RESIDUAL_MODE,
        action_dim: int | None = None,
        candidate_logit_scale: float = CANDIDATE_LOGIT_SCALE,
    ):
        super().__init__()
        self.obs_dim = obs_dim
        self.hidden_dim = hidden_dim
        self.param_gauge = str(param_gauge)
        self.action_space = str(action_space)
        self.residual_mode = str(residual_mode)
        self._residual_keys = (
            LEGACY_RESIDUAL_PARAM_KEYS if self.param_gauge == "additive" else RESIDUAL_PARAM_KEYS
        )
        if self.residual_mode == RESIDUAL_MODE_CANDIDATE:
            dim = int(action_dim if action_dim is not None else 63)
            scale_vec = torch.full((dim,), float(candidate_logit_scale), dtype=torch.float32)
            self._residual_keys = tuple(f"c{i}" for i in range(dim))
        else:
            scales = residual_scales or (
                LEGACY_RESIDUAL_SCALES if self.param_gauge == "additive" else DEFAULT_RESIDUAL_SCALES
            )
            scale_vec = torch.tensor(
                [float(scales[k]) for k in self._residual_keys], dtype=torch.float32
            )
        self.register_buffer("residual_scales", scale_vec)
        self.register_buffer("highway_length", torch.tensor(float(highway_length)))
        # Running statistics of the return, used to normalize the critic target.
        # Kept for older checkpoint / visualize_simulation compatibility.
        self.residual_scale = float(scale_vec.mean().item())

        actor_layers: list[nn.Module] = [
            nn.Linear(obs_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, int(scale_vec.numel())),
        ]
        if self.action_space != SQUASHED_ACTION_SPACE:
            actor_layers.append(nn.Tanh())
        self.actor = nn.Sequential(*actor_layers)
        self.critic = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )
        self.log_std = nn.Parameter(torch.full((int(scale_vec.numel()),), -1.2))
        if self.action_space != LEGACY_ACTION_SPACE:
            head = self.actor[4]
            nn.init.zeros_(head.weight)
            nn.init.zeros_(head.bias)

    # -- policy ---------------------------------------------------------------

    def action_bound(self) -> torch.Tensor:
        """Elementwise bound on the emitted action, before residual scaling."""
        if self.action_space == LEGACY_ACTION_SPACE:
            return self.residual_scales
        return torch.ones_like(self.residual_scales)

    def evaluate(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Distribution mean and the critic's *normalized* value prediction."""
        obs_n = normalize_obs(obs, float(self.highway_length.item()))
        mean = self.actor(obs_n)
        if self.action_space == LEGACY_ACTION_SPACE:
            mean = mean * self.residual_scales
        return mean, self.critic(obs_n).squeeze(-1)

    def forward(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mean, value_norm = self.evaluate(obs)
        return mean, self.denormalize_value(value_norm)

    def _stddev(self, mean: torch.Tensor) -> torch.Tensor:
        # Keep sigma in (e^{-5}, 1] so the pre-squash Gaussian cannot walk off
        # to infinity; that was the remaining source of entropy inflation.
        return torch.exp(self.log_std.clamp(-5.0, 0.0)).expand_as(mean)

    def distribution(self, obs: torch.Tensor) -> tuple[torch.distributions.Normal, torch.Tensor]:
        mean, value = self.forward(obs)
        return torch.distributions.Normal(mean, self._stddev(mean)), value

    def log_prob(self, obs: torch.Tensor, action: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Log-density of ``action`` plus the Gaussian entropy and normalized value.

        For the squashed policy ``action`` is the pre-squash sample ``u``; the
        tanh change of variables is subtracted so the density refers to the
        bounded residual that was actually executed.
        """
        mean, value_norm = self.evaluate(obs)
        std = self._stddev(mean)
        dist = torch.distributions.Normal(mean, std)
        log_prob = dist.log_prob(action).sum(dim=-1)
        if self.action_space == SQUASHED_ACTION_SPACE:
            log_prob = log_prob - torch.log(1.0 - torch.tanh(action).pow(2) + _SQUASH_EPS).sum(dim=-1)
        return log_prob, dist.entropy().sum(dim=-1), value_norm

    def _to_delta(self, action: torch.Tensor) -> torch.Tensor:
        if self.action_space == SQUASHED_ACTION_SPACE:
            return torch.tanh(action) * self.residual_scales
        bound = self.action_bound()
        bounded = torch.clamp(action, -bound, bound)
        if self.action_space == NORMALIZED_ACTION_SPACE:
            bounded = bounded * self.residual_scales
        return bounded

    def action_to_env(self, action: np.ndarray):
        """Convert a policy sample into the object ``MultiAgentTrafficEnv.step`` expects."""
        action_t = torch.as_tensor(np.asarray(action), dtype=torch.float32)
        delta = self._to_delta(action_t).detach().cpu().numpy().astype(float)
        if self.residual_mode == RESIDUAL_MODE_CANDIDATE:
            return delta
        return dict(zip(self._residual_keys, delta))

    def action_to_dict(self, action: np.ndarray) -> dict[str, float]:
        """Compatibility wrapper; candidate-logit mode returns indexed ``c*`` keys."""
        env_action = self.action_to_env(action)
        if isinstance(env_action, dict):
            return env_action
        return {f"c{i}": float(v) for i, v in enumerate(env_action)}

    def act(self, obs: np.ndarray, explore_std: float = 0.0):
        """Deterministic (mean) residual; used by evaluation and visualization."""
        from RL.obs import adapt_observation

        obs_t = torch.as_tensor(adapt_observation(obs, self.obs_dim), dtype=torch.float32)
        with torch.no_grad():
            mean, _ = self.evaluate(obs_t)
            if explore_std > 0:
                std = torch.full_like(mean, explore_std)
                dist = torch.distributions.Normal(mean, std)
                action = dist.sample()
                log_prob = dist.log_prob(action).sum()
            else:
                action = mean
                log_prob = torch.zeros(())
        return self.action_to_env(action.cpu().numpy()), log_prob

    def sample_action(self, obs: np.ndarray):
        """Sample the action the environment will execute, and score that action."""
        obs_t = torch.as_tensor(obs, dtype=torch.float32)
        with torch.no_grad():
            mean, value_norm = self.evaluate(obs_t)
            std = self._stddev(mean)
            action_t = torch.distributions.Normal(mean, std).sample()
            if self.action_space != SQUASHED_ACTION_SPACE:
                bound = self.action_bound()
                action_t = torch.clamp(action_t, -bound, bound)
            log_prob_t, _, _ = self.log_prob(obs_t, action_t)
            value = self.denormalize_value(value_norm)
        action_np = action_t.cpu().numpy()
        return self.action_to_env(action_np), action_np, float(log_prob_t), float(value)


@dataclass
class PPOMemory:
    observations: list[np.ndarray]
    actions: list[np.ndarray]
    log_probs: list[float]
    values: list[float]
    rewards: list[float]
    dones: list[float]
    #: 1.0 when the episode hit max_steps without full arrival (bootstrap V).
    timeouts: list[float]
    #: V(s_{t+1}) used when timeouts[t]=1; otherwise unused.
    bootstrap_values: list[float]
    #: Identifies the (episode, agent) trajectory a transition belongs to.
    traj_ids: list[int]


def collect_rollouts(
    env: MultiAgentTrafficEnv,
    policy: TorchResidualPolicy,
    episodes_per_update: int,
) -> tuple[PPOMemory, list[float], list[int], list[float], dict[str, float]]:
    memory = PPOMemory([], [], [], [], [], [], [], [], [])
    metrics: list[float] = []
    collisions: list[int] = []
    realism: list[float] = []
    flips = 0
    decisions = 0
    num_agents = len(env.agents) if env.agents else env.config.num_agents

    for episode in range(episodes_per_update):
        obs_list = env.reset()
        num_agents = len(env.agents)
        done = False
        info: dict = {}
        while not done:
            active = [i for i, a in enumerate(env.agents) if not a.reached_destination]
            residual_actions: list = [None for _ in env.agents]
            for i in active:
                env_action, action_np, log_prob, value = policy.sample_action(obs_list[i])
                residual_actions[i] = env_action
                memory.observations.append(obs_list[i])
                memory.actions.append(action_np)
                memory.log_probs.append(log_prob)
                memory.values.append(value)
                memory.traj_ids.append(episode * num_agents + i)

            obs_list, rewards, done, info = env.step(residual_actions)
            truncated = bool(info.get("truncated", False))
            flips += int(info.get("control_flips", 0))
            decisions += int(info.get("control_decisions", 0))
            for i in active:
                memory.rewards.append(float(rewards[i]))
                agent_terminal = bool(env.agents[i].reached_destination) or (
                    done and not truncated
                )
                memory.dones.append(float(agent_terminal))
                is_timeout = bool(truncated and not env.agents[i].reached_destination)
                memory.timeouts.append(float(is_timeout))
                boot = 0.0
                if is_timeout:
                    with torch.no_grad():
                        _, v_next = policy.forward(
                            torch.as_tensor(obs_list[i], dtype=torch.float32)
                        )
                    boot = float(v_next)
                memory.bootstrap_values.append(boot)

        metrics.append(env.rollout_metric())
        collisions.append(env.collision_count)
        realism.append(float(info.get("realism_distance", float("nan"))))

    aux = {
        "control_flip_rate": float(flips / decisions) if decisions else 0.0,
        "control_flips": float(flips),
        "control_decisions": float(decisions),
    }
    return memory, metrics, collisions, realism, aux


def compute_gae(
    rewards: np.ndarray,
    values: np.ndarray,
    dones: np.ndarray,
    gamma: float,
    gae_lambda: float,
    timeouts: np.ndarray | None = None,
    bootstrap_values: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """GAE along a single contiguous trajectory.

    ``timeouts[t]=1`` means step ``t`` ended by horizon truncation: bootstrap with
    ``bootstrap_values[t]`` (usually ``values[t]`` as a stand-in for ``V(s_{t+1})``
    when the next value was not stored) instead of treating the episode as terminal.
    """
    advantages = np.zeros_like(rewards, dtype=np.float32)
    last_gae = 0.0
    next_value = 0.0
    timeouts_arr = (
        np.zeros_like(rewards, dtype=np.float32)
        if timeouts is None
        else np.asarray(timeouts, dtype=np.float32)
    )
    boot_arr = (
        np.asarray(values, dtype=np.float32)
        if bootstrap_values is None
        else np.asarray(bootstrap_values, dtype=np.float32)
    )
    for t in reversed(range(len(rewards))):
        if timeouts_arr[t] > 0.5:
            # Truncation: not a true terminal; bootstrap with the critic.
            nonterminal = 1.0
            next_value = float(boot_arr[t])
        else:
            nonterminal = 1.0 - float(dones[t])
        delta = rewards[t] + gamma * next_value * nonterminal - values[t]
        last_gae = delta + gamma * gae_lambda * nonterminal * last_gae
        advantages[t] = last_gae
        next_value = values[t]
    returns = advantages + values
    return advantages, returns


def compute_gae_by_trajectory(
    rewards: np.ndarray,
    values: np.ndarray,
    traj_ids: np.ndarray,
    gamma: float,
    gae_lambda: float,
    dones: np.ndarray | None = None,
    timeouts: np.ndarray | None = None,
    bootstrap_values: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """GAE applied separately to every (episode, agent) trajectory in the buffer."""
    rewards = np.asarray(rewards, dtype=np.float32)
    values = np.asarray(values, dtype=np.float32)
    advantages = np.zeros_like(rewards, dtype=np.float32)
    dones_arr = (
        np.zeros_like(rewards, dtype=np.float32)
        if dones is None
        else np.asarray(dones, dtype=np.float32)
    )
    timeouts_arr = (
        np.zeros_like(rewards, dtype=np.float32)
        if timeouts is None
        else np.asarray(timeouts, dtype=np.float32)
    )
    boot_arr = (
        np.asarray(values, dtype=np.float32)
        if bootstrap_values is None
        else np.asarray(bootstrap_values, dtype=np.float32)
    )

    positions: dict[int, list[int]] = {}
    for pos, traj in enumerate(np.asarray(traj_ids)):
        positions.setdefault(int(traj), []).append(pos)

    for idx_list in positions.values():
        idx = np.asarray(idx_list, dtype=int)
        traj_adv, _ = compute_gae(
            rewards[idx],
            values[idx],
            dones_arr[idx],
            gamma,
            gae_lambda,
            timeouts=timeouts_arr[idx],
            bootstrap_values=boot_arr[idx],
        )
        advantages[idx] = traj_adv

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
    target_kl: float = 0.02,
) -> dict[str, float]:
    obs = torch.as_tensor(np.array(memory.observations), dtype=torch.float32)
    actions = torch.as_tensor(np.array(memory.actions), dtype=torch.float32)
    old_log_probs = torch.as_tensor(np.array(memory.log_probs), dtype=torch.float32)
    values_np = np.array(memory.values, dtype=np.float32)
    rewards_np = np.array(memory.rewards, dtype=np.float32)
    traj_np = np.array(memory.traj_ids, dtype=np.int64)
    dones_np = np.array(memory.dones, dtype=np.float32)
    timeouts_np = np.array(memory.timeouts, dtype=np.float32)
    boot_np = np.array(memory.bootstrap_values, dtype=np.float32)

    advantages_np, returns_np = compute_gae_by_trajectory(
        rewards_np,
        values_np,
        traj_np,
        gamma,
        gae_lambda,
        dones=dones_np,
        timeouts=timeouts_np,
        bootstrap_values=boot_np,
    )
    # How much of the return variation the critic actually explains.  Advantages
    # are only informative when this is well above zero.
    residual_var = float(np.var(returns_np - values_np))
    return_var = float(np.var(returns_np))
    explained_variance = 1.0 - residual_var / return_var if return_var > 1e-8 else 0.0

    advantages = torch.as_tensor(advantages_np, dtype=torch.float32)
    returns = torch.as_tensor(returns_np, dtype=torch.float32)
    advantages = (advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1e-8)

    policy.update_value_stats(returns)
    value_targets = policy.normalize_return(returns)

    n = obs.shape[0]
    indices = np.arange(n)
    last_stats = {
        "loss": 0.0,
        "policy_loss": 0.0,
        "value_loss": 0.0,
        "entropy": 0.0,
        "approx_kl": 0.0,
        "clip_frac": 0.0,
        "explained_variance": explained_variance,
        "epochs_run": 0.0,
    }
    for epoch in range(epochs):
        np.random.shuffle(indices)
        epoch_kl: list[float] = []
        epoch_clip: list[float] = []
        for start in range(0, n, minibatch_size):
            batch_idx = indices[start : start + minibatch_size]
            new_log_probs, entropy_terms, value_norm = policy.log_prob(
                obs[batch_idx], actions[batch_idx]
            )
            entropy = entropy_terms.mean()

            log_ratio = new_log_probs - old_log_probs[batch_idx]
            ratio = torch.exp(log_ratio)
            unclipped = ratio * advantages[batch_idx]
            clipped = torch.clamp(ratio, 1 - clip_coef, 1 + clip_coef) * advantages[batch_idx]
            policy_loss = -torch.min(unclipped, clipped).mean()
            value_loss = 0.5 * (value_targets[batch_idx] - value_norm).pow(2).mean()
            loss = policy_loss + value_coef * value_loss - entropy_coef * entropy

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(policy.parameters(), max_norm=0.5)
            optimizer.step()
            with torch.no_grad():
                policy.log_std.clamp_(-5.0, 0.0)

            with torch.no_grad():
                epoch_kl.append(float(((ratio - 1.0) - log_ratio).mean()))
                epoch_clip.append(float((ratio - 1.0).abs().gt(clip_coef).float().mean()))

            last_stats.update(
                loss=float(loss.detach()),
                policy_loss=float(policy_loss.detach()),
                value_loss=float(value_loss.detach()),
                entropy=float(entropy.detach()),
            )

        mean_kl = float(np.mean(epoch_kl)) if epoch_kl else 0.0
        last_stats["approx_kl"] = mean_kl
        last_stats["clip_frac"] = float(np.mean(epoch_clip)) if epoch_clip else 0.0
        last_stats["epochs_run"] = float(epoch + 1)
        # Stop before the policy walks outside the region the rollout supports.
        if target_kl > 0.0 and mean_kl > target_kl:
            break

    return last_stats


def make_env(
    args: argparse.Namespace,
    seed: int,
    *,
    obb_safety_filter: bool | None = None,
) -> MultiAgentTrafficEnv:
    base_params = None
    if args.calibration is not None:
        base_params = load_base_params(args.calibration, prefer=args.prefer_params)
    # Training turns the hard OBB filter off so residual/collision learning sees a
    # reachable safety signal; evaluation can override this back on.
    if obb_safety_filter is None:
        obb_safety_filter = bool(getattr(args, "train_obb_filter", False))
    cfg_kwargs = dict(
        max_steps=args.max_steps,
        num_agents=args.num_agents,
        base_params=base_params,
        collision_penalty=float(getattr(args, "collision_penalty", 0.0)),
        behavior_coef=float(getattr(args, "behavior_coef", 0.0)),
        behavior_shaping_coef=float(getattr(args, "behavior_shaping_coef", 0.0)),
        residual_mode=str(getattr(args, "residual_mode", DEFAULT_RESIDUAL_MODE)),
        leftover_coef=float(getattr(args, "leftover_coef", 0.05)),
        arrival_bonus=float(getattr(args, "arrival_bonus", 5.0)),
        obb_safety_filter=bool(obb_safety_filter),
    )
    if getattr(args, "dense_spawn", False):
        cfg_kwargs["spawn_s_range"] = (20.0, 80.0)
        cfg_kwargs["spawn_lateral_frac"] = 0.55
        cfg_kwargs["min_initial_spacing"] = 5.0
    cfg = EnvConfig(**cfg_kwargs)
    return MultiAgentTrafficEnv(cfg, seed=seed)


def evaluate_deterministic(
    args: argparse.Namespace,
    policy: TorchResidualPolicy | None,
    seeds: list[int],
    *,
    obb_safety_filter: bool = True,
) -> dict[str, float]:
    """Roll out ``policy`` with mean actions under the benchmark safety setting."""
    metrics: list[float] = []
    collisions: list[float] = []
    arrivals: list[float] = []
    usage: list[float] = []
    flips = 0
    decisions = 0
    scales = (
        policy.residual_scales.detach().cpu().numpy() if policy is not None else None
    )
    for seed in seeds:
        env = make_env(args, seed=int(seed), obb_safety_filter=obb_safety_filter)
        obs_list = env.reset()
        done = False
        info: dict = {}
        while not done:
            actions = None
            if policy is not None:
                actions = [policy.act(obs, 0.0)[0] for obs in obs_list]
                for delta in actions:
                    arr = np.asarray(
                        list(delta.values()) if isinstance(delta, dict) else delta,
                        dtype=float,
                    )
                    usage.append(float(np.mean(np.abs(arr) / scales)))
            obs_list, _, done, info = env.step(actions)
            flips += int(info.get("control_flips", 0))
            decisions += int(info.get("control_decisions", 0))
        metrics.append(float(env.rollout_metric()))
        collisions.append(float(env.collision_count))
        arrivals.append(float(np.mean([bool(a.reached_destination) for a in env.agents])))
    return {
        "metric": float(np.mean(metrics)),
        "collisions": float(np.mean(collisions)),
        "arrival_rate": float(np.mean(arrivals)),
        "residual_usage": float(np.mean(usage)) if usage else 0.0,
        "control_flip_rate": float(flips / decisions) if decisions else 0.0,
    }


def _write_curve(path: Path, rows: list[dict[str, float]]) -> None:
    """Persist the validation trace so the learning curve can be plotted."""
    import csv

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _plot_curve(path: Path, rows: list[dict[str, float]]) -> Path:
    """Write a four-panel PNG next to the CSV: leftover distance, arrival, usage, EV."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    updates = [r["update"] for r in rows]
    fig, axes = plt.subplots(2, 2, figsize=(8.5, 6.0), constrained_layout=True)

    axes[0, 0].plot(updates, [r["val_metric"] for r in rows], "o-", color="#1f77b4", label="residual")
    axes[0, 0].plot(
        updates, [r["prior_val_metric"] for r in rows], "--", color="#7f7f7f", label="utility prior"
    )
    axes[0, 0].set_ylabel("leftover distance (↓)")
    axes[0, 0].legend(frameon=False)

    axes[0, 1].plot(updates, [r["val_arrival_rate"] for r in rows], "o-", color="#2ca02c", label="residual")
    axes[0, 1].plot(
        updates,
        [r["prior_val_arrival_rate"] for r in rows],
        "--",
        color="#7f7f7f",
        label="utility prior",
    )
    axes[0, 1].set_ylabel("arrival rate (↑)")
    axes[0, 1].set_ylim(0.0, 1.05)
    axes[0, 1].legend(frameon=False)

    axes[1, 0].plot(updates, [r["residual_usage"] for r in rows], "o-", color="#ff7f0e", label="|Δ|/scale")
    if any("control_flip_rate" in r for r in rows):
        axes[1, 0].plot(
            updates,
            [r.get("control_flip_rate", 0.0) for r in rows],
            "s-",
            color="#d62728",
            label="control flip rate",
        )
    axes[1, 0].set_ylabel("usage / flip rate")
    axes[1, 0].set_xlabel("update")
    axes[1, 0].legend(frameon=False)

    axes[1, 1].plot(updates, [r["explained_variance"] for r in rows], "o-", color="#9467bd")
    axes[1, 1].axhline(0.0, color="#7f7f7f", lw=0.8)
    axes[1, 1].set_ylabel("critic explained variance")
    axes[1, 1].set_xlabel("update")

    for ax in axes.ravel():
        ax.grid(True, alpha=0.3)

    fig.suptitle("Residual PPO learning curve (deterministic validation)")
    png = path.with_suffix(".png")
    fig.savefig(png, dpi=140)
    plt.close(fig)
    return png


def _curve_row(
    update: int,
    train_metric: float,
    val: dict[str, float],
    prior_val: dict[str, float],
    stats: dict[str, float],
) -> dict[str, float]:
    return {
        "update": float(update),
        "train_metric": float(train_metric),
        "val_metric": val["metric"],
        "val_arrival_rate": val["arrival_rate"],
        "val_collisions": val["collisions"],
        "residual_usage": val.get("residual_usage", 0.0),
        "control_flip_rate": val.get("control_flip_rate", 0.0),
        "prior_val_metric": prior_val["metric"],
        "prior_val_arrival_rate": prior_val["arrival_rate"],
        "explained_variance": stats.get("explained_variance", float("nan")),
        "approx_kl": stats.get("approx_kl", float("nan")),
        "clip_frac": stats.get("clip_frac", float("nan")),
        "entropy": stats.get("entropy", float("nan")),
        "value_loss": stats.get("value_loss", float("nan")),
    }


def _persist_curve(path: Path | None, rows: list[dict[str, float]]) -> None:
    if path is None or not rows:
        return
    _write_curve(path, rows)
    _plot_curve(path, rows)


def _validation_score(stats: dict[str, float]) -> tuple[float, float]:
    """Rank validation points by collisions first, then leftover distance."""
    return (stats["collisions"], stats["metric"])


def baseline_metric(args: argparse.Namespace) -> float:
    seeds = [args.seed + i for i in range(max(1, args.baseline_episodes))]
    return evaluate_deterministic(args, None, seeds)["metric"]


def train(args: argparse.Namespace) -> None:
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    probe_env = make_env(args, seed=args.seed)
    residual_mode = str(getattr(args, "residual_mode", DEFAULT_RESIDUAL_MODE))
    policy = TorchResidualPolicy(
        probe_env.obs_dim,
        hidden_dim=args.hidden_dim,
        residual_scales=DEFAULT_RESIDUAL_SCALES,
        highway_length=float(probe_env.config.highway_length),
        residual_mode=residual_mode,
        action_dim=probe_env.residual_dim,
    )
    optimizer = torch.optim.Adam(policy.parameters(), lr=args.lr)

    if args.calibration is not None:
        print(f"Using {args.prefer_params} utility params from {args.calibration}")
    print(
        f"Residual mode={residual_mode} | action_dim={probe_env.residual_dim} | "
        f"train_obb_filter={bool(getattr(args, 'train_obb_filter', False))}"
    )
    if args.collision_penalty > 0.0:
        print(f"OBB collision penalty = {args.collision_penalty:.2f} per colliding agent-step")
    if args.behavior_coef > 0.0 or args.behavior_shaping_coef > 0.0:
        print(
            f"Behavioral data term: band={args.behavior_coef:.2f}, "
            f"shaping={args.behavior_shaping_coef:.2f}"
        )
    base = baseline_metric(args)
    print(f"Utility-only baseline metric over {args.baseline_episodes} episodes: {base:.3f}")

    # Three disjoint seed blocks: training (seed + update*1000), selection, and a
    # test set that is never used for selection so the reported number is honest.
    val_seeds = [args.seed + 900_000 + k for k in range(max(1, args.val_episodes))]
    test_seeds = [args.seed + 700_000 + k for k in range(max(1, args.test_episodes))]
    prior_val = evaluate_deterministic(args, None, val_seeds, obb_safety_filter=True)
    print(
        f"Prior on {len(val_seeds)} held-out validation episodes (filter ON): "
        f"metric={prior_val['metric']:.3f} | collisions={prior_val['collisions']:.2f} | "
        f"arrival={prior_val['arrival_rate']:.3f}"
    )

    curve_path = args.curve
    if curve_path is None and args.save is not None:
        curve_path = args.save.with_name(f"{args.save.stem}_curve.csv")
    curve_rows: list[dict[str, float]] = [
        _curve_row(
            0,
            float("nan"),
            {**prior_val, "residual_usage": 0.0, "control_flip_rate": 0.0},
            prior_val,
            {"explained_variance": float("nan"), "approx_kl": float("nan"),
             "clip_frac": float("nan"), "entropy": float("nan"), "value_loss": float("nan")},
        )
    ]
    _persist_curve(curve_path, curve_rows)

    best_metric = float("inf")
    best_val: dict[str, float] | None = None
    best_update = 0
    best_state = None
    val_every = max(1, args.val_every)
    for update in range(1, args.updates + 1):
        if args.anneal_lr:
            frac = 1.0 - (update - 1) / max(args.updates, 1)
            for group in optimizer.param_groups:
                group["lr"] = args.lr * frac

        env = make_env(args, seed=args.seed + update * 1000)
        memory, metrics, collisions, realism, aux = collect_rollouts(
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
            target_kl=args.target_kl,
        )

        mean_metric = float(np.mean(metrics))
        best_metric = min(best_metric, mean_metric)

        val_note = ""
        if update % val_every == 0 or update == args.updates:
            val = evaluate_deterministic(args, policy, val_seeds, obb_safety_filter=True)
            if best_val is None or _validation_score(val) < _validation_score(best_val):
                best_val = val
                best_update = update
                best_state = {k: v.detach().cpu().clone() for k, v in policy.state_dict().items()}
            val_note = (
                f" | val={val['metric']:8.3f} (prior {prior_val['metric']:.3f})"
                f" | val_arrival={val['arrival_rate']:5.3f}"
                f" | usage={val['residual_usage']:5.3f}"
                f" | flip={val['control_flip_rate']:5.3f}"
                f" | train_flip={aux['control_flip_rate']:5.3f}"
            )
            curve_rows.append(_curve_row(update, mean_metric, val, prior_val, stats))
            _persist_curve(curve_path, curve_rows)

        log_every = args.log_every if args.log_every > 0 else max(min(args.updates // 10, 10), 1)
        if update == 1 or update % log_every == 0 or val_note:
            realism_note = ""
            if np.any(np.isfinite(realism)):
                realism_note = f" | realism={np.nanmean(realism):5.3f}"
            print(
                f"Update {update:4d}/{args.updates} | metric={mean_metric:8.3f} | "
                f"best={best_metric:8.3f} | collisions={np.mean(collisions):5.2f}{realism_note} | "
                f"ev={stats['explained_variance']:6.3f} | kl={stats['approx_kl']:6.4f} | "
                f"ep={stats['epochs_run']:.0f} | entropy={stats['entropy']:6.3f}{val_note}"
            )

    if curve_path is not None and curve_rows:
        _persist_curve(curve_path, curve_rows)
        print(f"Wrote learning curve to {curve_path} and {curve_path.with_suffix('.png')}")

    if best_state is not None:
        policy.load_state_dict(best_state)

    # Honest report: the test seeds never took part in selection.
    prior_test = evaluate_deterministic(args, None, test_seeds, obb_safety_filter=True)
    policy_test = evaluate_deterministic(args, policy, test_seeds, obb_safety_filter=True)
    print(
        f"Held-out TEST ({len(test_seeds)} episodes, never used for selection): "
        f"metric={policy_test['metric']:.3f} (prior {prior_test['metric']:.3f}) | "
        f"arrival={policy_test['arrival_rate']:.3f} (prior {prior_test['arrival_rate']:.3f}) | "
        f"collisions={policy_test['collisions']:.2f} (prior {prior_test['collisions']:.2f}) | "
        f"flip={policy_test['control_flip_rate']:.3f}"
    )

    if best_val is not None and _validation_score(best_val) >= _validation_score(prior_val):
        print(
            "Warning: no checkpoint beat the utility prior on the held-out validation "
            f"episodes under safety-first selection (collisions={best_val['collisions']:.3f} "
            f"vs prior {prior_val['collisions']:.3f}; metric={best_val['metric']:.3f} vs prior "
            f"{prior_val['metric']:.3f}); the residual is not adding value here."
        )

    if args.save is not None:
        args.save.parent.mkdir(parents=True, exist_ok=True)
        scales_blob: dict[str, float]
        if residual_mode == RESIDUAL_MODE_CANDIDATE:
            scales_blob = {
                f"c{i}": float(policy.residual_scales[i])
                for i in range(int(policy.residual_scales.numel()))
            }
        else:
            scales_blob = {k: float(DEFAULT_RESIDUAL_SCALES[k]) for k in RESIDUAL_PARAM_KEYS}
        blob = {
            "protocol_version": 2,
                "state_dict": policy.state_dict(),
            "obs_dim": probe_env.obs_dim,
            "hidden_dim": args.hidden_dim,
            "residual_scale": policy.residual_scale,
            "residual_scales": scales_blob,
            "param_gauge": PARAM_GAUGE,
            "action_space": DEFAULT_ACTION_SPACE,
            "residual_mode": residual_mode,
            "action_dim": int(probe_env.residual_dim),
            "highway_length": float(probe_env.config.highway_length),
            "algo": "ppo",
            "prefer_params": args.prefer_params,
            "calibration": str(args.calibration) if args.calibration else None,
            "base_params": dict(probe_env.config.base_params),
            "train_obb_filter": bool(probe_env.config.sim_config["obb_safety_filter"]),
            "num_agents": args.num_agents,
            "max_steps": args.max_steps,
            "leftover_coef": probe_env.config.leftover_coef,
            "arrival_bonus": probe_env.config.arrival_bonus,
            "collision_penalty": float(args.collision_penalty),
            "behavior_coef": float(args.behavior_coef),
            "behavior_shaping_coef": float(args.behavior_shaping_coef),
            "updates": int(args.updates),
            "selected_update": int(best_update),
            "val_seeds": [int(s) for s in val_seeds],
            "val_metric": float(best_val["metric"]) if best_val else float("nan"),
            "val_collisions": float(best_val["collisions"]) if best_val else float("nan"),
            "val_arrival_rate": float(best_val["arrival_rate"]) if best_val else float("nan"),
            "val_control_flip_rate": float(best_val.get("control_flip_rate", 0.0)) if best_val else float("nan"),
            "prior_val_metric": float(prior_val["metric"]),
            "prior_val_collisions": float(prior_val["collisions"]),
            "prior_val_arrival_rate": float(prior_val["arrival_rate"]),
            "test_seeds": [int(s) for s in test_seeds],
            "test_metric": float(policy_test["metric"]),
            "test_arrival_rate": float(policy_test["arrival_rate"]),
            "test_collisions": float(policy_test["collisions"]),
            "test_control_flip_rate": float(policy_test.get("control_flip_rate", 0.0)),
            "prior_test_metric": float(prior_test["metric"]),
            "prior_test_arrival_rate": float(prior_test["arrival_rate"]),
            "prior_test_collisions": float(prior_test["collisions"]),
            "best_collisions": float(best_val["collisions"]) if best_val else float("nan"),
        }
        torch.save(blob, args.save)
        if best_val is None:
            print(f"Saved PPO residual policy to {args.save}")
        else:
            print(
                f"Saved PPO residual policy to {args.save} (update {best_update}: "
                f"val metric={best_val['metric']:.3f} vs prior {prior_val['metric']:.3f}, "
                f"val arrival={best_val['arrival_rate']:.3f} vs prior "
                f"{prior_val['arrival_rate']:.3f}, val collisions={best_val['collisions']:.2f})"
            )


def main() -> None:
    parser = argparse.ArgumentParser(description="Train residual MARL policy with PPO")
    parser.add_argument("--updates", type=int, default=100)
    parser.add_argument("--episodes-per-update", type=int, default=4)
    parser.add_argument("--num-agents", type=int, default=10)
    parser.add_argument("--max-steps", type=int, default=240)
    parser.add_argument("--baseline-episodes", type=int, default=5)
    parser.add_argument(
        "--val-episodes",
        type=int,
        default=8,
        help="Held-out episodes used for deterministic checkpoint selection",
    )
    parser.add_argument(
        "--val-every",
        type=int,
        default=10,
        help="Run the deterministic validation rollouts every N updates",
    )
    parser.add_argument(
        "--test-episodes",
        type=int,
        default=16,
        help="Held-out episodes scored once at the end; never used for selection",
    )
    parser.add_argument(
        "--curve",
        type=Path,
        default=None,
        help="CSV for the validation trace (default: alongside --save)",
    )
    parser.add_argument(
        "--target-kl",
        type=float,
        default=0.02,
        help="Stop the epoch loop once the approximate KL exceeds this (0 disables)",
    )
    parser.add_argument(
        "--no-anneal-lr",
        dest="anneal_lr",
        action="store_false",
        help="Keep the learning rate constant instead of annealing it to zero",
    )
    parser.set_defaults(anneal_lr=True)
    parser.add_argument(
        "--residual-mode",
        choices=(RESIDUAL_MODE_CANDIDATE, RESIDUAL_MODE_PARAM),
        default=DEFAULT_RESIDUAL_MODE,
        help="candidate_logits (default) adds to the discrete utility grid; "
        "param_delta edits Θ (ablation)",
    )
    parser.add_argument(
        "--train-obb-filter",
        action="store_true",
        help="Keep the hard OBB filter on during training (default: off so collisions "
        "remain a reachable learning signal)",
    )
    parser.add_argument(
        "--leftover-coef",
        type=float,
        default=0.05,
        help="Per-step penalty weight on remaining station / highway length",
    )
    parser.add_argument(
        "--arrival-bonus",
        type=float,
        default=5.0,
        help="One-shot reward when an agent reaches its destination station",
    )
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--clip-coef", type=float, default=0.2)
    parser.add_argument("--value-coef", type=float, default=0.5)
    # The residual starts on the prior and is evaluated with mean actions, so an
    # entropy bonus only inflates sigma and widens the train/eval gap.
    parser.add_argument("--entropy-coef", type=float, default=0.0)
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
        choices=("robust", "best", "nominal"),
        default="robust",
        help="Which calibrated parameter set to freeze as Θ_base",
    )
    parser.add_argument(
        "--save",
        type=Path,
        default=Path("RL/checkpoints/v2/residual_policy.pt"),
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
