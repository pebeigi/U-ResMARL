#!/usr/bin/env python
"""Train the residual utility policy with shared-policy PPO.

Run from the repo root:
  python -m RL.train_ppo --calibration Calibration/utility_calibration.json
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import copy
from typing import Any

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
from RL.traffic_env import EnvConfig, MultiAgentTrafficEnv, SPAWN_PROTOCOL_VERSION
from RL.transition import DRIVING_REWARD_REVISION

try:
    import torch
    import torch.nn as nn
except ImportError as exc:
    raise SystemExit("PyTorch is required for training. Install with: pip install torch") from exc


def normalize_obs(obs: torch.Tensor, highway_length: float = 500.0) -> torch.Tensor:
    """Scale Frenet / body-frame features for the actor-critic."""
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


# Action-space tags stored in checkpoints so policies replay as trained.
# legacy / normalized_tanh / squashed_tanh: continuous residual on parameters.
# categorical_utility: masked categorical over utility + residual scores.
LEGACY_ACTION_SPACE = "legacy"
NORMALIZED_ACTION_SPACE = "normalized_tanh"
SQUASHED_ACTION_SPACE = "squashed_tanh"
DEFAULT_ACTION_SPACE = SQUASHED_ACTION_SPACE
CATEGORICAL_ACTION_SPACE = "categorical_utility"

# Residual interfaces: candidate logits (default) or parameter delta (ablation).
RESIDUAL_MODE_CANDIDATE = "candidate_logits"
RESIDUAL_MODE_PARAM = "param_delta"
DEFAULT_RESIDUAL_MODE = RESIDUAL_MODE_CANDIDATE
CANDIDATE_LOGIT_SCALE = 2.0
TRAINING_REVISION = 7

#: Floor inside log(1 - tanh(u)^2) so the squash correction stays finite.
_SQUASH_EPS = 1e-6


def action_space_from_blob(blob: dict) -> str:
    """Action-space variant a checkpoint was trained with."""
    return str(blob.get("action_space", LEGACY_ACTION_SPACE))


def residual_mode_from_blob(blob: dict) -> str:
    return str(blob.get("residual_mode", RESIDUAL_MODE_PARAM))


from RL.value_normalization import ValueNormalizer


class TorchResidualPolicy(ValueNormalizer):
    """Shared actor-critic for candidate-logit or continuous parameter residuals.

    Default (``candidate_logits``): additive residual on the discrete utility grid;
    training samples softmax(U + residual), eval takes argmax. Continuous modes
    score a Gaussian residual on utility parameters instead.
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
        candidate_temperature: float = 0.05,
    ):
        super().__init__()
        self.obs_dim = obs_dim
        self.hidden_dim = hidden_dim
        self.param_gauge = str(param_gauge)
        self.action_space = str(action_space)
        self.residual_mode = str(residual_mode)
        if self.action_space == CATEGORICAL_ACTION_SPACE:
            if self.residual_mode != RESIDUAL_MODE_CANDIDATE:
                raise ValueError("Categorical utility policy requires candidate_logits mode")
            if not np.isfinite(candidate_temperature) or candidate_temperature <= 0:
                raise ValueError("candidate_temperature must be finite and positive")
            self.register_buffer("candidate_temperature", torch.tensor(float(candidate_temperature)))
        self._residual_keys = (
            LEGACY_RESIDUAL_PARAM_KEYS if self.param_gauge == "additive" else RESIDUAL_PARAM_KEYS
        )
        if self.residual_mode == RESIDUAL_MODE_CANDIDATE:
            dim = int(action_dim if action_dim is not None else 63)
            if not np.isfinite(candidate_logit_scale) or candidate_logit_scale <= 0:
                raise ValueError("candidate_logit_scale must be finite and positive")
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
        if self.action_space == CATEGORICAL_ACTION_SPACE:
            raise ValueError("Use categorical_distribution with utility scores and an action mask")
        mean, value = self.forward(obs)
        return torch.distributions.Normal(mean, self._stddev(mean)), value

    def categorical_distribution(self, obs, utilities, mask):
        if utilities is None or mask is None:
            raise ValueError("Categorical PPO requires the rollout utility scores and action mask")
        mean, value_norm = self.evaluate(obs)
        utilities = torch.as_tensor(utilities, dtype=mean.dtype, device=mean.device)
        mask = torch.as_tensor(mask, dtype=torch.bool, device=mean.device)
        if utilities.shape != mean.shape or mask.shape != mean.shape or not mask.any(dim=-1).all():
            raise ValueError("Invalid categorical context shape or empty action mask")
        logits = (utilities + mean * self.residual_scales) / self.candidate_temperature
        logits = logits.masked_fill(~mask, -torch.inf)
        return torch.distributions.Categorical(logits=logits), value_norm

    def log_prob(self, obs: torch.Tensor, action: torch.Tensor, utilities=None,
                 mask=None) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Action log-probability, distribution entropy and normalized value.

        For the squashed policy ``action`` is the pre-squash sample ``u``; the
        tanh change of variables is subtracted so the density refers to the
        bounded residual that was actually executed.
        """
        if self.action_space == CATEGORICAL_ACTION_SPACE:
            dist, value_norm = self.categorical_distribution(obs, utilities, mask)
            return dist.log_prob(action.long()), dist.entropy(), value_norm
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
        if self.action_space in (NORMALIZED_ACTION_SPACE, CATEGORICAL_ACTION_SPACE):
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

        if self.action_space == CATEGORICAL_ACTION_SPACE and explore_std > 0:
            raise ValueError("Categorical exploration requires sample_action with utility context")

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

    def sample_action(self, obs: np.ndarray, utilities=None, mask=None):
        """Sample the action the environment will execute, and score that action."""
        obs_t = torch.as_tensor(obs, dtype=torch.float32)
        if self.action_space == CATEGORICAL_ACTION_SPACE:
            from RL.candidate_policy import CandidateIndex
            with torch.no_grad():
                dist, value_norm = self.categorical_distribution(obs_t, utilities, mask)
                action = dist.sample()
            return (CandidateIndex(int(action)), action.cpu().numpy(),
                    float(dist.log_prob(action)), float(self.denormalize_value(value_norm)))
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
    utilities: list[np.ndarray] | None = None
    action_masks: list[np.ndarray] | None = None


def collect_rollouts(
    env: MultiAgentTrafficEnv,
    policy: TorchResidualPolicy,
    episodes_per_update: int,
    *, episode_seeds: list[int] | None = None, gamma: float = 0.99, max_env_steps: int | None = None,
) -> tuple[PPOMemory, list[float], list[int], list[float], dict[str, float]]:
    memory = PPOMemory([], [], [], [], [], [], [], [], [])
    categorical = policy.action_space == CATEGORICAL_ACTION_SPACE
    if categorical:
        memory.utilities, memory.action_masks = [], []
    metrics: list[float] = []
    collisions: list[int] = []
    events: list[int] = []
    realism: list[float] = []
    episode_returns, discounted_returns = [], []
    flips = 0
    decisions = 0
    num_agents = len(env.agents) if env.agents else env.config.num_agents

    environment_steps = 0
    for episode in range(episodes_per_update):
        if max_env_steps is not None and environment_steps >= max_env_steps:
            break
        if episode_seeds is not None:
            env.rng = np.random.default_rng(episode_seeds[episode])
        obs_list = env.reset()
        num_agents = len(env.agents)
        done = False
        info: dict = {}
        episode_return = discounted_return = 0.0
        while not done:
            active = [i for i, a in enumerate(env.agents)
                      if not a.reached_destination and i not in env.collided_agents]
            residual_actions: list = [None for _ in env.agents]
            for i in active:
                if categorical:
                    context = env.candidate_context(i)
                    env_action, action_np, log_prob, value = policy.sample_action(
                        obs_list[i], context.utilities, context.mask)
                    memory.utilities.append(context.utilities.copy())
                    memory.action_masks.append(context.mask.copy())
                else:
                    env_action, action_np, log_prob, value = policy.sample_action(obs_list[i])
                residual_actions[i] = env_action
                memory.observations.append(obs_list[i])
                memory.actions.append(action_np)
                memory.log_probs.append(log_prob)
                memory.values.append(value)
                memory.traj_ids.append(episode * num_agents + i)

            step = env.step_count
            obs_list, rewards, done, info = env.step(residual_actions)
            episode_return += float(np.sum(rewards)) / num_agents
            discounted_return += gamma**step * float(np.sum(rewards)) / num_agents
            environment_steps += 1
            budget_end = max_env_steps is not None and environment_steps >= max_env_steps
            truncated = bool(info.get("truncated", False) or (budget_end and not done))
            done = bool(done or budget_end)
            flips += int(info.get("control_flips", 0))
            decisions += int(info.get("control_decisions", 0))
            for i in active:
                memory.rewards.append(float(rewards[i]))
                agent_terminal = bool(env.agents[i].reached_destination) or (
                    i in env.collided_agents) or (done and not truncated
                )
                memory.dones.append(float(agent_terminal))
                is_timeout = bool(truncated and not env.agents[i].reached_destination
                                  and i not in env.collided_agents)
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
        events.append(env.collision_events)
        realism.append(float(info.get("realism_distance", float("nan"))))
        episode_returns.append(episode_return)
        discounted_returns.append(discounted_return)

    aux = {
        "environment_steps": environment_steps,
        "active_agent_transitions": len(memory.actions),
        "collision_events": float(np.mean(events)),
        "control_flip_rate": float(flips / decisions) if decisions else 0.0,
        "control_flips": float(flips),
        "control_decisions": float(decisions),
        "mean_return": float(np.mean(episode_returns)),
        "discounted_return": float(np.mean(discounted_returns)),
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


def categorical_kl_from_logits(old_logits, new_logits, mask):
    """KL on the shared rollout mask without treating underflow as lost support.

    Categorical probabilities can round to zero at low temperature even though
    their log probabilities are finite. Compute in log space with float64;
    only the recorded hard mask defines which actions are impossible.
    """
    if (not torch.isfinite(old_logits[mask]).all()
            or not torch.isfinite(new_logits[mask]).all()):
        raise FloatingPointError("Non-finite categorical logit on an eligible action")
    old_logp = torch.log_softmax(old_logits.double().masked_fill(~mask, -torch.inf), dim=-1)
    new_logp = torch.log_softmax(new_logits.double().masked_fill(~mask, -torch.inf), dim=-1)
    difference = (old_logp - new_logp).masked_fill(~mask, 0.)
    return (old_logp.exp() * difference).sum(dim=-1).clamp_min(0.)


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
    utilities = (None if memory.utilities is None else
                 torch.as_tensor(np.asarray(memory.utilities), dtype=torch.float32))
    action_masks = (None if memory.action_masks is None else
                    torch.as_tensor(np.asarray(memory.action_masks), dtype=torch.bool))
    old_distribution = None
    if policy.action_space == CATEGORICAL_ACTION_SPACE:
        with torch.no_grad():
            old_distribution, _ = policy.categorical_distribution(obs, utilities, action_masks)
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
        "early_stop": 0.0,
    }
    stopped = False
    for epoch in range(epochs):
        np.random.shuffle(indices)
        epoch_kl: list[float] = []
        epoch_clip: list[float] = []
        for start in range(0, n, minibatch_size):
            batch_idx = indices[start : start + minibatch_size]
            if old_distribution is not None:
                current, value_norm = policy.categorical_distribution(
                    obs[batch_idx], utilities[batch_idx], action_masks[batch_idx])
                new_log_probs = current.log_prob(actions[batch_idx].long())
                entropy_terms = current.entropy()
                with torch.no_grad():
                    batch_kl = float(categorical_kl_from_logits(
                        old_distribution.logits[batch_idx], current.logits,
                        action_masks[batch_idx]).mean())
            else:
                new_log_probs, entropy_terms, value_norm = policy.log_prob(obs[batch_idx], actions[batch_idx])
                log_ratio_check = new_log_probs - old_log_probs[batch_idx]
                batch_kl = float(((log_ratio_check.exp() - 1) - log_ratio_check).mean().detach())
            epoch_kl.append(batch_kl)
            # Check before taking another step, not after a full PPO epoch.
            if target_kl > 0 and batch_kl > target_kl:
                stopped = True
                last_stats["early_stop"] = 1.0
                break
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
        if stopped:
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
    # Match the controller used at evaluation; filter-off is an explicit ablation.
    if obb_safety_filter is None:
        obb_safety_filter = bool(getattr(args, "train_obb_filter", True))
    cfg_kwargs = dict(
        max_steps=args.max_steps,
        num_agents=args.num_agents,
        base_params=base_params,
        collision_penalty=float(getattr(args, "collision_penalty", 0.0)),
        collision_event_penalty=float(getattr(args, "collision_event_penalty", 0.0)),
        behavior_coef=float(getattr(args, "behavior_coef", 0.0)),
        behavior_shaping_coef=float(getattr(args, "behavior_shaping_coef", 0.0)),
        residual_mode=str(getattr(args, "residual_mode", DEFAULT_RESIDUAL_MODE)),
        leftover_coef=float(getattr(args, "leftover_coef", 0.08)),
        arrival_bonus=float(getattr(args, "arrival_bonus", 8.0)),
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
    label: str | None = None,
) -> dict[str, Any]:
    """Roll out ``policy`` with mean actions under the benchmark safety setting."""
    from Baselines.metrics import rollout_metrics
    from Baselines.runner import RolloutRecorder
    from Baselines.scenario import scenario_from_env
    metrics: list[float] = []
    collisions: list[float] = []
    collision_events: list[float] = []
    arrivals: list[float] = []
    usage: list[float] = []
    flips = 0
    decisions = 0
    episodes = []
    scales = (
        policy.residual_scales.detach().cpu().numpy() if policy is not None else None
    )
    for episode, seed in enumerate(seeds, 1):
        if label:
            print(f"{label}: episode {episode}/{len(seeds)} (seed {seed})", flush=True)
        env = make_env(args, seed=int(seed), obb_safety_filter=obb_safety_filter)
        env.config.accept_residual_if_better = policy is not None
        obs_list = env.reset()
        recorder = RolloutRecorder(scenario_from_env(env, int(seed)), env.agents,
                                   "utility" if policy is None else "residual_marl")
        done = False
        info: dict = {}
        episode_return = discounted_return = 0.0
        while not done:
            actions = None
            if policy is not None:
                actions = [policy.act(obs, 0.0)[0] for obs in obs_list]
                for agent, delta in zip(env.agents, actions):
                    if agent.reached_destination:
                        continue
                    arr = np.asarray(
                        list(delta.values()) if isinstance(delta, dict) else delta,
                        dtype=float,
                    )
                    usage.append(float(np.mean(np.abs(arr) / scales)))
            step = env.step_count
            obs_list, rewards, done, info = env.step(actions)
            episode_return += float(np.sum(rewards)) / len(env.agents)
            discounted_return += float(getattr(args, "gamma", .99))**step * float(np.sum(rewards)) / len(env.agents)
            recorder.record(env.agents,
                            [(c["accel"], c["steering"]) for c in info["selected_controls"]],
                            env._collision_pairs, proposed_controls=info.get("proposed_controls"))
            flips += int(info.get("control_flips", 0))
            decisions += int(info.get("control_decisions", 0))
        metrics.append(float(env.rollout_metric()))
        collisions.append(float(env.collision_count))
        collision_events.append(float(env.collision_events))
        arrivals.append(float(np.mean([bool(a.reached_destination) for a in env.agents])))
        episodes.append({**rollout_metrics(recorder.result()), "seed": int(seed), "metric": metrics[-1],
                         "collision_pair_steps": collisions[-1],
                         "collision_events": collision_events[-1], "arrival_rate": arrivals[-1],
                         "mean_return": episode_return, "discounted_return": discounted_return,
                         "leftover_distance": metrics[-1] - 10.0 * collisions[-1]})
    # The same per-episode definitions as Baselines.benchmark, excluding metadata.
    metric_keys = ("offroad_rate", "offroad_agent_frac", "collision_rate_per_agent",
                   "mean_return", "discounted_return",
                   "min_gap_m", "p5_gap_m", "min_ttc_s", "unsafe_ttc_rate", "goal_progress",
                   "mean_travel_time_s", "mean_capped_travel_time_s", "mean_speed_mps",
                   "speed_std_mps", "mean_abs_accel", "rms_jerk", "mean_abs_steering",
                   "mean_abs_lateral_m", "min_clearance_m", "wall_time_per_agent_step_ms",
                   "closed_loop_score", "shield_intervention_rate")
    shared_metrics = {}
    for key in metric_keys:
        values = [row[key] for row in episodes if np.isfinite(row[key])]
        shared_metrics[key] = float(np.mean(values)) if values else float("nan")
    return {
        **shared_metrics,
        "metric": float(np.mean(metrics)),
        "collisions": float(np.mean(collisions)),
        "collision_events": float(np.mean(collision_events)),
        "arrival_rate": float(np.mean(arrivals)),
        "residual_usage": float(np.mean(usage)) if usage else 0.0,
        "control_flip_rate": float(flips / decisions) if decisions else 0.0,
        "episode_results": episodes,
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
    axes[0, 0].set_ylabel("distance + 10 × collision pair-steps (↓)")
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
        "val_closed_loop_score": val.get("closed_loop_score", float("nan")),
        "val_arrival_rate": val["arrival_rate"],
        "val_collisions": val["collisions"],
        "val_collision_events": val.get("collision_events", float("nan")),
        "residual_usage": val.get("residual_usage", 0.0),
        "control_flip_rate": val.get("control_flip_rate", 0.0),
        "prior_val_metric": prior_val["metric"],
        "prior_val_closed_loop_score": prior_val.get("closed_loop_score", float("nan")),
        "prior_val_collisions": prior_val["collisions"],
        "prior_val_collision_events": prior_val.get("collision_events", float("nan")),
        "prior_val_arrival_rate": prior_val["arrival_rate"],
        "train_return": stats.get("mean_return", float("nan")),
        "train_discounted_return": stats.get("discounted_return", float("nan")),
        "val_return": val.get("mean_return", float("nan")),
        "val_discounted_return": val.get("discounted_return", float("nan")),
        "prior_val_return": prior_val.get("mean_return", float("nan")),
        "prior_val_discounted_return": prior_val.get("discounted_return", float("nan")),
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


from RL.experiment_protocol import (validation_score as _validation_score,
    regressions as validation_regressions, SAFETY_KEYS, SELECTION_RULE, TrainingBudget)

VALIDATION_OBJECTIVES = {key: -1 for key in SAFETY_KEYS}


def _validation_improves(candidate, incumbent, prior):
    return not validation_regressions(candidate, prior) and _validation_score(candidate) < _validation_score(incumbent)


def experiment_seeds(args):
    """Explicit scenario seeds; all episodes, not just update seeds, are checked."""
    for key in ("updates", "episodes_per_update", "num_agents", "max_steps",
                "val_episodes", "test_episodes"):
        if int(getattr(args, key)) <= 0:
            raise ValueError(f"{key} must be positive")
    if args.episodes_per_update >= 1000:
        raise ValueError("episodes_per_update must be below the 1000-seed update stride")
    from RL.experiment_protocol import training_seed
    training = [[training_seed(args, update, ep)
                 for ep in range(args.episodes_per_update)]
                for update in range(1, args.updates + 1)]
    validation = list(range(int(getattr(args, "validation_seed_start", 910_000)),
                            int(getattr(args, "validation_seed_start", 910_000)) + args.val_episodes))
    test = list(range(int(getattr(args, "test_seed_start", 810_000)),
                      int(getattr(args, "test_seed_start", 810_000)) + args.test_episodes))
    train_set = {seed for block in training for seed in block}
    if (train_set & set(validation) or train_set & set(test)
            or set(validation) & set(test)):
        raise ValueError("Training, validation and test scenario seeds must be disjoint")
    return training, validation, test


def _atomic_save(blob, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(blob, temporary)
    temporary.replace(path)


def _export_checkpoint(blob, path):
    """Write the controller and a readable report without a separate verifier script."""
    import json
    def finite_json(value):
        if isinstance(value, dict):
            return {key: finite_json(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [finite_json(item) for item in value]
        if isinstance(value, float) and not np.isfinite(value):
            return None
        return value
    _atomic_save(blob, path)
    summary = Path(path).with_suffix(".summary.json")
    temporary = summary.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(finite_json({k: v for k, v in blob.items() if k != "state_dict"}),
                                    indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(summary)


def _training_fingerprint(args):
    """Reject a resume if code or the frozen calibration has changed."""
    import hashlib
    root = Path(__file__).resolve().parent.parent
    sources = sorted((root / "RL").glob("*.py")) + [root / "utility_model.py"]
    sources += [root / "Baselines" / name for name in ("runner.py", "metrics.py", "scenario.py")]
    if args.calibration is not None:
        sources.append(Path(args.calibration))
    if getattr(args, 'calibration', None) is not None and Path(args.calibration).name == 'utility_calibration_tgsim.json':
        sources += sorted((root/'TGSIM Case').glob('*.py'))
        sources.append(root/'data/TGSIM FB/derived_boundaries/street_boundaries.csv')
    digest = hashlib.sha256()
    for source in sources:
        digest.update(source.name.encode())
        digest.update(source.read_bytes())
    return digest.hexdigest()


def _checkpoint_metadata(args, env, policy):
    from RL.experiment_protocol import validation_config
    return {
        "protocol_version": 3, "spawn_protocol_version": SPAWN_PROTOCOL_VERSION,
        "training_revision": TRAINING_REVISION, "collision_filter_revision": 2,
        "driving_reward_revision": DRIVING_REWARD_REVISION,
        "selection_rule": SELECTION_RULE,
        "validation_config": validation_config(args, env.config.sim_config, env.config.base_params),
        "decision_protocol_version": 1,
        "training_seed_rule": "10000000 + seed + update*1000 + episode",
        "validation_objectives": VALIDATION_OBJECTIVES,
        "obs_dim": env.obs_dim, "hidden_dim": args.hidden_dim,
        "residual_scale": policy.residual_scale,
        "residual_scales": dict(zip(policy._residual_keys, policy.residual_scales.tolist())),
        "param_gauge": PARAM_GAUGE, "action_space": policy.action_space,
        "candidate_temperature": float(getattr(args, "candidate_temperature", .005)),
        "residual_mode": policy.residual_mode, "action_dim": env.residual_dim,
        "highway_length": env.config.highway_length, "algo": "ppo",
        "base_params": dict(env.config.base_params), "prefer_params": args.prefer_params,
        "calibration": str(args.calibration) if args.calibration else None,
        "train_obb_filter": bool(env.config.sim_config["obb_safety_filter"]),
        "boundary_safety_filter": bool(env.config.sim_config["boundary_safety_filter"]),
        "boundary_margin": env.config.sim_config["boundary_margin"],
        "num_agents": args.num_agents, "max_steps": args.max_steps,
        "dense_spawn": bool(getattr(args, "dense_spawn", False)),
        "seed": args.seed, "reward_weights": dict(env.config.reward_weights),
        "leftover_coef": env.config.leftover_coef, "arrival_bonus": env.config.arrival_bonus,
        "collision_penalty": args.collision_penalty,
        "collision_event_penalty": float(getattr(args, "collision_event_penalty", 0.0)),
        "gamma": float(args.gamma),
        "behavior_coef": args.behavior_coef, "behavior_shaping_coef": args.behavior_shaping_coef,
    }


def train(args: argparse.Namespace) -> None:
    train_seeds, val_seeds, test_seeds = experiment_seeds(args)
    from RL.experiment_protocol import source_manifest
    sources = source_manifest()
    fingerprint = _training_fingerprint(args)
    resumed = None
    if getattr(args, "resume", None) is not None:
        resumed = torch.load(args.resume, map_location="cpu", weights_only=False)
        if resumed["fingerprint"] != fingerprint:
            raise ValueError("Code or calibration changed since the resume checkpoint")
        if resumed.get("complete", False):
            print("This run is already complete.")
            return
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    probe_env = make_env(args, seed=args.seed)
    residual_mode = str(getattr(args, "residual_mode", DEFAULT_RESIDUAL_MODE))
    action_space = getattr(args, "policy_action_space", "auto")
    if action_space == "auto":
        action_space = (CATEGORICAL_ACTION_SPACE if residual_mode == RESIDUAL_MODE_CANDIDATE
                        else SQUASHED_ACTION_SPACE)
    policy = TorchResidualPolicy(
        probe_env.obs_dim,
        hidden_dim=args.hidden_dim,
        residual_scales=DEFAULT_RESIDUAL_SCALES,
        highway_length=float(probe_env.config.highway_length),
        residual_mode=residual_mode,
        action_space=action_space,
        action_dim=probe_env.residual_dim,
        candidate_logit_scale=float(getattr(args, "candidate_logit_scale", CANDIDATE_LOGIT_SCALE)),
        candidate_temperature=float(getattr(args, "candidate_temperature", 0.005)),
    )
    optimizer = torch.optim.Adam(policy.parameters(), lr=args.lr)
    budget = TrainingBudget(args, resumed.get("budget") if resumed else None)
    budget.watch(optimizer)

    if args.calibration is not None:
        print(f"Using {args.prefer_params} utility params from {args.calibration}")
    print(
        f"Residual mode={residual_mode} | policy={action_space} | action_dim={probe_env.residual_dim} | "
        f"train_obb_filter={bool(probe_env.config.sim_config['obb_safety_filter'])}"
    )
    if args.collision_penalty > 0.0 or float(getattr(args, "collision_event_penalty", 0.0)) > 0.0:
        print(
            f"OBB contact terminal = {float(getattr(args, 'collision_event_penalty', 0.0)):.2f}, "
            f"duration = {args.collision_penalty:.2f} "
            f"(gamma={args.gamma:.3f})"
        )
    if args.behavior_coef > 0.0 or args.behavior_shaping_coef > 0.0:
        print(
            f"Behavioral data term: band={args.behavior_coef:.2f}, "
            f"shaping={args.behavior_shaping_coef:.2f}"
        )
    if resumed is None:
        prior_val = evaluate_deterministic(args, None, val_seeds, obb_safety_filter=True, label="Prior validation")
    else:
        prior_val = resumed["prior_val"]
    print(
        f"Prior on {len(val_seeds)} held-out validation episodes (filter ON): "
        f"metric={prior_val['metric']:.3f} | pair_steps={prior_val['collisions']:.2f} | "
        f"events={prior_val['collision_events']:.2f} | "
        f"arrival={prior_val['arrival_rate']:.3f} | "
        f"pdms={prior_val.get('closed_loop_score', float('nan')):.3f}"
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
    curve_rows[0].update(environment_steps=0, active_agent_transitions=0, optimizer_steps=0)
    if resumed is None:
        _persist_curve(curve_path, curve_rows)

    best_metric = float("inf")
    # The zero-initialized deterministic actor exactly reproduces the prior.
    # Include it in selection so failed learning cannot replace it with a
    # validation-regressing residual. This is a fallback, not learned improvement.
    best_val: dict[str, float] | None = dict(prior_val)
    best_update = 0
    validation_history = []
    best_state = {k: v.detach().cpu().clone() for k, v in policy.state_dict().items()}
    val_every = max(1, args.val_every)
    start_update = 1
    if resumed is not None:
        policy.load_state_dict(resumed["state_dict"])
        optimizer.load_state_dict(resumed["optimizer"])
        best_state, best_val = resumed["best_state"], resumed["best_val"]
        best_update, best_metric = resumed["best_update"], resumed["best_metric"]
        curve_rows = resumed["curve_rows"]
        validation_history = resumed.get("validation_history", [])
        torch.set_rng_state(resumed["torch_rng"])
        np.random.set_state(resumed["numpy_rng"])
        start_update = resumed["update"] + (0 if resumed["pending_evaluation"] else 1)
        print(f"Resuming after saved update {resumed['update']}")

    def selected_checkpoint(update, *, policy_test=None, prior_test=None):
        blob = _checkpoint_metadata(args, probe_env, policy)
        blob.update(state_dict=best_state, selected_update=best_update, updates=update, source_hashes=sources,
                    selection_status="utility_fallback" if best_update == 0 else "learned_residual",
                    evaluation_status="validation_only" if policy_test is None else "test_complete",
                    val_seeds=val_seeds, test_seeds=test_seeds, train_scenario_seeds=train_seeds,
                    validation=best_val, prior_validation=prior_val,
                    validation_history=validation_history, budget=budget.state(),
                    stopping_reason="environment_budget" if budget.exhausted else "update_limit")
        reports = {"val": best_val, "prior_val": prior_val}
        if policy_test is not None:
            blob.update(test=policy_test, prior_test=prior_test)
            reports.update(test=policy_test, prior_test=prior_test)
        # Preserve the small flat summary consumed by older analysis scripts.
        for prefix, report in reports.items():
            for key in ("metric", "collisions", "collision_events", "arrival_rate", "control_flip_rate"):
                blob[f"{prefix}_{key}"] = float(report.get(key, 0.0))
        return blob

    def save_progress(update, *, pending=False, diagnostics=None, complete=False):
        if args.save is None:
            return
        state = {
            "fingerprint": fingerprint, "args": vars(args), "update": update,
            "pending_evaluation": pending, "diagnostics": diagnostics,
            "state_dict": policy.state_dict(), "optimizer": optimizer.state_dict(),
            "best_state": best_state, "best_val": best_val, "best_update": best_update,
            "best_metric": best_metric, "prior_val": prior_val, "curve_rows": curve_rows,
            "validation_history": validation_history, "budget": budget.state(),
            "torch_rng": torch.get_rng_state(), "numpy_rng": np.random.get_state(),
            "complete": complete,
        }
        _atomic_save(state, args.save.with_suffix(".resume.pt"))
        if complete:
            return
        # Export the selected controller before the expensive test phase too.
        _export_checkpoint(selected_checkpoint(update), args.save)

    if resumed is None:
        save_progress(0)
    completed_update = start_update-1
    for update in range(start_update, args.updates + 1):
        pending_resume = resumed is not None and resumed["pending_evaluation"] and update == resumed["update"]
        if budget.exhausted and not pending_resume:
            break
        completed_update = update
        if args.anneal_lr:
            frac = 1.0 - (update - 1) / max(args.updates, 1)
            for group in optimizer.param_groups:
                group["lr"] = args.lr * frac

        if resumed is not None and resumed["pending_evaluation"] and update == resumed["update"]:
            metrics, collisions, realism, aux, stats = resumed["diagnostics"]
        else:
            env = make_env(args, seed=train_seeds[update - 1][0])
            memory, metrics, collisions, realism, aux = collect_rollouts(
                env, policy, args.episodes_per_update, episode_seeds=train_seeds[update - 1], gamma=args.gamma,
                max_env_steps=budget.remaining(args.max_steps*args.episodes_per_update))
            budget.add(aux["environment_steps"], aux["active_agent_transitions"])
            budget.updates = update
            stats = ppo_update(
                policy, optimizer, memory, gamma=args.gamma, gae_lambda=args.gae_lambda,
                clip_coef=args.clip_coef, value_coef=args.value_coef, entropy_coef=args.entropy_coef,
                epochs=args.ppo_epochs, minibatch_size=args.minibatch_size, target_kl=args.target_kl)
            stats.update(mean_return=aux.get("mean_return", float("nan")),
                         discounted_return=aux.get("discounted_return", float("nan")))

        mean_metric = float(np.mean(metrics))
        best_metric = min(best_metric, mean_metric)
        save_progress(update, pending=True, diagnostics=(metrics, collisions, realism, aux, stats))

        val_note = ""
        if budget.validation_due(args, update):
            val = evaluate_deterministic(args, policy, val_seeds, obb_safety_filter=True,
                                         label=f"Validation update {update}")
            val["regressions_vs_utility"] = validation_regressions(val, prior_val)
            validation_history.append({"update": update, **budget.state(), **val})
            if val["regressions_vs_utility"]:
                print("Validation regressions vs utility: " + ", ".join(val["regressions_vs_utility"]), flush=True)
            if _validation_improves(val, best_val, prior_val):
                best_val = val
                best_update = update
                best_state = {k: v.detach().cpu().clone() for k, v in policy.state_dict().items()}
            val_note = (
                f" | val={val['metric']:8.3f} (prior {prior_val['metric']:.3f})"
                f" | val_pdms={val.get('closed_loop_score', float('nan')):5.3f}"
                f" | val_arrival={val['arrival_rate']:5.3f}"
                f" | val_events={val['collision_events']:5.2f}"
                f" | val_pair_steps={val['collisions']:5.2f}"
                f" | val_return={val.get('mean_return', float('nan')):7.2f}"
                f" | val_discounted={val.get('discounted_return', float('nan')):7.2f}"
                f" | usage={val['residual_usage']:5.3f}"
                f" | flip={val['control_flip_rate']:5.3f}"
                f" | train_flip={aux['control_flip_rate']:5.3f}"
            )
            row = _curve_row(update, mean_metric, val, prior_val, stats)
            row.update(environment_steps=budget.env_steps, active_agent_transitions=budget.agent_steps, optimizer_steps=budget.optimizer_steps)
            curve_rows.append(row)
            _persist_curve(curve_path, curve_rows)

        log_every = args.log_every if args.log_every > 0 else max(min(args.updates // 10, 10), 1)
        if update == 1 or update % log_every == 0 or val_note:
            realism_note = ""
            if np.any(np.isfinite(realism)):
                realism_note = f" | realism={np.nanmean(realism):5.3f}"
            print(
                f"Update {update:4d}/{args.updates} | metric={mean_metric:8.3f} | "
                f"best={best_metric:8.3f} | pair_steps={np.mean(collisions):5.2f} | "
                f"events={aux.get('collision_events', float('nan')):5.2f}{realism_note} | "
                f"return={aux.get('mean_return', float('nan')):7.2f} | "
                f"ev={stats['explained_variance']:6.3f} | kl={stats['approx_kl']:6.4f} | "
                f"episodes={args.episodes_per_update} | epochs={stats['epochs_run']:.0f} | "
                f"entropy={stats['entropy']:6.3f}{val_note}"
            )
        save_progress(update)

    if curve_path is not None and curve_rows:
        _persist_curve(curve_path, curve_rows)
        print(f"Wrote learning curve to {curve_path} and {curve_path.with_suffix('.png')}")

    if getattr(args, "skip_test", False):
        save_progress(completed_update, complete=True)
        print(f"Development run complete; selected update {best_update} "
              f"({'utility fallback' if best_update == 0 else 'learned residual'}); "
              "held-out test was not evaluated.")
        return

    # Keep the latest optimizer/actor paired in the resumable state.
    latest_state = copy.deepcopy(policy.state_dict())
    if best_state is not None:
        policy.load_state_dict(best_state)

    # Honest report: the test seeds never took part in selection.
    prior_test = evaluate_deterministic(args, None, test_seeds, obb_safety_filter=True, label="Prior test")
    policy_test = evaluate_deterministic(args, policy, test_seeds, obb_safety_filter=True, label="Selected policy test")
    print(
        f"Held-out TEST ({len(test_seeds)} episodes, never used for selection): "
        f"metric={policy_test['metric']:.3f} (prior {prior_test['metric']:.3f}) | "
        f"arrival={policy_test['arrival_rate']:.3f} (prior {prior_test['arrival_rate']:.3f}) | "
        f"pair_steps={policy_test['collisions']:.2f} (prior {prior_test['collisions']:.2f}) | "
        f"events={policy_test['collision_events']:.2f} (prior {prior_test['collision_events']:.2f}) | "
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
        _export_checkpoint(selected_checkpoint(completed_update, policy_test=policy_test, prior_test=prior_test), args.save)
        policy.load_state_dict(latest_state)
        save_progress(completed_update, complete=True)
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
    from RL.experiment_protocol import add_budget_args
    add_budget_args(parser)
    parser.add_argument("--updates", type=int, default=100)
    parser.add_argument("--episodes-per-update", type=int, default=4)
    parser.add_argument("--num-agents", type=int, default=10)
    parser.add_argument("--max-steps", type=int, default=240)
    parser.add_argument(
        "--val-episodes",
        type=int,
        default=16,
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
        "param_delta edits utility parameters (ablation)",
    )
    parser.add_argument(
        "--train-obb-filter",
        action="store_true",
        default=True,
        help="Keep the same OBB filter as evaluation during training (default)",
    )
    parser.add_argument("--no-train-obb-filter", dest="train_obb_filter", action="store_false",
                        help="Explicit ablation: train without the evaluation OBB filter")
    parser.add_argument("--candidate-logit-scale", type=float, default=0.5,
                        help="Bound on candidate residual scores; recorded in checkpoint buffers")
    parser.add_argument("--policy-action-space", default="auto",
                        choices=("auto", CATEGORICAL_ACTION_SPACE, SQUASHED_ACTION_SPACE),
                        help="auto: categorical utility sampling for candidate residuals, Gaussian for parameters")
    parser.add_argument("--candidate-temperature", type=float, default=0.005,
                        help="Temperature of softmax(U + residual) during categorical training")
    parser.add_argument("--validation-seed-start", type=int, default=910_000)
    parser.add_argument("--test-seed-start", type=int, default=810_000,
                        help="Fresh block; previous 700000/800000 cases are development cases")
    parser.add_argument("--skip-test", action="store_true",
                        help="Development run: validate and save without inspecting held-out test scenarios")
    parser.add_argument("--resume", type=Path, default=None,
                        help="Resume a .resume.pt file with its stored settings and random states")
    parser.add_argument(
        "--leftover-coef",
        type=float,
        default=0.08,
        help="Per-step penalty weight on remaining station / highway length",
    )
    parser.add_argument(
        "--arrival-bonus",
        type=float,
        default=8.0,
        help="One-shot reward when an agent reaches its destination station",
    )
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument(
        "--gamma",
        type=float,
        default=0.95,
        help="Discount; 0.99 undervalued late contacts at dt=0.5 in development replays",
    )
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
        help="Calibration JSON; uses validation-selected working_params by default",
    )
    parser.add_argument(
        "--prefer-params",
        choices=("working", "robust", "best", "nominal"),
        default="working",
        help="Which calibrated parameter set to freeze as the utility prior",
    )
    parser.add_argument(
        "--save",
        type=Path,
        default=Path("RL/checkpoints/residual_policy.pt"),
    )
    parser.add_argument(
        "--collision-penalty",
        type=float,
        default=0.0,
        help="Optional per-step duration penalty on first contact (CaRL default is 0)",
    )
    parser.add_argument(
        "--collision-event-penalty",
        type=float,
        default=1.0,
        help="Terminal contact cost charged once when a colliding pair first appears",
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
    if args.resume is not None:
        import sys
        if any(arg.startswith("--") and arg.split("=")[0] != "--resume" for arg in sys.argv[1:]):
            parser.error("--resume restores the saved settings; pass it without other training options")
        resume_path = args.resume
        saved = torch.load(resume_path, map_location="cpu", weights_only=False)
        args = argparse.Namespace(**saved["args"])
        args.resume = resume_path
    if args.calibration is not None and not args.calibration.exists():
        print(f"Warning: calibration file missing ({args.calibration}); using DEFAULT_BASE_PARAMS")
        args.calibration = None
    train(args)


if __name__ == "__main__":
    main()
