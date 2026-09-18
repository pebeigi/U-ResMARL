"""Direct discrete RL: π(o) → logits over the same (accel, δ) grid as the utility family.

Matches the residual policy on every non-interpretability axis:
  - same local observations and Frenet normalization as residual training
  - same kinematic bicycle candidates
  - same 1.5 s / 4-substep OBB conflict rejection before executing a move
  - same scenario spawn logic, reward, and collision penalty (when configured)

The only difference vs. residual MARL is the action parameterization:
  residual:  π(o) → ΔΘ  → argmax_a U(a; Θ_base + ΔΘ)
  direct:    π(o) → logits over the discrete candidate grid → sample / argmax
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

import Baselines._paths  # noqa: F401
from Baselines.controllers import BaseController
from Baselines.discrete_action import (
    feasible_action_mask,
    grid_control,
    num_grid_actions,
)
from Baselines.dynamics import observation, observation_dim
from RL.obs import adapt_observation
from RL.train_ppo import normalize_obs
from utility_model import TrafficAgent

if TYPE_CHECKING:  # pragma: no cover
    from Baselines.scenario import Scenario

try:
    import torch
    import torch.nn as nn
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "PyTorch is required for the direct discrete RL baseline. Install with: pip install torch"
    ) from exc

DEFAULT_CHECKPOINT = Path("Baselines/checkpoints/revision5/direct_discrete_policy.pt")


from RL.value_normalization import ValueNormalizer


class DirectDiscretePolicy(ValueNormalizer):
    """Shared actor-critic with a categorical policy over the candidate grid."""

    def __init__(
        self,
        obs_dim: int,
        num_actions: int,
        hidden_dim: int = 128,
        highway_length: float = 500.0,
    ):
        super().__init__()
        self.obs_dim = int(obs_dim)
        self.num_actions = int(num_actions)
        self.hidden_dim = int(hidden_dim)
        self.register_buffer("highway_length", torch.tensor(float(highway_length)))

        self.actor = nn.Sequential(
            nn.Linear(self.obs_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, self.num_actions),
        )
        self.critic = nn.Sequential(
            nn.Linear(self.obs_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )

    def _features(self, obs: torch.Tensor) -> torch.Tensor:
        return normalize_obs(obs, float(self.highway_length.item()))

    def logits(self, obs: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        features = self._features(obs)
        logits = self.actor(features)
        if mask is not None:
            logits = logits.masked_fill(~mask, -1e8)
        return logits

    def distribution(
        self,
        obs: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> tuple["torch.distributions.Categorical", torch.Tensor]:
        logits = self.logits(obs, mask)
        dist = torch.distributions.Categorical(logits=logits)
        value = self.critic(self._features(obs)).squeeze(-1)
        return dist, self.denormalize_value(value)

    def sample_action(
        self,
        obs: np.ndarray,
        mask: np.ndarray,
    ) -> tuple[int, float, float]:
        obs_t = torch.as_tensor(adapt_observation(obs, self.obs_dim), dtype=torch.float32)
        mask_t = torch.as_tensor(mask, dtype=torch.bool)
        with torch.no_grad():
            dist, value = self.distribution(obs_t, mask_t)
            action = dist.sample()
            log_prob = dist.log_prob(action)
        return int(action.item()), float(log_prob), float(value)

    def act(
        self,
        obs: np.ndarray,
        mask: np.ndarray,
        explore: bool = False,
    ) -> int:
        obs_t = torch.as_tensor(adapt_observation(obs, self.obs_dim), dtype=torch.float32)
        mask_t = torch.as_tensor(mask, dtype=torch.bool)
        with torch.no_grad():
            logits = self.logits(obs_t, mask_t)
            if explore:
                dist = torch.distributions.Categorical(logits=logits)
                return int(dist.sample().item())
            return int(torch.argmax(logits).item())


def load_direct_discrete_policy(checkpoint: Path, obs_dim: int, num_actions: int) -> DirectDiscretePolicy:
    blob = torch.load(checkpoint, map_location="cpu")
    from RL.protocol import validate_checkpoint
    validate_checkpoint(blob, obs_dim, checkpoint)
    policy = DirectDiscretePolicy(
        obs_dim=int(blob.get("obs_dim", obs_dim)),
        num_actions=int(blob.get("num_actions", num_actions)),
        hidden_dim=int(blob.get("hidden_dim", 128)),
        highway_length=float(blob.get("highway_length", 500.0)),
    )
    policy.load_state_dict(blob["state_dict"])
    policy.eval()
    return policy


class DirectDiscreteRLController(BaseController):
    """Categorical policy over the utility candidate grid with OBB rejection."""

    name = "direct_discrete_rl"

    def __init__(
        self,
        checkpoint: Path | str | None = DEFAULT_CHECKPOINT,
        policy: DirectDiscretePolicy | None = None,
        explore: bool = False,
        name: str | None = None,
    ):
        self.checkpoint = Path(checkpoint) if checkpoint is not None else None
        self.policy = policy
        self.explore = bool(explore)
        if name:
            self.name = name
        self._warned = False

    def reset(self, scenario: "Scenario") -> None:
        if self.policy is not None or self.checkpoint is None:
            return
        if not self.checkpoint.exists():
            raise FileNotFoundError(f"Missing trained checkpoint {self.checkpoint}; train before evaluation")
        self.policy = load_direct_discrete_policy(
            self.checkpoint,
            observation_dim(scenario),
            num_grid_actions(scenario.sim_config),
        )

    def compute_controls(
        self,
        agents: list[TrafficAgent],
        scenario: "Scenario",
        step: int,
    ) -> list[tuple[float, float]]:
        controls: list[tuple[float, float]] = []
        for i, agent in enumerate(agents):
            if agent.reached_destination:
                controls.append((0.0, 0.0))
                continue
            mask = feasible_action_mask(i, agent, agents, scenario)
            obs = observation(agents, i, scenario)
            action_idx = self.policy.act(obs, mask, explore=self.explore)
            if not mask[action_idx]:
                fallback = int(np.flatnonzero(mask)[0])
                action_idx = fallback
            accel, steering = grid_control(action_idx, scenario.sim_config)
            controls.append((float(accel), float(steering)))
        return controls
