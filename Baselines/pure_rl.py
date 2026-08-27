"""Pure-RL baseline: no behavioural prior, no utility function.

The policy maps the same local observation used by the residual model directly
to a bicycle command (accel, steering). This is the "purely learning-based
reinforcement learning without behavioral priors" comparison in the paper.

The Gaussian lives in a normalised action space ([-1, 1] per dimension, scaled
afterwards to the physical limits) and observations are whitened by a running
mean/variance estimate, so the baseline is not handicapped by input or action
scaling.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

import Baselines._paths  # noqa: F401
from Baselines.controllers import BaseController
from Baselines.dynamics import MAX_STEERING, observation, observation_dim
from Baselines.nets import RunningNorm
from utility_model import TrafficAgent

if TYPE_CHECKING:  # pragma: no cover
    from Baselines.scenario import Scenario

try:
    import torch
    import torch.nn as nn
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "PyTorch is required for the pure-RL baseline. Install with: pip install torch"
    ) from exc

DEFAULT_CHECKPOINT = Path("Baselines/checkpoints/pure_rl_policy.pt")
ACTION_DIM = 2


class PureRLPolicy(nn.Module):
    """Shared actor-critic pi(o) -> (accel, steering), tanh-bounded Gaussian."""

    def __init__(
        self,
        obs_dim: int,
        hidden_dim: int = 128,
        max_accel: float = 4.0,
        max_steering: float = MAX_STEERING,
        init_log_std: float = -1.6,
    ):
        super().__init__()
        self.obs_dim = int(obs_dim)
        self.hidden_dim = int(hidden_dim)
        scale = torch.tensor([float(max_accel), float(max_steering)], dtype=torch.float32)
        self.register_buffer("action_scale", scale)
        self.obs_norm = RunningNorm(self.obs_dim)

        self.actor = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, ACTION_DIM),
            nn.Tanh(),
        )
        self.critic = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )
        self.log_std = nn.Parameter(torch.full((ACTION_DIM,), float(init_log_std)))

    def forward(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the mean action in normalised [-1, 1] space and the value."""
        obs_n = self.obs_norm(obs)
        return self.actor(obs_n), self.critic(obs_n).squeeze(-1)

    def distribution(self, obs: torch.Tensor) -> tuple["torch.distributions.Normal", torch.Tensor]:
        mean, value = self.forward(obs)
        std = torch.exp(self.log_std).expand_as(mean)
        return torch.distributions.Normal(mean, std), value

    def to_control(self, action_norm: np.ndarray) -> tuple[float, float]:
        """Normalised action -> physical (accel, steering)."""
        scale = self.action_scale.detach().cpu().numpy()
        control = np.clip(action_norm, -1.0, 1.0) * scale
        return float(control[0]), float(control[1])

    def sample_action(self, obs: np.ndarray) -> tuple[np.ndarray, float, float]:
        obs_t = torch.as_tensor(obs, dtype=torch.float32)
        with torch.no_grad():
            dist, value = self.distribution(obs_t)
            action = dist.sample()
            log_prob = dist.log_prob(action).sum()
        return action.cpu().numpy(), float(log_prob), float(value)

    def act(self, obs: np.ndarray, explore_std: float = 0.0) -> tuple[float, float]:
        obs_t = torch.as_tensor(obs, dtype=torch.float32)
        with torch.no_grad():
            mean, _ = self.forward(obs_t)
            if explore_std > 0.0:
                mean = torch.distributions.Normal(mean, torch.full_like(mean, explore_std)).sample()
        return self.to_control(mean.cpu().numpy())


def load_pure_rl_policy(checkpoint: Path, obs_dim: int) -> PureRLPolicy:
    blob = torch.load(checkpoint, map_location="cpu")
    policy = PureRLPolicy(
        obs_dim=int(blob.get("obs_dim", obs_dim)),
        hidden_dim=int(blob.get("hidden_dim", 128)),
        max_accel=float(blob.get("max_accel", 4.0)),
        max_steering=float(blob.get("max_steering", MAX_STEERING)),
    )
    try:
        policy.load_state_dict(blob["state_dict"])
    except RuntimeError as exc:
        raise RuntimeError(
            f"{checkpoint} was written by an incompatible version of PureRLPolicy. "
            f"Retrain it with: python -m Baselines.train_pure_rl --save {checkpoint}"
        ) from exc
    policy.eval()
    return policy


class PureRLController(BaseController):
    """End-to-end learned controller with no utility prior."""

    name = "pure_rl"

    def __init__(
        self,
        checkpoint: Path | str | None = DEFAULT_CHECKPOINT,
        policy: PureRLPolicy | None = None,
        explore_std: float = 0.0,
        name: str | None = None,
    ):
        self.checkpoint = Path(checkpoint) if checkpoint is not None else None
        self.policy = policy
        self.explore_std = float(explore_std)
        if name:
            self.name = name
        self._warned = False

    def reset(self, scenario: "Scenario") -> None:
        if self.policy is not None or self.checkpoint is None:
            return
        if not self.checkpoint.exists():
            if not self._warned:
                print(
                    f"[pure_rl] checkpoint {self.checkpoint} not found; using an untrained "
                    "policy. Train it with: python -m Baselines.train_pure_rl"
                )
                self._warned = True
            self.policy = PureRLPolicy(observation_dim(scenario))
            self.policy.eval()
            return
        self.policy = load_pure_rl_policy(self.checkpoint, observation_dim(scenario))

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
            obs = observation(agents, i, scenario)
            controls.append(self.policy.act(obs, self.explore_std))
        return controls
