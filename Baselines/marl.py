"""Cooperative MARL baselines: MAPPO, HAPPO and HATRPO.

References:
- Yu et al., "The Surprising Effectiveness of PPO in Cooperative Multi-Agent
  Games" (MAPPO), NeurIPS 2022 Datasets & Benchmarks.
- Kuba et al., "Trust Region Policy Optimisation in Multi-Agent Reinforcement
  Learning" (HATRPO / HAPPO), ICLR 2022.

All three share the actor architecture and the reward with the pure-RL (IPPO)
baseline, so the comparison isolates the algorithm:

| Algorithm | Actors | Critic input | Update |
| --- | --- | --- | --- |
| IPPO (`pure_rl`) | shared | local observation | PPO-clip, simultaneous |
| MAPPO | shared | centralised state | PPO-clip, simultaneous |
| HAPPO | one per agent | centralised state | PPO-clip, sequential with the multi-agent advantage factor |
| HATRPO | one per agent | centralised state | KL trust region, sequential with the same factor |
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

import Baselines._paths  # noqa: F401
from Baselines.controllers import BaseController
from Baselines.dynamics import MAX_STEERING, observation, observation_dim, project_and_clearances
from Baselines.nets import RunningNorm, mlp
from utility_model import TrafficAgent

if TYPE_CHECKING:  # pragma: no cover
    from Baselines.scenario import Scenario

try:
    import torch
    import torch.nn as nn
except ImportError as exc:  # pragma: no cover
    raise SystemExit("PyTorch is required for the MARL baselines. Install with: pip install torch") from exc

ACTION_DIM = 2
FEATURES_PER_AGENT = 6
ALGORITHMS = ("ippo", "mappo", "happo", "hatrpo")
SEQUENTIAL_ALGORITHMS = ("happo", "hatrpo")

DEFAULT_CHECKPOINTS = {
    "mappo": Path("Baselines/checkpoints/mappo_policy.pt"),
    "happo": Path("Baselines/checkpoints/happo_policy.pt"),
    "hatrpo": Path("Baselines/checkpoints/hatrpo_policy.pt"),
}


def agent_features(agents: list[TrafficAgent], scenario: "Scenario") -> np.ndarray:
    """Per-agent corridor state used to build the centralised state: (n, 6)."""
    corridor_length = float(scenario.corridor.length)
    max_speed = float(scenario.sim_config.get("max_agent_speed", 16.0))
    out = np.zeros((len(agents), FEATURES_PER_AGENT), dtype=np.float32)
    for i, agent in enumerate(agents):
        station, lateral, _, _, _ = project_and_clearances(scenario.corridor, agent.pos)
        out[i] = (
            station / corridor_length,
            lateral,
            agent.speed / max_speed,
            float(np.cos(agent.heading)),
            float(np.sin(agent.heading)),
            float(agent.reached_destination),
        )
    return out


def centralised_state(features: np.ndarray, obs: np.ndarray, num_agents: int) -> np.ndarray:
    """Agent-specific centralised state: all agents' corridor state + own observation.

    The joint part is padded or truncated to the number of agents the critic was
    trained with, so a policy can be evaluated at a different traffic density.
    """
    joint = np.zeros((num_agents, FEATURES_PER_AGENT), dtype=np.float32)
    take = min(num_agents, features.shape[0])
    joint[:take] = features[:take]
    return np.concatenate([joint.ravel(), np.asarray(obs, dtype=np.float32)])


def state_dimension(obs_dim: int, num_agents: int) -> int:
    return FEATURES_PER_AGENT * num_agents + obs_dim


class Actor(nn.Module):
    """pi(o) -> Gaussian over normalised actions in [-1, 1]^2."""

    def __init__(self, obs_dim: int, hidden_dim: int = 128, init_log_std: float = -1.6):
        super().__init__()
        self.body = mlp(obs_dim, hidden_dim, ACTION_DIM, final_tanh=True)
        self.log_std = nn.Parameter(torch.full((ACTION_DIM,), float(init_log_std)))

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.body(obs)

    def distribution(self, obs: torch.Tensor) -> "torch.distributions.Normal":
        mean = self.forward(obs)
        return torch.distributions.Normal(mean, torch.exp(self.log_std).expand_as(mean))


class MARLPolicy(nn.Module):
    """Container for the actors, the (centralised) critic and the normalisers."""

    def __init__(
        self,
        obs_dim: int,
        num_agents: int,
        algo: str = "mappo",
        hidden_dim: int = 128,
        max_accel: float = 4.0,
        max_steering: float = MAX_STEERING,
        init_log_std: float = -1.6,
    ):
        super().__init__()
        if algo not in ALGORITHMS:
            raise ValueError(f"algo must be one of {ALGORITHMS}, got {algo!r}")
        self.algo = algo
        self.obs_dim = int(obs_dim)
        self.num_agents = int(num_agents)
        self.hidden_dim = int(hidden_dim)
        self.centralised = algo != "ippo"
        self.state_dim = state_dimension(self.obs_dim, self.num_agents) if self.centralised else self.obs_dim

        n_actors = self.num_agents if algo in SEQUENTIAL_ALGORITHMS else 1
        self.actors = nn.ModuleList(
            [Actor(self.obs_dim, hidden_dim, init_log_std) for _ in range(n_actors)]
        )
        self.critic = mlp(self.state_dim, hidden_dim, 1)
        self.obs_norm = RunningNorm(self.obs_dim)
        self.state_norm = RunningNorm(self.state_dim)
        self.register_buffer(
            "action_scale", torch.tensor([float(max_accel), float(max_steering)], dtype=torch.float32)
        )

    @property
    def per_agent_actors(self) -> bool:
        return len(self.actors) > 1

    def actor_for(self, agent_idx: int) -> Actor:
        if not self.per_agent_actors:
            return self.actors[0]
        return self.actors[agent_idx % len(self.actors)]

    def value(self, state: torch.Tensor) -> torch.Tensor:
        return self.critic(self.state_norm(state)).squeeze(-1)

    def distribution(self, obs: torch.Tensor, agent_idx: int) -> "torch.distributions.Normal":
        return self.actor_for(agent_idx).distribution(self.obs_norm(obs))

    def to_control(self, action_norm: np.ndarray) -> tuple[float, float]:
        scale = self.action_scale.detach().cpu().numpy()
        control = np.clip(action_norm, -1.0, 1.0) * scale
        return float(control[0]), float(control[1])

    def sample_action(
        self, obs: np.ndarray, state: np.ndarray, agent_idx: int
    ) -> tuple[np.ndarray, float, float]:
        obs_t = torch.as_tensor(obs, dtype=torch.float32)
        state_t = torch.as_tensor(state, dtype=torch.float32)
        with torch.no_grad():
            dist = self.distribution(obs_t, agent_idx)
            action = dist.sample()
            log_prob = dist.log_prob(action).sum()
            value = self.value(state_t)
        return action.cpu().numpy(), float(log_prob), float(value)

    def act(self, obs: np.ndarray, agent_idx: int) -> tuple[float, float]:
        obs_t = torch.as_tensor(obs, dtype=torch.float32)
        with torch.no_grad():
            mean = self.actor_for(agent_idx)(self.obs_norm(obs_t))
        return self.to_control(mean.cpu().numpy())


def save_marl_policy(policy: MARLPolicy, path: Path, extra: dict | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    blob = {
        "state_dict": policy.state_dict(),
        "obs_dim": policy.obs_dim,
        "num_agents": policy.num_agents,
        "hidden_dim": policy.hidden_dim,
        "algo": policy.algo,
        "max_accel": float(policy.action_scale[0].item()),
        "max_steering": float(policy.action_scale[1].item()),
    }
    blob.update(extra or {})
    torch.save(blob, path)


def load_marl_policy(checkpoint: Path, obs_dim: int) -> MARLPolicy:
    blob = torch.load(checkpoint, map_location="cpu")
    policy = MARLPolicy(
        obs_dim=int(blob.get("obs_dim", obs_dim)),
        num_agents=int(blob["num_agents"]),
        algo=str(blob.get("algo", "mappo")),
        hidden_dim=int(blob.get("hidden_dim", 128)),
        max_accel=float(blob.get("max_accel", 4.0)),
        max_steering=float(blob.get("max_steering", MAX_STEERING)),
    )
    try:
        policy.load_state_dict(blob["state_dict"])
    except RuntimeError as exc:
        raise RuntimeError(
            f"{checkpoint} is incompatible with the current MARLPolicy. Retrain with: "
            f"python -m Baselines.train_marl --algo {blob.get('algo', 'mappo')}"
        ) from exc
    policy.eval()
    return policy


class MARLController(BaseController):
    """Evaluation wrapper for a trained MAPPO / HAPPO / HATRPO policy."""

    def __init__(
        self,
        algo: str = "mappo",
        checkpoint: Path | str | None = None,
        policy: MARLPolicy | None = None,
        name: str | None = None,
    ):
        if algo not in ALGORITHMS:
            raise ValueError(f"algo must be one of {ALGORITHMS}, got {algo!r}")
        self.algo = algo
        self.checkpoint = Path(checkpoint) if checkpoint is not None else DEFAULT_CHECKPOINTS.get(algo)
        self.policy = policy
        self.name = name or algo
        self._warned = False

    def reset(self, scenario: "Scenario") -> None:
        if self.policy is not None or self.checkpoint is None:
            return
        if not self.checkpoint.exists():
            if not self._warned:
                print(
                    f"[{self.name}] checkpoint {self.checkpoint} not found; using an untrained "
                    f"policy. Train it with: python -m Baselines.train_marl --algo {self.algo}"
                )
                self._warned = True
            self.policy = MARLPolicy(
                obs_dim=observation_dim(scenario),
                num_agents=scenario.num_agents,
                algo=self.algo,
            )
            self.policy.eval()
            return
        self.policy = load_marl_policy(self.checkpoint, observation_dim(scenario))

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
            controls.append(self.policy.act(observation(agents, i, scenario), i))
        return controls
