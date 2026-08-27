"""Proposed model: calibrated utility prior + learned residual on the parameters.

Loads a policy trained by `RL/train_ppo.py`; the RL package itself is untouched.
If no checkpoint is given the controller falls back to zero residual, which
reduces exactly to the utility-only prior.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

import Baselines._paths  # noqa: F401
from Baselines.controllers import BaseController
from Baselines.dynamics import observation
from RL.calibration_io import apply_residual, load_base_params, residual_scales_for_checkpoint
from RL.param_gauge import AMPLITUDE_KEYS, AMPLITUDE_LOGIT_KEYS
from utility_model import TrafficAgent, select_best_candidate

if TYPE_CHECKING:  # pragma: no cover
    from Baselines.scenario import Scenario

DEFAULT_CHECKPOINT = Path("RL/checkpoints/residual_policy.pt")


def load_residual_policy(checkpoint: Path, obs_dim: int) -> Any:
    """Rebuild a `TorchResidualPolicy` from a training checkpoint."""
    import torch

    from RL.train_ppo import TorchResidualPolicy

    blob = torch.load(checkpoint, map_location="cpu")
    scales = residual_scales_for_checkpoint(blob)
    gauge = str(blob.get("param_gauge", "additive" if "S_v" in (blob.get("residual_scales") or {}) else "logit_simplex"))
    policy = TorchResidualPolicy(
        obs_dim=int(blob.get("obs_dim", obs_dim)),
        hidden_dim=int(blob.get("hidden_dim", 128)),
        residual_scales=scales,
        highway_length=float(blob.get("highway_length", 500.0)),
        param_gauge=gauge,
    )
    policy.load_state_dict(blob["state_dict"])
    policy.eval()
    return policy


class ResidualMARLController(BaseController):
    """U(a; gauge(Theta_base, Delta(o))) with shared PPO residual on logits + shape."""

    def __init__(
        self,
        checkpoint: Path | str | None = DEFAULT_CHECKPOINT,
        calibration: Path | None = None,
        prefer: str = "robust",
        explore_std: float = 0.0,
        freeze_keys: tuple[str, ...] | list[str] | None = None,
        name: str = "residual_marl",
    ):
        self.base_params = load_base_params(calibration, prefer=prefer)
        self.checkpoint = Path(checkpoint) if checkpoint is not None else None
        self.explore_std = float(explore_std)
        self.freeze_keys = tuple(freeze_keys or ())
        self.policy = None
        self.name = name
        self._warned = False

    def reset(self, scenario: "Scenario") -> None:
        if self.policy is not None or self.checkpoint is None:
            return
        if not self.checkpoint.exists():
            if not self._warned:
                print(
                    f"[{self.name}] checkpoint {self.checkpoint} not found; "
                    "running with zero residual (== utility prior)."
                )
                self._warned = True
            return
        from Baselines.dynamics import observation_dim

        self.policy = load_residual_policy(self.checkpoint, observation_dim(scenario))

    def _effective_freeze_keys(self) -> tuple[str, ...]:
        if not self.freeze_keys:
            return ()
        if self.policy is not None and self.policy.param_gauge == "additive":
            logit_to_amp = dict(zip(AMPLITUDE_LOGIT_KEYS, AMPLITUDE_KEYS))
            return tuple(logit_to_amp.get(key, key) for key in self.freeze_keys)
        return self.freeze_keys

    def _apply_delta(self, delta: dict[str, float]) -> dict[str, float]:
        freeze_keys = self._effective_freeze_keys()
        if freeze_keys:
            delta = dict(delta)
            for key in freeze_keys:
                delta[key] = 0.0
        return apply_residual(
            self.base_params,
            delta,
            legacy=(self.policy.param_gauge == "additive") if self.policy else None,
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
            if self.policy is not None:
                obs = observation(agents, i, scenario)
                delta, _ = self.policy.act(np.asarray(obs, dtype=np.float32), self.explore_std)
                params = self._apply_delta(delta)
            else:
                params = self.base_params
            chosen = select_best_candidate(i, agent, agents, params, scenario.sim_config)
            controls.append(
                (
                    float(chosen.get("accel_longitudinal", 0.0)),
                    float(chosen.get("steering_angle", 0.0)),
                )
            )
        return controls
