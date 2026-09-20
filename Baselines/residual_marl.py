"""Proposed model: calibrated utility prior + learned residual.

Default residual interface adds a continuous bias to the discrete utility grid
(``candidate_logits``). Legacy checkpoints that edit Θ (``param_delta``) still load.
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
from RL.decision import select_best_candidate, select_candidate_with_logit_residual
from utility_model import (
    TrafficAgent,
)

if TYPE_CHECKING:  # pragma: no cover
    from Baselines.scenario import Scenario

DEFAULT_CHECKPOINT = Path("RL/checkpoints/revision5/residual_policy.pt")


def build_random_residual_policy(
    obs_dim: int,
    *,
    seed: int = 0,
    highway_length: float = 500.0,
    template: Path | None = None,
) -> Any:
    """Untrained residual with the paper architecture; actor head is not zeroed.

    ``TorchResidualPolicy`` zeroes the last layer so a fresh network reproduces
    the utility argmax. This control redraws every actor Linear so the gate is
    tested against unstructured residual noise rather than the prior itself.
    """
    import torch
    from torch import nn

    from RL.train_ppo import (
        CANDIDATE_LOGIT_SCALE,
        CATEGORICAL_ACTION_SPACE,
        TorchResidualPolicy,
        action_space_from_blob,
        residual_mode_from_blob,
    )

    hidden_dim = 128
    action_dim = 63
    action_space = CATEGORICAL_ACTION_SPACE
    residual_mode = "candidate_logits"
    candidate_logit_scale = CANDIDATE_LOGIT_SCALE
    if template is not None and Path(template).exists():
        blob = torch.load(template, map_location="cpu")
        obs_dim = int(blob.get("obs_dim", obs_dim))
        hidden_dim = int(blob.get("hidden_dim", hidden_dim))
        action_dim = int(blob.get("action_dim", action_dim) or action_dim)
        action_space = action_space_from_blob(blob)
        residual_mode = residual_mode_from_blob(blob)
        candidate_logit_scale = float(blob.get("candidate_logit_scale", candidate_logit_scale))
        highway_length = float(blob.get("highway_length", highway_length))
    torch.manual_seed(int(seed))
    policy = TorchResidualPolicy(
        obs_dim=int(obs_dim),
        hidden_dim=hidden_dim,
        highway_length=float(highway_length),
        action_space=action_space,
        residual_mode=residual_mode,
        action_dim=action_dim,
        candidate_logit_scale=candidate_logit_scale,
    )
    for module in policy.actor.modules():
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                bound = 1.0 / max(1, int(module.out_features)) ** 0.5
                nn.init.uniform_(module.bias, -bound, bound)
    policy.eval()
    return policy


def load_residual_policy(checkpoint: Path, obs_dim: int, *, allow_legacy: bool = False) -> Any:
    """Rebuild a `TorchResidualPolicy` from a training checkpoint."""
    import torch

    from RL.train_ppo import (
        TorchResidualPolicy,
        action_space_from_blob,
        residual_mode_from_blob,
    )

    blob = torch.load(checkpoint, map_location="cpu")
    if not allow_legacy:
        from RL.protocol import validate_checkpoint
        validate_checkpoint(blob, obs_dim, checkpoint)
    scales = residual_scales_for_checkpoint(blob)
    gauge = str(
        blob.get(
            "param_gauge",
            "additive" if "S_v" in (blob.get("residual_scales") or {}) else "logit_simplex",
        )
    )
    residual_mode = residual_mode_from_blob(blob)
    action_dim = blob.get("action_dim")
    if action_dim is None and residual_mode == "candidate_logits":
        action_dim = len(blob.get("residual_scales") or {})
    policy = TorchResidualPolicy(
        obs_dim=int(blob.get("obs_dim", obs_dim)),
        hidden_dim=int(blob.get("hidden_dim", 128)),
        residual_scales=scales,
        highway_length=float(blob.get("highway_length", 500.0)),
        param_gauge=gauge,
        action_space=action_space_from_blob(blob),
        residual_mode=residual_mode,
        action_dim=int(action_dim) if action_dim is not None else None,
    )
    missing, unexpected = policy.load_state_dict(blob["state_dict"], strict=False)
    stale = [k for k in missing if not k.startswith("value_")]
    if stale or unexpected:
        raise ValueError(
            f"Checkpoint {checkpoint} does not match the policy: "
            f"missing={stale}, unexpected={list(unexpected)}"
        )
    policy.training_base_params = blob.get("base_params")
    policy.selection_status = blob.get("selection_status", "unreported")
    policy.training_revision = blob.get("training_revision", 0)
    policy.eval()
    return policy


class ResidualMARLController(BaseController):
    """Utility prior + residual (candidate logits by default, or param ΔΘ)."""

    def __init__(
        self,
        checkpoint: Path | str | None = DEFAULT_CHECKPOINT,
        calibration: Path | None = None,
        prefer: str = "robust",
        explore_std: float = 0.0,
        freeze_keys: tuple[str, ...] | list[str] | None = None,
        name: str = "residual_marl",
        accept_if_better: bool = True,
        random_init: bool = False,
        random_seed: int = 0,
    ):
        self.base_params = load_base_params(calibration, prefer=prefer)
        self.checkpoint = Path(checkpoint) if checkpoint is not None else None
        self.explore_std = float(explore_std)
        self.freeze_keys = tuple(freeze_keys or ())
        self.policy = None
        self.name = name
        self.accept_if_better = bool(accept_if_better)
        self.random_init = bool(random_init)
        self.random_seed = int(random_seed)
        self._warned = False

    def reset(self, scenario: "Scenario") -> None:
        if self.policy is not None:
            return
        from Baselines.dynamics import observation_dim

        obs_dim = observation_dim(scenario)
        if self.random_init:
            template = self.checkpoint
            if template is None:
                seed_path = DEFAULT_CHECKPOINT.with_name(
                    f"{DEFAULT_CHECKPOINT.stem}_seed{self.random_seed}{DEFAULT_CHECKPOINT.suffix}"
                )
                if seed_path.exists():
                    template = seed_path
                elif DEFAULT_CHECKPOINT.exists():
                    template = DEFAULT_CHECKPOINT
            highway = float(getattr(getattr(scenario, "corridor", None), "length", 500.0) or 500.0)
            self.policy = build_random_residual_policy(
                obs_dim,
                seed=self.random_seed,
                highway_length=highway,
                template=template,
            )
            return
        if self.checkpoint is None:
            return
        if not self.checkpoint.exists():
            raise FileNotFoundError(f"{self.name}: missing {self.checkpoint}; train this model before evaluation")
        self.policy = load_residual_policy(self.checkpoint, obs_dim)
        if self.freeze_keys and self.policy.residual_mode != "param_delta":
            raise ValueError("Parameter ablations require a param_delta checkpoint; candidate_logits masks would be no-ops")
        trained_base = self.policy.training_base_params
        if trained_base is not None and trained_base != self.base_params:
            raise ValueError("Evaluation prior differs from the checkpoint's frozen training prior")

    def _effective_freeze_keys(self) -> tuple[str, ...]:
        if not self.freeze_keys:
            return ()
        if self.policy is not None and getattr(self.policy, "param_gauge", "") == "additive":
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
        mode = getattr(self.policy, "residual_mode", "param_delta") if self.policy else "param_delta"
        for i, agent in enumerate(agents):
            if agent.reached_destination:
                controls.append((0.0, 0.0))
                continue
            if self.policy is None:
                chosen = select_best_candidate(
                    i, agent, agents, self.base_params, scenario.sim_config
                )
            elif mode == "candidate_logits":
                from RL.closed_loop_score import residual_proposal_improves
                obs = observation(agents, i, scenario)
                residual, _ = self.policy.act(np.asarray(obs, dtype=np.float32), self.explore_std)
                chosen, idx, prior_idx = select_candidate_with_logit_residual(
                    i,
                    agent,
                    agents,
                    self.base_params,
                    scenario.sim_config,
                    logit_residual=np.asarray(residual, dtype=float),
                )
                if self.accept_if_better and idx != prior_idx:
                    prior, _, _ = select_candidate_with_logit_residual(
                        i, agent, agents, self.base_params, scenario.sim_config, None)
                    if not residual_proposal_improves(
                        agent, chosen, prior, agents, i, scenario.sim_config, scenario.corridor):
                        chosen = prior
            else:
                obs = observation(agents, i, scenario)
                delta, _ = self.policy.act(np.asarray(obs, dtype=np.float32), self.explore_std)
                if not isinstance(delta, dict):
                    delta = {}
                # Inference-time ablations only apply to param residuals.
                params = self._apply_delta(delta)
                chosen = select_best_candidate(i, agent, agents, params, scenario.sim_config)
            controls.append(
                (
                    float(chosen.get("accel_longitudinal", 0.0)),
                    float(chosen.get("steering_angle", 0.0)),
                )
            )
        return controls
