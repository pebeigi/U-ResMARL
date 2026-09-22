"""Utility-only baseline: the calibrated prospect-theory prior with no learning.

This is the "stochastic utility-based (PT)" comparison in the paper. With
`temperature=0` the agent takes the argmax action (deterministic discrete
choice); with `temperature>0` it samples from a logit choice model over the
candidate set, which is the classical random-utility formulation.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

import Baselines._paths  # noqa: F401
from Baselines.controllers import BaseController
from RL.calibration_io import load_base_params
from RL.decision import select_best_candidate
from utility_model import (
    TrafficAgent,
    build_step_context,
    candidate_obb_conflict,
    evaluate_candidate_utility,
    generate_candidate_actions,
)

if TYPE_CHECKING:  # pragma: no cover
    from Baselines.scenario import Scenario


class UtilityPriorController(BaseController):
    """argmax (or logit choice) over discrete bicycle candidates under U(a; Theta)."""

    def __init__(
        self,
        params: dict[str, float] | None = None,
        calibration: Path | None = None,
        prefer: str = "working",
        temperature: float = 0.0,
        seed: int = 0,
        name: str = "utility_pt",
    ):
        self.params = dict(params) if params is not None else load_base_params(calibration, prefer=prefer)
        self.temperature = float(temperature)
        self.rng = np.random.default_rng(seed)
        self.name = name

    def reset(self, scenario: "Scenario") -> None:
        return None

    def _logit_choice(
        self,
        idx: int,
        agent: TrafficAgent,
        agents: list[TrafficAgent],
        scenario: "Scenario",
    ) -> tuple[float, float]:
        from utility_model import evaluate_candidate_utility
        from RL.decision import local_agents
        agents = local_agents(agents, idx, scenario.sim_config)
        idx = 0
        candidates = generate_candidate_actions(agent, scenario.dt, scenario.sim_config)
        context = build_step_context(idx, agent, agents, scenario.sim_config)
        utilities = np.array(
            [
                evaluate_candidate_utility(
                    idx, agent, c, agents, self.params, scenario.sim_config, context=context
                )
                for c in candidates
            ],
            dtype=float,
        )
        free_idx = [
            k
            for k, c in enumerate(candidates)
            if not scenario.sim_config.get("obb_safety_filter", True) or not candidate_obb_conflict(c, idx, agents, scenario.sim_config, context=context)
        ]
        pool = free_idx if free_idx else list(range(len(candidates)))
        utilities = utilities[pool]
        logits = utilities / max(self.temperature, 1e-6)
        logits -= logits.max()
        probs = np.exp(logits)
        probs /= probs.sum()
        chosen = candidates[pool[int(self.rng.choice(len(pool), p=probs))]]
        return float(chosen.get("accel_longitudinal", 0.0)), float(chosen.get("steering_angle", 0.0))

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
            if self.temperature > 0.0:
                controls.append(self._logit_choice(i, agent, agents, scenario))
                continue
            chosen = select_best_candidate(i, agent, agents, self.params, scenario.sim_config)
            controls.append(
                (
                    float(chosen.get("accel_longitudinal", 0.0)),
                    float(chosen.get("steering_angle", 0.0)),
                )
            )
        return controls
