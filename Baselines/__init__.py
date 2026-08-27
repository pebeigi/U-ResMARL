"""Benchmark suite: baselines compared against the residual MARL model.

Every model is driven through the same scenarios, the same bicycle dynamics and
the same metrics; the RL package is imported read-only and never modified.
"""

from __future__ import annotations

import Baselines._paths  # noqa: F401
from Baselines.controllers import BaseController, Controller
from Baselines.metrics import aggregate, metrics_frame, rollout_metrics, to_latex
from Baselines.registry import DEFAULT_MODELS, LABELS, REGISTRY, build_controller
from Baselines.runner import RolloutResult, rollout
from Baselines.scenario import Scenario, build_scenario, build_scenarios

__all__ = [
    "BaseController",
    "Controller",
    "DEFAULT_MODELS",
    "LABELS",
    "REGISTRY",
    "RolloutResult",
    "Scenario",
    "aggregate",
    "build_controller",
    "build_scenario",
    "build_scenarios",
    "metrics_frame",
    "rollout",
    "rollout_metrics",
    "to_latex",
]
