"""Shared discrete (accel, steering) grid used by the utility and direct-RL policies."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

import Baselines._paths  # noqa: F401
from utility_model import (
    TrafficAgent,
    build_step_context,
    candidate_obb_conflict,
    kinematic_bicycle_rollout,
)

if TYPE_CHECKING:  # pragma: no cover
    from Baselines.scenario import Scenario


def accel_grid(sim_config: dict[str, Any]) -> list[float]:
    return [float(x) for x in sim_config.get("candidate_accel_grid", [-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0])]


def steering_grid(sim_config: dict[str, Any]) -> list[float]:
    return [
        float(x)
        for x in sim_config.get(
            "candidate_steering_grid",
            [-0.45, -0.3375, -0.225, -0.1125, 0.0, 0.1125, 0.225, 0.3375, 0.45],
        )
    ]


def num_grid_actions(sim_config: dict[str, Any]) -> int:
    return len(accel_grid(sim_config)) * len(steering_grid(sim_config))


def grid_index(accel: float, steering: float, sim_config: dict[str, Any]) -> int:
    accels = accel_grid(sim_config)
    steerings = steering_grid(sim_config)
    ai = min(range(len(accels)), key=lambda i: abs(accels[i] - float(accel)))
    si = min(range(len(steerings)), key=lambda i: abs(steerings[i] - float(steering)))
    return ai * len(steerings) + si


def grid_control(index: int, sim_config: dict[str, Any]) -> tuple[float, float]:
    steerings = steering_grid(sim_config)
    ai, si = divmod(int(index), len(steerings))
    return accel_grid(sim_config)[ai], steerings[si]


def grid_candidate(
    agent: TrafficAgent,
    index: int,
    dt: float,
    sim_config: dict[str, Any],
) -> dict[str, Any]:
    accel, steering = grid_control(index, sim_config)
    return kinematic_bicycle_rollout(
        agent.pos,
        float(agent.heading),
        float(agent.speed),
        float(accel),
        float(steering),
        float(dt),
        sim_config,
    )


def feasible_action_mask(
    agent_idx: int,
    agent: TrafficAgent,
    agents: list[TrafficAgent],
    scenario: "Scenario",
) -> np.ndarray:
    """Boolean mask over the fixed (accel, steering) grid with OBB conflict rejection."""
    sim_config = scenario.sim_config
    n = num_grid_actions(sim_config)
    mask = np.zeros(n, dtype=bool)
    context = build_step_context(agent_idx, agent, agents, sim_config)
    for k in range(n):
        cand = grid_candidate(agent, k, scenario.dt, sim_config)
        if not candidate_obb_conflict(cand, agent_idx, agents, sim_config, context=context):
            mask[k] = True
    if not mask.any():
        mask[:] = True
    return mask
