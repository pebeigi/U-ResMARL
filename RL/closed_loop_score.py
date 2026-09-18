"""NAVSIM/PDM-style closed-loop scoring for episodes and one-step proposals.

Episode scores follow NAVSIM PDMS: collision and off-road are multiplicative
zeros; progress, TTC and comfort are a weighted average. Comfort is a band, not
a request to minimize jerk below the utility prior. Proposal scores use the same
structure on a single kinematic candidate so a residual action is executed only
when it beats the frozen prior on that local score.
"""
from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import numpy as np

# NAVSIM / nuPlan-style weights and comfort bands (Dauner et al., NAVSIM).
PROGRESS_WEIGHT = 5.0
TTC_WEIGHT = 5.0
COMFORT_WEIGHT = 2.0
COMFORT_ABS_ACCEL = 2.40  # m/s^2
COMFORT_RMS_JERK = 8.37  # m/s^3
COMFORT_ABS_STEER = 0.10  # rad; mean |steer| band, not the 0.45 actuator limit
UNSAFE_TTC_RATE_CAP = 0.05
SOFT_PENALTY_FLOOR = 0.1
PROPOSAL_STEER_BOUND = 0.3375


def _weighted_average(progress: float, ttc: float, comfort: float) -> float:
    total = PROGRESS_WEIGHT + TTC_WEIGHT + COMFORT_WEIGHT
    return float((PROGRESS_WEIGHT * progress + TTC_WEIGHT * ttc + COMFORT_WEIGHT * comfort) / total)


def comfort_subscore(stats: dict[str, Any]) -> float:
    """1 if reported comfort stats stay in-band, else 0. Missing keys do not fail."""
    if "mean_abs_accel" in stats and float(stats["mean_abs_accel"]) > COMFORT_ABS_ACCEL:
        return 0.0
    if "rms_jerk" in stats and float(stats["rms_jerk"]) > COMFORT_RMS_JERK:
        return 0.0
    if "mean_abs_steering" in stats and float(stats["mean_abs_steering"]) > COMFORT_ABS_STEER:
        return 0.0
    return 1.0


def ttc_subscore(stats: dict[str, Any]) -> float:
    rate = float(stats.get("unsafe_ttc_rate", 0.0))
    return float(np.clip(1.0 - rate / UNSAFE_TTC_RATE_CAP, 0.0, 1.0))


def closed_loop_score(stats: dict[str, Any]) -> float:
    """PDMS-like scalar in [0, 1] from closed-loop episode or aggregate stats."""
    events = float(stats.get("collision_events", stats.get("collisions", 0.0)))
    nc = 0.0 if events > 1e-12 else 1.0
    dac = 0.0 if float(stats.get("offroad_rate", 0.0)) > 1e-12 else 1.0
    progress = stats.get("goal_progress")
    if progress is None:
        progress = stats.get("arrival_rate", 0.0)
    progress = float(np.clip(progress, 0.0, 1.0))
    return float(nc * dac * _weighted_average(progress, ttc_subscore(stats), comfort_subscore(stats)))


def soft_safety_factor(safety_cost: float, weight: float = 1.0) -> float:
    """Map additive proximity cost (<= 0) to a CaRL-style multiplier in (0, 1]."""
    if weight <= 0.0:
        return 1.0
    return float(np.clip(1.0 + weight * float(safety_cost) / 16.0, SOFT_PENALTY_FLOOR, 1.0))


def soft_comfort_factor(accel: float, steering: float, weight: float = 1.0) -> float:
    """Binary in-band comfort; out-of-band multiplies progress but stays > 0."""
    if weight <= 0.0:
        return 1.0
    in_band = abs(float(accel)) <= COMFORT_ABS_ACCEL and abs(float(steering)) <= PROPOSAL_STEER_BOUND
    if in_band:
        return 1.0
    return float(max(SOFT_PENALTY_FLOOR, 1.0 - 0.5 * weight))


def proposal_closed_loop_score(
    agent,
    candidate: dict[str, Any],
    agents,
    agent_idx: int,
    sim: dict[str, Any],
    corridor,
) -> float:
    """One-step PDM score of a kinematic candidate against the current scene."""
    from RL.boundary import candidate_boundary_safe
    from RL.obs import footprint_surface_gap
    from utility_model import build_step_context, candidate_obb_conflict

    if not candidate_boundary_safe(agent, candidate, sim, corridor):
        return 0.0
    context = build_step_context(agent_idx, agent, agents, sim)
    nc = 0.0 if candidate_obb_conflict(candidate, agent_idx, agents, sim, context=context) else 1.0
    dt = max(float(sim["dt"]), 1e-6)
    speed_cap = max(float(sim.get("max_agent_speed", 16.0)), 1e-6)
    s0 = float(corridor.project(agent.pos)[0])
    s1 = float(corridor.project(np.asarray(candidate["pos"], dtype=float))[0])
    progress = float(np.clip((s1 - s0) / (speed_cap * dt), 0.0, 1.0))
    ego_next = SimpleNamespace(pos=np.asarray(candidate["pos"], dtype=float),
                               heading=float(candidate["heading"]))
    length = float(sim.get("vehicle_length", 4.5))
    width = float(sim.get("vehicle_width", 1.8))
    ttc = 1.0
    for j, other in enumerate(agents):
        if j == agent_idx or other.reached_destination:
            continue
        gap = footprint_surface_gap(ego_next, other, length, width)
        if gap <= 0.0:
            ttc = 0.0
            break
        ttc = min(ttc, float(np.clip(gap / 2.0, 0.0, 1.0)))
    comfort = 1.0 if (
        abs(float(candidate.get("accel_longitudinal", 0.0))) <= COMFORT_ABS_ACCEL
        and abs(float(candidate.get("steering_angle", 0.0))) <= PROPOSAL_STEER_BOUND
    ) else 0.0
    return float(nc * _weighted_average(progress, ttc, comfort))


def residual_proposal_improves(
    agent,
    residual_candidate: dict[str, Any],
    prior_candidate: dict[str, Any],
    agents,
    agent_idx: int,
    sim: dict[str, Any],
    corridor,
) -> bool:
    """True only when the residual proposal strictly beats the utility proposal."""
    residual = proposal_closed_loop_score(
        agent, residual_candidate, agents, agent_idx, sim, corridor)
    prior = proposal_closed_loop_score(
        agent, prior_candidate, agents, agent_idx, sim, corridor)
    return residual > prior + 1e-12
