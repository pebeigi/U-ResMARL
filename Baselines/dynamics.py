"""Shared kinematics, observations and reward used by every benchmarked model.

All controllers plug into the same bicycle integrator, the same oriented-box
collision test and the same corridor-progress arrival rule, so differences in
the metrics come from the policy and nothing else.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

import Baselines._paths  # noqa: F401
from RL.corridor import boundary_reward
from RL.obs import local_observation
from RL.transition import DEFAULT_REWARD_WEIGHTS, driving_reward
from utility_model import (
    TrafficAgent,
    build_step_context,
    candidate_obb_conflict,
    emergency_brake_command,
    kinematic_bicycle_rollout,
    sanitize_control_command,
)

if TYPE_CHECKING:  # pragma: no cover
    from Baselines.scenario import Scenario

MAX_STEERING = 0.45


def wrap_angle(angle: float) -> float:
    return float((angle + np.pi) % (2.0 * np.pi) - np.pi)


def project_and_clearances(corridor, point: np.ndarray) -> tuple[float, float, np.ndarray, float, float]:
    """Single corridor projection reused for both Frenet state and edge clearances."""
    s, lateral, tangent, seg_i, t = corridor.project(point)
    lower, upper = corridor.edge_points_at(seg_i, t)
    chord = upper - lower
    chord_len = float(np.linalg.norm(chord))
    if chord_len < 1e-6:
        return s, lateral, tangent, 0.0, 0.0
    unit = chord / chord_len
    mid = 0.5 * (lower + upper)
    signed = float((np.asarray(point, dtype=float) - mid) @ unit)
    half = 0.5 * chord_len
    return s, lateral, tangent, half + signed, half - signed


def neighbors_of(agents: list[TrafficAgent], idx: int, scenario: "Scenario") -> list[int]:
    """Nearest neighbours inside the perception radius (same rule as the RL env)."""
    ego = agents[idx]
    radius = float(scenario.sim_config["perception_radius"])
    max_n = int(scenario.sim_config["max_neighbors"])
    ranked: list[tuple[float, int]] = []
    for j, other in enumerate(agents):
        if j == idx or other.reached_destination:
            continue
        d = float(np.linalg.norm(other.pos - ego.pos))
        if d <= radius:
            ranked.append((d, j))
    ranked.sort(key=lambda x: x[0])
    return [j for _, j in ranked[:max_n]]


def observation(agents: list[TrafficAgent], idx: int, scenario: "Scenario") -> np.ndarray:
    """Frenet ego state plus body-frame neighbors (shared with residual training)."""
    dest_s = float(scenario.agents[idx].dest_s) if idx < len(scenario.agents) else None
    return local_observation(
        agents[idx],
        agents,
        neighbors_of(agents, idx, scenario),
        scenario.corridor,
        int(scenario.sim_config["max_neighbors"]),
        dest_s=dest_s,
    )


def observation_dim(scenario: "Scenario") -> int:
    from RL.obs import observation_dim as _obs_dim

    return _obs_dim(int(scenario.sim_config["max_neighbors"]))



def compute_reward(
    agents: list[TrafficAgent],
    idx: int,
    scenario: "Scenario",
    control: tuple[float, float],
    weights: dict[str, float] | None = None,
) -> float:
    """Shared pre-transition reward; terminal bonuses are added by advance_agents."""
    return driving_reward(agents, idx, scenario.corridor, scenario.sim_config,
                          scenario.agents[idx].dest_s, control, weights,
                          scenario.sim_config.get("leftover_coef", 0.05))


def control_from_bicycle(
    agent_idx: int,
    agent: TrafficAgent,
    agents: list[TrafficAgent],
    accel: float,
    steering: float,
    scenario: "Scenario",
) -> dict:
    """One-step bicycle candidate dict (same shape as utility_model candidates)."""
    return kinematic_bicycle_rollout(
        agent.pos,
        float(agent.heading),
        float(agent.speed),
        float(accel),
        float(np.clip(steering, -MAX_STEERING, MAX_STEERING)),
        float(scenario.dt),
        scenario.sim_config,
    )


def control_obb_conflict(
    agent_idx: int,
    agent: TrafficAgent,
    agents: list[TrafficAgent],
    accel: float,
    steering: float,
    scenario: "Scenario",
) -> bool:
    """True if the one-step bicycle command overlaps a CV-predicted neighbour OBB."""
    if not scenario.sim_config.get("obb_safety_filter", True):
        return False
    cand = control_from_bicycle(agent_idx, agent, agents, accel, steering, scenario)
    context = build_step_context(agent_idx, agent, agents, scenario.sim_config)
    return candidate_obb_conflict(
        cand, agent_idx, agents, scenario.sim_config, context=context
    )


def emergency_brake(agent: TrafficAgent, scenario: "Scenario") -> tuple[float, float]:
    """Hard deceleration used when the preferred command fails the OBB filter."""
    return emergency_brake_command(agent, scenario.sim_config)


def sanitize_control(
    agent_idx: int,
    agent: TrafficAgent,
    agents: list[TrafficAgent],
    control: tuple[float, float],
    scenario: "Scenario",
) -> tuple[float, float]:
    """Apply the shared closed-loop OBB safety filter to any controller command."""
    return sanitize_control_command(
        agent_idx, agent, agents, control, scenario.sim_config
    )


def lookahead_point(
    agent: TrafficAgent,
    scenario: "Scenario",
    dest_s: float,
    lookahead: float = 15.0,
) -> np.ndarray:
    """
    Carrot point on the corridor ahead of the agent.

    The corridor is not lane-structured and arrival is defined by along-track
    station, so the agent holds its own lateral offset instead of being pulled
    onto the centreline.
    """
    s, lateral, _, _, _ = scenario.corridor.project(agent.pos)
    target_s = min(s + lookahead, dest_s)
    point, _ = scenario.corridor.xy_from_frenet(target_s, lateral)
    return point


def preferred_velocity(
    agent: TrafficAgent,
    scenario: "Scenario",
    dest_s: float,
    lookahead: float = 15.0,
) -> np.ndarray:
    """Goal-directed velocity that follows corridor curvature."""
    target = lookahead_point(agent, scenario, dest_s, lookahead)
    delta = target - agent.pos
    norm = float(np.linalg.norm(delta))
    if norm < 1e-6:
        return np.zeros(2, dtype=float)
    s, _, _, _, _ = scenario.corridor.project(agent.pos)
    remaining = max(dest_s - s, 0.0)
    # Slow down smoothly over the last few metres so the agent stops at the goal.
    speed = min(agent.desired_speed, max(0.0, remaining) / max(scenario.dt, 1e-6))
    return (delta / norm) * speed


def velocity_to_control(
    agent: TrafficAgent,
    v_desired: np.ndarray,
    scenario: "Scenario",
    heading_gain: float = 1.0,
) -> tuple[float, float]:
    """Invert the bicycle model: desired velocity -> (accel, steering)."""
    dt = float(scenario.dt)
    sim = scenario.sim_config
    max_accel = float(sim.get("max_accel", 4.0))
    wheelbase = float(sim.get("wheelbase", 2.8))

    target_speed = float(np.linalg.norm(v_desired))
    accel = float(np.clip((target_speed - agent.speed) / max(dt, 1e-6), -max_accel, max_accel))

    if target_speed < 1e-3:
        return accel, 0.0

    target_heading = float(np.arctan2(v_desired[1], v_desired[0]))
    heading_error = wrap_angle(target_heading - agent.heading)
    yaw_rate = heading_gain * heading_error / max(dt, 1e-6)
    # psi_dot = (v / L) tan(delta)
    speed = max(agent.speed, 1e-3)
    steering = float(np.arctan(np.clip(yaw_rate * wheelbase / speed, -20.0, 20.0)))
    return accel, float(np.clip(steering, -MAX_STEERING, MAX_STEERING))


def apply_control(
    agent: TrafficAgent,
    control: tuple[float, float],
    scenario: "Scenario",
) -> dict[str, Any]:
    """Advance one agent by one step with the shared bicycle integrator."""
    accel, steering = control
    candidate = kinematic_bicycle_rollout(
        agent.pos,
        float(agent.heading),
        float(agent.speed),
        float(accel),
        float(np.clip(steering, -MAX_STEERING, MAX_STEERING)),
        float(scenario.dt),
        scenario.sim_config,
    )
    agent.update_state_from_candidate(
        candidate,
        float(scenario.dt),
        float(scenario.sim_config.get("destination_threshold", 1.0)),
    )
    return candidate


def goal_approach_control(agent: TrafficAgent, scenario: "Scenario", dest_s: float):
    """Slow near a goal while retaining enough speed to enter its arrival region."""
    s = float(scenario.corridor.project(agent.pos)[0])
    remaining = max(float(dest_s) - s, 0.0)
    if remaining >= max(agent.speed * scenario.dt * 2.0, 3.0):
        return None
    tol = float(scenario.sim_config.get("destination_threshold", 1.0))
    distance = max(remaining - 0.5 * tol, 0.0)
    max_accel = float(scenario.sim_config.get("max_accel", 4.0))
    target_speed = min(agent.desired_speed, distance / scenario.dt,
                       float(np.sqrt(2.0 * max_accel * distance)))
    velocity = preferred_velocity(agent, scenario, dest_s)
    norm = float(np.linalg.norm(velocity))
    if norm > 1e-9:
        velocity = velocity * target_speed / norm
    return velocity_to_control(agent, velocity, scenario)


def simulate_bicycle_batch(
    pos: np.ndarray,
    heading: float,
    speed: float,
    accels: np.ndarray,
    steerings: np.ndarray,
    scenario: "Scenario",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Roll out K control sequences with the same bicycle model used by the sim.

    `accels` and `steerings` are (K, H). Returns positions (K, H+1, 2), speeds
    (K, H+1) and headings (K, H+1).
    """
    sim = scenario.sim_config
    dt = float(scenario.dt)
    max_speed = float(sim["max_agent_speed"])
    max_accel = float(sim.get("max_accel", 4.0))
    wheelbase = float(sim.get("wheelbase", 2.8))

    k, horizon = accels.shape
    positions = np.empty((k, horizon + 1, 2), dtype=float)
    speeds = np.empty((k, horizon + 1), dtype=float)
    headings = np.empty((k, horizon + 1), dtype=float)

    positions[:, 0] = np.asarray(pos, dtype=float)
    speeds[:, 0] = float(speed)
    headings[:, 0] = float(heading)

    accels = np.clip(accels, -max_accel, max_accel)
    steerings = np.clip(steerings, -MAX_STEERING, MAX_STEERING)

    for h in range(horizon):
        v_prev = speeds[:, h]
        v_next = np.clip(v_prev + accels[:, h] * dt, 0.0, max_speed)
        yaw_rate = (v_prev / wheelbase) * np.tan(steerings[:, h])
        psi = headings[:, h] + yaw_rate * dt
        positions[:, h + 1, 0] = positions[:, h, 0] + v_next * np.cos(psi) * dt
        positions[:, h + 1, 1] = positions[:, h, 1] + v_next * np.sin(psi) * dt
        speeds[:, h + 1] = v_next
        headings[:, h + 1] = psi
    return positions, speeds, headings


def hold_still(agent: TrafficAgent) -> None:
    agent.vel[:] = 0.0
    agent.prev_accel[:] = 0.0
    agent.prev_control = {"accel": 0.0, "steering": 0.0}
