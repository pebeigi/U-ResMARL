"""One synchronous traffic step shared by every trainer and evaluator."""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from RL.corridor import boundary_reward, boxes_overlap
from RL.obs import contact_safety_reward
from utility_model import kinematic_bicycle_rollout, sanitize_control_command

DEFAULT_REWARD_WEIGHTS = {"progress": 1.0, "safety": 0.5, "smooth": 0.2}


def driving_reward(agents, idx, corridor, sim, dest_s, control, weights=None,
                   leftover_coef=0.05):
    """Pre-transition state reward; arrival and contact are added after movement."""
    ego = agents[idx]
    w = DEFAULT_REWARD_WEIGHTS if weights is None else weights
    s, _, tangent, _, _ = corridor.project(ego.pos)
    heading = float(np.arctan2(tangent[1], tangent[0]))
    progress = ego.speed * np.cos(ego.heading - heading)
    neighbors = sorted(
        (float(np.linalg.norm(other.pos - ego.pos)), j)
        for j, other in enumerate(agents)
        if j != idx and not other.reached_destination
        and np.linalg.norm(other.pos - ego.pos) <= sim["perception_radius"]
    )[:int(sim["max_neighbors"])]
    safety = sum(contact_safety_reward(d, sim.get("vehicle_length", 4.5))
                 for d, _ in neighbors)
    accel, steering = control
    smooth = -(accel ** 2 + sim.get("steering_penalty_weight", 0.5) * steering ** 2)
    lo, hi, _ = corridor.clearances(ego.pos)
    boundary, _ = boundary_reward(lo, hi)
    return float(w.get("progress", 1.0) * progress + w.get("safety", 0.5) * safety
                 + w.get("smooth", 0.2) * smooth + boundary
                 - leftover_coef * max(float(dest_s) - s, 0.0) / max(corridor.length, 1.0))


@dataclass
class StepResult:
    controls: list
    rewards: list
    candidates: list
    active_before: list
    arrived: set
    collision_pairs: set
    colliding_agents: set


def advance_agents(agents, controls, corridor, sim, dest_s, *, reward_weights=None,
                   leftover_coef=0.05, arrival_bonus=5.0, collision_penalty=0.0):
    """Filter all commands on the same state, then move and record new arrivals.

    Contacts on an agent's arrival step still count. Previously arrived agents
    are inactive and do not participate in the transition.
    """
    if len(controls) != len(agents) or len(dest_s) != len(agents):
        raise ValueError("Expected one command and destination per agent")
    dt = float(sim["dt"])
    active = [not a.reached_destination for a in agents]
    executed, moves, rewards = [], [], []
    for i, agent in enumerate(agents):
        if not active[i]:
            executed.append((0.0, 0.0)); moves.append(None); rewards.append(0.0)
            continue
        accel, steering = controls[i]
        control = (float(np.clip(accel, -sim.get("max_accel", 4.0), sim.get("max_accel", 4.0))),
                   float(np.clip(steering, -0.45, 0.45)))
        if not np.isfinite(control).all():
            raise ValueError("Non-finite control command")
        control = sanitize_control_command(i, agent, agents, control, sim)
        executed.append(control)
        moves.append(kinematic_bicycle_rollout(agent.pos, agent.heading, agent.speed,
                                               *control, dt, sim))
        moves[-1]["realized_accel"] = (moves[-1]["speed"] - agent.speed) / dt
        rewards.append(driving_reward(agents, i, corridor, sim, dest_s[i], control,
                                      reward_weights, leftover_coef))

    # Atomic check using the actual scenario geometry before any agent moves.
    from RL.boundary import candidate_boundary_safe, BoundaryInfeasibleError
    for i, move in enumerate(moves):
        if active[i] and not candidate_boundary_safe(agents[i], move, sim, corridor):
            raise BoundaryInfeasibleError(f"Agent {i}: boundary validation failed; state was not advanced")

    arrived = set()
    tol = float(sim.get("destination_threshold", 1.0))
    for i, agent in enumerate(agents):
        if active[i]:
            agent.update_state_from_candidate(moves[i], dt, tol)
            if agent.reached_destination or corridor.project(agent.pos)[0] >= dest_s[i] - tol:
                agent.reached_destination = True
                arrived.add(i)
                rewards[i] += arrival_bonus
        if agent.reached_destination:
            agent.vel[:] = 0.0
            agent.prev_accel[:] = 0.0
            agent.prev_control = {"accel": 0.0, "steering": 0.0}

    pairs = set()
    for i in range(len(agents)):
        for j in range(i + 1, len(agents)):
            if not (active[i] and active[j]):
                continue
            if sim.get("use_obb_collisions", True):
                hit = boxes_overlap(agents[i].pos, agents[i].heading, agents[j].pos,
                                    agents[j].heading, sim.get("vehicle_length", 4.5),
                                    sim.get("vehicle_width", 1.8))
            else:
                hit = np.linalg.norm(agents[i].pos - agents[j].pos) < sim.get("collision_threshold", 1.5)
            if hit:
                pairs.add((i, j))
    colliding = {i for pair in pairs for i in pair}
    for i in colliding:
        rewards[i] -= collision_penalty
    return StepResult(executed, rewards, moves, active, arrived, pairs, colliding)
