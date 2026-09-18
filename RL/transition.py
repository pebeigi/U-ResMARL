"""One synchronous traffic step shared by every trainer and evaluator."""

from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
import numpy as np

from RL.closed_loop_score import soft_comfort_factor, soft_safety_factor
from RL.corridor import boundary_reward, boxes_overlap
from RL.obs import contact_safety_reward, footprint_surface_gap
from utility_model import kinematic_bicycle_rollout, sanitize_control_command

# Revision 6 follows CaRL (Jaeger et al.): route progress is the only dense
# positive term; TTC/comfort multiply that progress; hard contact is a terminal
# reward (no further return). Additive competing costs are the diagnosed failure
# mode: discounted PPO could prefer collision-worse trajectories.
DEFAULT_REWARD_WEIGHTS = {"progress": 1.0, "time": 0.0, "safety": 1.0, "smooth": 1.0}
DRIVING_REWARD_REVISION = 6


def driving_reward(agents, idx, corridor, sim, dest_s, control, weights=None,
                   leftover_coef=0.08, *, previous_station=None):
    """CaRL-style progress reward with multiplicative soft constraints.

    Forward station progress is at most one per active step and is multiplied by
    proximity and comfort factors in (0, 1]. Waiting therefore cannot accumulate
    a competing additive bonus. Leftover remaining-distance is a small potential;
    time is off by default. Hard contacts are handled in ``advance_agents``.
    """
    ego = agents[idx]
    w = DEFAULT_REWARD_WEIGHTS if weights is None else weights
    s, _, tangent, _, _ = corridor.project(ego.pos)
    heading = float(np.arctan2(tangent[1], tangent[0]))
    speed_cap = max(float(sim.get("max_agent_speed", 16.0)), 1e-6)
    if previous_station is None:
        progress = ego.speed * np.cos(ego.heading - heading) / speed_cap
    else:
        progress = (min(s, dest_s) - min(previous_station, dest_s)) / (speed_cap * sim["dt"])
    progress = float(np.clip(progress, -1.0, 1.0))
    neighbors = sorted(
        (float(np.linalg.norm(other.pos - ego.pos)), j)
        for j, other in enumerate(agents)
        if j != idx and not other.reached_destination
        and np.linalg.norm(other.pos - ego.pos) <= sim["perception_radius"]
    )[:int(sim["max_neighbors"])]
    safety = min((contact_safety_reward(footprint_surface_gap(
        ego, agents[j], sim.get("vehicle_length", 4.5), sim.get("vehicle_width", 1.8)))
        for _, j in neighbors), default=0.0)
    accel, steering = control
    route = max(progress, 0.0) * soft_safety_factor(safety, w.get("safety", 1.0)) * soft_comfort_factor(
        accel, steering, w.get("smooth", 1.0))
    lo, hi, _ = corridor.clearances(ego.pos)
    boundary, _ = boundary_reward(lo, hi)
    return float(w.get("progress", 1.0) * route - w.get("time", 0.0)
                 + boundary
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
                   leftover_coef=0.08, arrival_bonus=8.0, collision_penalty=0.0,
                   collision_event_penalty=0.0, previous_collision_pairs=None):
    """Filter all commands on the same state, then move and record new arrivals.

    Contacts on an agent's arrival step still count. Previously arrived agents
    are inactive and do not participate in the transition. Hard contact is a
    terminal CaRL penalty: ``collision_event_penalty`` (and optional duration
    ``collision_penalty``) apply on onset; already-colliding agents receive no
    further driving reward. Physics still advance so evaluation metrics see the
    full rollout.
    """
    if len(controls) != len(agents) or len(dest_s) != len(agents):
        raise ValueError("Expected one command and destination per agent")
    dt = float(sim["dt"])
    active = [not a.reached_destination for a in agents]
    stations_before = [float(corridor.project(a.pos)[0]) for a in agents]
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
        rewards.append(0.0)

    # Atomic check using the actual scenario geometry before any agent moves.
    from RL.boundary import candidate_boundary_safe, BoundaryInfeasibleError
    for i, move in enumerate(moves):
        if active[i] and not candidate_boundary_safe(agents[i], move, sim, corridor):
            raise BoundaryInfeasibleError(f"Agent {i}: boundary validation failed; state was not advanced")

    # All rewards see the same joint next state, including cars arriving now.
    # No live agent is changed until every proposed move has been validated.
    reward_agents = [SimpleNamespace(
        pos=move["pos"] if move is not None else a.pos,
        heading=move["heading"] if move is not None else a.heading,
        speed=move["speed"] if move is not None else a.speed,
        reached_destination=not active[i],
    ) for i, (a, move) in enumerate(zip(agents, moves))]
    for i in range(len(agents)):
        if active[i]:
            rewards[i] = driving_reward(
                reward_agents, i, corridor, sim, dest_s[i], executed[i],
                reward_weights, leftover_coef, previous_station=stations_before[i])

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
    previous = set(previous_collision_pairs or ())
    previous_agents = {i for pair in previous for i in pair}
    for i in colliding:
        if i in previous_agents:
            # CaRL hard constraint: no further route-completion reward after contact.
            rewards[i] = 0.0
        else:
            rewards[i] -= collision_penalty
    for i, j in pairs - previous:
        rewards[i] -= collision_event_penalty
        rewards[j] -= collision_event_penalty
    return StepResult(executed, rewards, moves, active, arrived, pairs, colliding)
