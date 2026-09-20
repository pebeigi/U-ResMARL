"""Common decision-time information and feasibility queries.

The calibrated utility functions are unchanged. Callers supply the same local
observations as the actors; the global execution shield is a separate shared layer.
"""
from __future__ import annotations

import numpy as np

DECISION_PROTOCOL_VERSION = 1


def neighbor_indices(agents, index, sim):
    ego = agents[index]
    ranked = [(float(np.linalg.norm(a.pos-ego.pos)), j) for j, a in enumerate(agents)
              if j != index and not a.reached_destination]
    ranked.sort(key=lambda item: (item[0], item[1]))
    return [j for distance, j in ranked if distance <= float(sim.get('perception_radius', 60.))][
        :int(sim.get('max_neighbors', 6))]


def local_agents(agents, index, sim):
    return [agents[index]] + [agents[j] for j in neighbor_indices(agents, index, sim)]


def select_best_candidate(index, agent, agents, params, sim):
    from utility_model import select_best_candidate as select
    return select(0, agent, local_agents(agents, index, sim), params, sim)


def select_candidate_with_logit_residual(index, agent, agents, params, sim, logit_residual=None):
    from utility_model import select_candidate_with_logit_residual as select
    return select(0, agent, local_agents(agents, index, sim), params, sim, logit_residual)


def control_feasible(index, agents, accel, steering, sim, corridor=None):
    from utility_model import kinematic_bicycle_rollout, build_step_context, candidate_obb_conflict
    from RL.boundary import candidate_boundary_safe
    local = local_agents(agents, index, sim)
    ego = local[0]
    candidate = kinematic_bicycle_rollout(ego.pos, ego.heading, ego.speed, float(accel),
                                          float(steering), float(sim['dt']), sim)
    if not candidate_boundary_safe(ego, candidate, sim, corridor):
        return False
    return not sim.get('obb_safety_filter', True) or not candidate_obb_conflict(
        candidate, 0, local, sim, context=build_step_context(0, ego, local, sim))


def trajectory_obb_free(positions, headings, times, agents, index, sim):
    """CV OBB feasibility at native trajectory times and interval subdivisions.

    Predictor receives only the common local agent set. This query is available
    regardless of whether the optional global execution OBB shield is enabled.
    """
    from RL.corridor import boxes_overlap_any, oriented_box_corners_batch
    positions = np.asarray(positions)
    headings = np.asarray(headings)
    times = np.broadcast_to(times, positions.shape[:2])
    valid = np.ones(len(positions), dtype=bool)
    neighbors = [agents[j] for j in neighbor_indices(agents, index, sim)]
    if not neighbors:
        return valid
    p0 = np.array([a.pos for a in neighbors]); velocity = np.array([a.vel for a in neighbors])
    yaw = np.array([a.heading for a in neighbors])
    length, width = sim.get('vehicle_length', 4.5), sim.get('vehicle_width', 1.8)
    spacing = min(float(sim['dt']), float(sim.get('conflict_horizon', 1.5))/max(1, int(sim.get('conflict_substeps', 4))))
    prediction_cache = {}
    for k in range(len(positions)):
        for j in range(1, positions.shape[1]):
            duration = times[k, j]-times[k, j-1]
            if duration <= 1e-10:
                continue
            dh = (headings[k, j]-headings[k, j-1]+np.pi) % (2*np.pi)-np.pi
            for alpha in np.linspace(0., 1., max(1, int(np.ceil(duration/spacing)))+1)[1:]:
                t = times[k, j-1]+alpha*duration
                key = round(float(t), 10)
                if key not in prediction_cache:
                    mu = p0+velocity*t
                    prediction_cache[key] = (mu, oriented_box_corners_batch(mu, yaw, length, width))
                mu, corners = prediction_cache[key]
                xy = (1-alpha)*positions[k, j-1]+alpha*positions[k, j]
                if boxes_overlap_any(xy, headings[k, j-1]+alpha*dh, corners, mu, length, width):
                    valid[k] = False
                    break
            if not valid[k]:
                break
    return valid
