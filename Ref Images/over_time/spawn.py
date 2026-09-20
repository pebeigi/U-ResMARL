"""Qualitative packs for the over-time strips (not used in training)."""
from __future__ import annotations

from typing import Optional, Tuple

import numpy as np


def nearby_pool(corridor, center, radius, inner: float = 0.0):
    pool = np.asarray(corridor._interior_pool, float)
    if pool.size == 0:
        return pool.reshape(0, 2)
    d = np.linalg.norm(pool - np.asarray(center, float).reshape(1, 2), axis=1)
    mask = d <= float(radius)
    if inner > 0:
        mask &= d >= float(inner)
    return pool[mask]


def tgsim_junction(corridor):
    """NW Foggy Bottom intersection (west stub + north-south + east street)."""
    from shapely.geometry import Polygon

    hole = Polygon(corridor.roadway.interiors[0]).simplify(8.0)
    corners = np.asarray(hole.exterior.coords, float)[:-1]
    corner = corners[int(np.argmax(-corners[:, 0] + corners[:, 1]))]
    near = nearby_pool(corridor, corner, 22.0)
    if len(near) < 8:
        near = nearby_pool(corridor, corner, 32.0)
    return near.mean(axis=0) if len(near) else corner


def roundabout_hub(corridor):
    from shapely.geometry import Polygon

    island = max(corridor.roadway.interiors, key=lambda ring: Polygon(ring).area)
    return np.asarray(Polygon(island).centroid.coords[0], float)


def spawn_around(
    env,
    center,
    *,
    n_agents: int,
    radius: float,
    spacing: float,
    dest_lo: float,
    dest_hi: float,
    min_cars: int,
    speeds: Tuple[float, ...],
    cruise: float,
    desired: Tuple[float, float, float],
    inner: float = 0.0,
    min_speed: Optional[float] = None,
    ring_center=None,
    min_clearance: float = 0.0,
):
    from RL.boundary import footprint_clearance
    from RL.corridor import boxes_overlap
    from RL.spawn_safety import has_straight_braking_backup
    from utility_model import TrafficAgent, generate_candidate_actions

    corridor = env.corridor
    rng = env.rng
    sim = env.config.sim_config
    length = float(sim["vehicle_length"])
    width = float(sim["vehicle_width"])
    margin = float(sim.get("boundary_margin", 0.1))
    clearance = max(0.5 * width + 0.15, 0.35)
    on_road = corridor.on_road
    center = np.asarray(center, float).reshape(2)
    near = nearby_pool(corridor, center, radius, inner=inner)
    if len(near) < 12:
        near = nearby_pool(corridor, center, radius + 3.0, inner=inner)
    if len(near) < 12 and inner <= 0:
        near = nearby_pool(corridor, center, radius)
    if len(near) < 12:
        raise RuntimeError("Not enough on-road samples around the plot focus")
    floor = cruise if min_speed is None else float(min_speed)

    dest_pool = nearby_pool(corridor, center, dest_hi + radius, inner=max(0.0, dest_lo - radius))
    if len(dest_pool) < 24:
        dest_pool = np.asarray(corridor._interior_pool, float)

    def sample_dest(pos):
        """Own-goal arrival is Euclidean, so dest must be far in metres, not just geodesic."""
        order = rng.permutation(len(dest_pool))[:80]
        for idx in order:
            cand = dest_pool[int(idx)]
            eucl = float(np.linalg.norm(cand - pos))
            if eucl < dest_lo or eucl > dest_hi:
                continue
            rem, dest_dir = on_road.remaining_and_dir(pos[None], cand)
            remaining = float(rem[0])
            if remaining >= 1.0e8 or remaining < dest_lo:
                continue
            tangent = np.asarray(dest_dir[0], float)
            if float(np.linalg.norm(tangent)) < 0.2:
                continue
            return np.asarray(cand, float), tangent
        return None, None

    def accept(pos, heading, dest, i, agents):
        if footprint_clearance(corridor, pos, heading, length, width) < margin:
            return None
        if min_clearance > 0.0:
            gap = float(corridor.clearances(pos)[0])
            if gap < min_clearance:
                return None
        if any(np.linalg.norm(pos - a.pos) < spacing for a in agents):
            return None
        speed = float(np.clip(rng.normal(cruise, 0.2 * cruise), 0.85 * cruise, 1.2 * cruise))
        dt = float(sim.get("dt", 0.5))
        nxt = pos + speed * dt * np.array([np.cos(heading), np.sin(heading)])
        if footprint_clearance(corridor, nxt, heading, length, width) < margin:
            return None
        vel = np.array([speed * np.cos(heading), speed * np.sin(heading)], float)
        agent = TrafficAgent(
            agent_id=i,
            pos=np.asarray(pos, float),
            vel=vel,
            dest=np.asarray(dest, float),
            desired_speed=float(np.clip(rng.normal(desired[0], desired[1]), desired[2], desired[0] + 4.0)),
            nominal_y=float(pos[1]),
            run_id=0,
            lane_kf=0,
            heading_angle=float(heading),
        )
        if any(boxes_overlap(agent.pos, agent.heading, o.pos, o.heading, length, width) for o in agents):
            return None
        moving = 0
        for cand in generate_candidate_actions(agent, dt, sim, dedupe=False):
            if float(cand.get("speed", 0.0)) < 0.5:
                continue
            if footprint_clearance(corridor, cand["pos"], cand["heading"], length, width) >= margin:
                moving += 1
        if moving < 4:
            return None
        return agent

    last_err = "Could not pack moving cars around the plot focus"
    for _layout in range(18):
        agents, dest_s = [], []
        for idx in rng.permutation(len(near)):
            if len(agents) >= n_agents:
                break
            pos = near[int(idx)]
            dest, tangent = sample_dest(pos)
            if dest is None:
                continue
            tangent = np.asarray(tangent, float)
            if ring_center is not None:
                radial = pos - np.asarray(ring_center, float).reshape(2)
                ru = radial / max(float(np.linalg.norm(radial)), 1e-6)
                ccw = np.array([-ru[1], ru[0]], float)
                dest_ang = np.arctan2(dest[1] - ring_center[1], dest[0] - ring_center[0])
                pos_ang = np.arctan2(pos[1] - ring_center[1], pos[0] - ring_center[0])
                ahead = (dest_ang - pos_ang) % (2.0 * np.pi)
                if ahead < 0.8 or ahead > 2.7:
                    continue
                tangent = ccw
            if float(np.dot(tangent, dest - pos)) <= 0.0:
                continue
            heading = float(np.arctan2(tangent[1], tangent[0]) + rng.normal(0.0, 0.03))
            agent = accept(pos, heading, dest, len(agents), agents)
            if agent is None:
                continue
            agents.append(agent)
            dest_s.append(0.0)
        if len(agents) < min_cars:
            last_err = f"Local pack too sparse ({len(agents)} cars)"
            continue

        def set_speeds(group, speed):
            for agent in group:
                agent.vel = float(speed) * np.array([np.cos(agent.heading), np.sin(agent.heading)])

        chosen = None
        parked = list(agents)
        parked_dest = list(dest_s)
        for speed in speeds:
            if speed < floor - 1e-9:
                continue
            trial, trial_dest = list(parked), list(parked_dest)
            while len(trial) >= min_cars:
                set_speeds(trial, speed)
                if has_straight_braking_backup(trial, sim, corridor):
                    chosen = (trial, trial_dest, speed)
                    break
                trial.pop()
                trial_dest.pop()
            if chosen is not None:
                break
        if chosen is None:
            last_err = "Could not keep a moving speed with a collision-free braking backup"
            continue
        agents, dest_s, speed = chosen
        set_speeds(agents, speed)
        for i, agent in enumerate(agents):
            agent.agent_id = i
        print(
            f"Packed {len(agents)} cars at {speed:.1f} m/s around ({center[0]:.1f}, {center[1]:.1f})",
            flush=True,
        )
        return agents, dest_s
    raise RuntimeError(last_err)


def finish_site_scenario(env, seed, agents, dest_s, origin, half):
    from Baselines.scenario import _finalize_sim_config, scenario_from_env

    env.agents = agents
    env._dest_s = dest_s
    env.config.num_agents = len(agents)
    scenario = scenario_from_env(env, seed)
    scenario.sim_config = _finalize_sim_config(scenario.sim_config, None, "full", scenario.dt)
    scenario.plot_origin = np.asarray(origin, float)
    scenario.plot_half = float(half)
    return scenario
