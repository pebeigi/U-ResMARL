"""Hard road containment for the discrete bicycle simulator.

The swept footprint encloses linear position/heading interpolation between
discrete poses. A straight braking rollout must also fit, retaining a feasible
backup for the next step. This is a model-based filter, not a real-car guarantee.
"""
from __future__ import annotations

import math
import numpy as np
from shapely.geometry import MultiPoint, Polygon
from shapely.prepared import prep

from RL.corridor import load_corridor, oriented_box_corners


class BoundaryInfeasibleError(RuntimeError):
    """No verified road-contained command exists; do not advance the state."""


_ROADS = {}


def road_region(corridor, margin=0.0):
    if not np.isfinite(margin) or margin < 0:
        raise ValueError("Boundary margin must be finite and nonnegative")
    key = (id(corridor), float(margin))
    if key not in _ROADS:
        polygon = Polygon(np.vstack((corridor.lower, corridor.upper[::-1])))
        if not polygon.is_valid or polygon.is_empty:
            raise ValueError("Road boundaries must form a valid polygon")
        region = polygon.buffer(-float(margin)) if margin else polygon
        if region.is_empty:
            raise ValueError("Boundary margin leaves no drivable road")
        _ROADS[key] = (corridor, region, prep(region))
    return _ROADS[key][1:]


def footprint_clearance(corridor, pos, heading, length=4.5, width=1.8):
    road, prepared = road_region(corridor)
    footprint = Polygon(oriented_box_corners(pos, heading, length, width))
    if prepared.covers(footprint):
        return float(road.boundary.distance(footprint))
    # Negative indicates any overhang, including edge crossings in concave roads.
    return -max(1e-9, float(footprint.difference(road).area) / max(length, width))


def _sweep_inside(pos, heading, candidate, corridor, sim):
    length = float(sim.get("vehicle_length", 4.5))
    width = float(sim.get("vehicle_width", 1.8))
    first = oriented_box_corners(pos, heading, length, width)
    last = oriented_box_corners(candidate["pos"], candidate["heading"], length, width)
    angle = abs(float(candidate["heading"]) - float(heading))
    # A full rotation needs a disk enclosure, not the short-arc sagitta bound.
    radius = 0.5 * math.hypot(length, width)
    padding = radius * (1.0 - math.cos(min(angle, 2 * math.pi) / 2))
    if angle >= 2 * math.pi:
        padding = 2 * radius
    sweep = MultiPoint(np.vstack((first, last))).convex_hull
    if padding > 1e-12:
        # Shapely's circular buffers are inscribed polygons: enlarge so their
        # apothem, too, covers the required rotation allowance.
        sweep = sweep.buffer(padding / math.cos(math.pi / 64), resolution=16)
    _, prepared = road_region(corridor, float(sim.get("boundary_margin", 0.1)))
    return bool(prepared.covers(sweep))


def braking_accel(sim):
    grid = sim.get("candidate_accel_grid", [-3., -2., -1., 0., 1., 2., 3.])
    braking = min(float(sim.get("max_accel", 4.0)), -min(map(float, grid)))
    if braking <= 0:
        raise ValueError("Boundary filtering requires a braking action in the grid")
    return braking


def candidate_boundary_safe(agent, candidate, sim, corridor=None):
    if not sim.get("boundary_safety_filter", False):
        return True
    corridor = corridor if corridor is not None else load_corridor(
        int(sim["run_id"]), int(sim["lane_kf"]))
    if not _sweep_inside(agent.pos, agent.heading, candidate, corridor, sim):
        return False
    speed = float(candidate["speed"])
    dt = float(sim["dt"])
    decel = braking_accel(sim)
    steps = max(0, int(math.ceil(speed / (decel * dt))) - 1)
    distance = dt * (steps * speed - decel * dt * steps * (steps + 1) / 2)
    heading = float(candidate["heading"])
    stopped = dict(candidate)
    stopped["pos"] = candidate["pos"] + distance * np.array([math.cos(heading), math.sin(heading)])
    return _sweep_inside(candidate["pos"], heading, stopped, corridor, sim)


def filter_boundary_control(agent, control, sim, corridor=None):
    """Keep the command if feasible, else choose the closest feasible grid command."""
    if not sim.get("boundary_safety_filter", False):
        return control
    from utility_model import kinematic_bicycle_rollout

    def safe(command):
        candidate = kinematic_bicycle_rollout(agent.pos, agent.heading, agent.speed,
                                              *command, float(sim["dt"]), sim)
        return candidate_boundary_safe(agent, candidate, sim, corridor)

    if safe(control):
        return control
    alternatives = [(float(a), float(s))
                    for a in sim.get("candidate_accel_grid", [-3., -2., -1., 0., 1., 2., 3.])
                    for s in sim.get("candidate_steering_grid", np.linspace(-.45, .45, 9))]
    alternatives.sort(key=lambda c: ((c[0] - control[0]) / sim.get("max_accel", 4.)) ** 2
                      + ((c[1] - control[1]) / .45) ** 2)
    for command in alternatives:
        if safe(command):
            return command
    raise BoundaryInfeasibleError(
        f"Agent {agent.agent_id}: no road-contained command with a safe braking backup; "
        "state was not advanced")
