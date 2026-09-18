"""Check a simultaneous straight-braking backup before accepting a spawn.

Positions evolve under the simulator's discrete braking integrator, with linear
interpolation between poses. Fixed headings allow exact swept OBB tests using
the interval of overlap on each separating axis. This is an initial-condition
check, not a replacement for learning or a runtime car-collision shield.
"""
from __future__ import annotations
import numpy as np


def swept_fixed_heading_pairs(start, end, headings, length, width):
    n = len(start)
    ia, ib = np.triu_indices(n, 1)
    if not len(ia):
        return set()
    longitudinal = np.stack((np.cos(headings), np.sin(headings)), axis=-1)
    lateral = np.stack((-np.sin(headings), np.cos(headings)), axis=-1)
    axes = np.stack((longitudinal[ia], lateral[ia], longitudinal[ib], lateral[ib]), axis=1)
    radii = .5 * length * (np.abs(np.einsum("pki,pi->pk", axes, longitudinal[ia]))
                          + np.abs(np.einsum("pki,pi->pk", axes, longitudinal[ib])))
    radii += .5 * width * (np.abs(np.einsum("pki,pi->pk", axes, lateral[ia]))
                          + np.abs(np.einsum("pki,pi->pk", axes, lateral[ib])))
    separation = np.einsum("pki,pi->pk", axes, start[ib] - start[ia])
    change = np.einsum("pki,pi->pk", axes, (end[ib] - start[ib]) - (end[ia] - start[ia]))
    moving = np.abs(change) > 1e-12
    denominator = np.where(moving, change, 1.)
    first, last = (-radii-separation)/denominator, (radii-separation)/denominator
    entry = np.where(moving, np.minimum(first, last), -np.inf)
    leave = np.where(moving, np.maximum(first, last), np.inf)
    separated = np.any(~moving & (np.abs(separation) > radii), axis=1)
    hit = (~separated & (np.maximum(entry.max(axis=1), 0.)
                         <= np.minimum(leave.min(axis=1), 1.)))
    return {(int(i), int(j)) for i, j in zip(ia[hit], ib[hit])}


def has_straight_braking_backup(agents, sim, corridor=None):
    from RL.boundary import _sweep_inside
    positions = np.asarray([a.pos for a in agents], dtype=float)
    headings = np.asarray([a.heading for a in agents], dtype=float)
    speeds = np.asarray([a.speed for a in agents], dtype=float)
    directions = np.stack((np.cos(headings), np.sin(headings)), axis=-1)
    dt, deceleration = float(sim["dt"]), float(sim.get("max_accel", 4.))
    if dt <= 0 or deceleration <= 0:
        raise ValueError("Spawn braking check requires positive dt and deceleration")
    length, width = sim.get("vehicle_length", 4.5), sim.get("vehicle_width", 1.8)
    if not len(agents):
        return True
    for _ in range(max(1, int(np.ceil(speeds.max() / (deceleration * dt))))):
        speeds = np.clip(speeds - deceleration * dt, 0., float(sim["max_agent_speed"]))
        next_positions = positions + speeds[:, None] * directions * dt
        if swept_fixed_heading_pairs(positions, next_positions, headings, length, width):
            return False
        if corridor is not None and sim.get("boundary_safety_filter", False):
            for i in range(len(agents)):
                candidate = {"pos": next_positions[i], "heading": headings[i]}
                if not _sweep_inside(positions[i], headings[i], candidate, corridor, sim):
                    return False
        positions = next_positions
    return True
