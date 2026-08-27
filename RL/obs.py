"""Local observations for residual MARL and the shared benchmark.

World (x, y) on the Lebanon corridor is a long diagonal, so a fixed y/12 scale
saturates the actor. Features are therefore Frenet for the ego vehicle and
body-frame for neighbors.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Sequence

import numpy as np

if TYPE_CHECKING:
    from RL.corridor import HighwayCorridor
    from utility_model import TrafficAgent


def wrap_angle(angle: float) -> float:
    return float((angle + np.pi) % (2.0 * np.pi) - np.pi)


def body_frame_delta(delta_world: np.ndarray, heading: float) -> np.ndarray:
    c = float(np.cos(heading))
    s = float(np.sin(heading))
    dx, dy = float(delta_world[0]), float(delta_world[1])
    return np.array([c * dx + s * dy, -s * dx + c * dy], dtype=float)


def local_observation(
    ego: "TrafficAgent",
    agents: Sequence["TrafficAgent"],
    neighbor_ids: Sequence[int],
    corridor: "HighwayCorridor",
    max_neighbors: int,
) -> np.ndarray:
    """[s, n, v, heading_err, goal_err, c_lo, c_hi] + k * [dx_fwd, dy_left, dv_fwd, dv_left]."""
    obs = np.zeros(7 + 4 * int(max_neighbors), dtype=np.float32)
    s, n, tangent, _, _ = corridor.project(ego.pos)
    tangent_angle = float(np.arctan2(tangent[1], tangent[0]))
    c_lo, c_hi, _ = corridor.clearances(ego.pos)
    heading = float(ego.heading)
    obs[0] = float(s)
    obs[1] = float(n)
    obs[2] = float(ego.speed)
    obs[3] = wrap_angle(heading - tangent_angle)
    obs[4] = wrap_angle(float(ego.goal_heading) - tangent_angle)
    obs[5] = float(c_lo)
    obs[6] = float(c_hi)

    start = 7
    for j in neighbor_ids:
        other = agents[j]
        d_body = body_frame_delta(np.asarray(other.pos, dtype=float) - np.asarray(ego.pos, dtype=float), heading)
        v_body = body_frame_delta(np.asarray(other.vel, dtype=float) - np.asarray(ego.vel, dtype=float), heading)
        obs[start : start + 4] = [d_body[0], d_body[1], v_body[0], v_body[1]]
        start += 4
    return obs


def contact_safety_reward(distance: float, vehicle_length: float = 4.5) -> float:
    """Reward term for one neighbor: strong near/inside the footprint, soft at range."""
    gap = float(distance) - float(vehicle_length)
    if gap <= 0.0:
        return -16.0 * (1.0 - gap)
    return -float(np.exp(-gap / 2.0))
