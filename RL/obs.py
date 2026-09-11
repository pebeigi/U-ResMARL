"""Local observations for residual training and the benchmark."""

from __future__ import annotations

from typing import TYPE_CHECKING, Sequence

import numpy as np

if TYPE_CHECKING:
    from RL.corridor import HighwayCorridor
    from utility_model import TrafficAgent

# Ego features: s, n, v, heading_err, goal_err, c_lo, c_hi, remaining_s
EGO_OBS_DIM = 8
LEGACY_EGO_OBS_DIM = 7


def wrap_angle(angle: float) -> float:
    return float((angle + np.pi) % (2.0 * np.pi) - np.pi)


def body_frame_delta(delta_world: np.ndarray, heading: float) -> np.ndarray:
    c = float(np.cos(heading))
    s = float(np.sin(heading))
    dx, dy = float(delta_world[0]), float(delta_world[1])
    return np.array([c * dx + s * dy, -s * dx + c * dy], dtype=float)


def observation_dim(max_neighbors: int, *, ego_dim: int = EGO_OBS_DIM) -> int:
    return int(ego_dim) + 4 * int(max_neighbors)


def ego_feature_count(obs_dim: int) -> int:
    """Infer whether ``obs_dim`` uses the 7- or 8-feature ego layout."""
    if (int(obs_dim) - EGO_OBS_DIM) % 4 == 0 and int(obs_dim) >= EGO_OBS_DIM:
        return EGO_OBS_DIM
    if (int(obs_dim) - LEGACY_EGO_OBS_DIM) % 4 == 0 and int(obs_dim) >= LEGACY_EGO_OBS_DIM:
        return LEGACY_EGO_OBS_DIM
    raise ValueError(f"Cannot infer ego feature count for obs_dim={obs_dim}")


def adapt_observation(obs: np.ndarray, target_dim: int) -> np.ndarray:
    """Map between the legacy 7-ego and current 8-ego observation layouts.

    Checkpoints trained before remaining-station was added expect ``7 + 4K``
    features.  Drop / insert the remaining-station slot at index 7 so old
    direct-discrete / MAPPO / pure-RL policies keep working under the new env.
    """
    arr = np.asarray(obs, dtype=np.float32)
    target = int(target_dim)
    if arr.shape[-1] == target:
        return arr

    src_ego = ego_feature_count(arr.shape[-1])
    dst_ego = ego_feature_count(target)
    if src_ego == EGO_OBS_DIM and dst_ego == LEGACY_EGO_OBS_DIM and arr.shape[-1] == target + 1:
        return np.concatenate([arr[..., :7], arr[..., 8:]], axis=-1)
    if src_ego == LEGACY_EGO_OBS_DIM and dst_ego == EGO_OBS_DIM and arr.shape[-1] + 1 == target:
        zeros = np.zeros(arr.shape[:-1] + (1,), dtype=np.float32)
        return np.concatenate([arr[..., :7], zeros, arr[..., 7:]], axis=-1)
    raise ValueError(
        f"Cannot adapt observation of dim {arr.shape[-1]} to target dim {target}"
    )


def local_observation(
    ego: "TrafficAgent",
    agents: Sequence["TrafficAgent"],
    neighbor_ids: Sequence[int],
    corridor: "HighwayCorridor",
    max_neighbors: int,
    dest_s: float | None = None,
) -> np.ndarray:
    """[s, n, v, heading_err, goal_err, c_lo, c_hi, remaining_s] + k neighbors."""
    obs = np.zeros(observation_dim(max_neighbors), dtype=np.float32)
    s, n, tangent, _, _ = corridor.project(ego.pos)
    tangent_angle = float(np.arctan2(tangent[1], tangent[0]))
    c_lo, c_hi, _ = corridor.clearances(ego.pos)
    heading = float(ego.heading)
    if dest_s is None:
        dest_s = float(corridor.project(ego.dest)[0])
    remaining = max(float(dest_s) - float(s), 0.0)

    obs[0] = float(s)
    obs[1] = float(n)
    obs[2] = float(ego.speed)
    obs[3] = wrap_angle(heading - tangent_angle)
    obs[4] = wrap_angle(float(ego.goal_heading) - tangent_angle)
    obs[5] = float(c_lo)
    obs[6] = float(c_hi)
    obs[7] = float(remaining)

    start = EGO_OBS_DIM
    for j in neighbor_ids:
        other = agents[j]
        d_body = body_frame_delta(
            np.asarray(other.pos, dtype=float) - np.asarray(ego.pos, dtype=float), heading
        )
        v_body = body_frame_delta(
            np.asarray(other.vel, dtype=float) - np.asarray(ego.vel, dtype=float), heading
        )
        obs[start : start + 4] = [d_body[0], d_body[1], v_body[0], v_body[1]]
        start += 4
    return obs


def contact_safety_reward(distance: float, vehicle_length: float = 4.5) -> float:
    """Reward term for one neighbor: strong near/inside the footprint, soft at range."""
    gap = float(distance) - float(vehicle_length)
    if gap <= 0.0:
        return -16.0 * (1.0 - gap)
    return -float(np.exp(-gap / 2.0))
