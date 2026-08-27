"""Lebanon highway corridor geometry for the RL environment.

Default corridor: run_id=2, lane_kf=1 (user-selected).
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np

import RL._paths  # noqa: F401
from RL._paths import REPO_ROOT

DEFAULT_RUN_ID = 2
DEFAULT_LANE_KF = 1
DEFAULT_BOUNDARY_CSV = (
    REPO_ROOT
    / "data"
    / "Lebanon_Highway"
    / "derived_highway_boundaries"
    / "highway_boundaries.csv"
)

# Passenger-car footprint used for OBB collisions / drawing.
DEFAULT_VEHICLE_LENGTH = 4.5
DEFAULT_VEHICLE_WIDTH = 1.8

# A flat -10 off-corridor penalty leaves an exploit: the progress earned by
# cutting across the corridor's curves outweighs it, so leaving the road pays.
# The base penalty therefore has to exceed the largest per-step progress reward,
# which is `max_agent_speed` = 16, and the depth term adds an inward gradient.
# The depth term is capped because an unbounded penalty dominates the return of
# an untrained policy, which spends most of its early episodes off the corridor,
# and learning stalls before it ever discovers progress.
OFFROAD_BASE_PENALTY = 20.0
OFFROAD_DEPTH_PENALTY = 3.0
OFFROAD_MAX_DEPTH = 5.0


def boundary_reward(clearance_low: float, clearance_high: float) -> tuple[float, bool]:
    """Off-corridor penalty and whether the vehicle is outside the corridor."""
    violation = -min(float(clearance_low), float(clearance_high))
    if violation <= 0.0:
        return 0.0, False
    depth = min(violation, OFFROAD_MAX_DEPTH)
    return -(OFFROAD_BASE_PENALTY + OFFROAD_DEPTH_PENALTY * depth), True


@dataclass(frozen=True)
class HighwayCorridor:
    run_id: int
    lane_kf: int
    center: np.ndarray  # (N, 2)
    lower: np.ndarray
    upper: np.ndarray
    cumulative_s: np.ndarray  # (N,)
    tangents: np.ndarray  # (N-1, 2) unit

    @property
    def length(self) -> float:
        return float(self.cumulative_s[-1])

    def project(self, point: np.ndarray) -> tuple[float, float, np.ndarray, int, float]:
        """Return (s, lateral, tangent, seg_i, t) at closest centerline point."""
        point = np.asarray(point, dtype=float)
        # Coarse nearest vertex, then refine on a local segment window.
        d2 = np.sum((self.center - point) ** 2, axis=1)
        i0 = int(np.argmin(d2))
        i_lo = max(0, i0 - 2)
        i_hi = min(len(self.center) - 2, i0 + 2)

        # Vectorized search over the window, then the winning segment is recomputed
        # with the original scalar expressions so results stay bit-identical.
        a_win = self.center[i_lo : i_hi + 1]
        ab_win = self.center[i_lo + 1 : i_hi + 2] - a_win
        pa_win = point - a_win
        t_win = np.clip(
            np.sum(pa_win * ab_win, axis=1) / (np.sum(ab_win * ab_win, axis=1) + 1e-12), 0.0, 1.0
        )
        rel_win = pa_win - t_win[:, None] * ab_win
        i = i_lo + int(np.argmin(np.sum(rel_win * rel_win, axis=1)))

        a = self.center[i]
        ab = self.center[i + 1] - a
        denom = float(ab @ ab) + 1e-12
        t = float(np.clip(((point - a) @ ab) / denom, 0.0, 1.0))
        q = a + t * ab
        tangent = self.tangents[i]
        normal = np.array([-tangent[1], tangent[0]], dtype=float)
        return (
            float(self.cumulative_s[i] + t * np.linalg.norm(ab)),
            float((point - q) @ normal),
            tangent,
            i,
            t,
        )

    def edge_points_at(self, seg_i: int, t: float) -> tuple[np.ndarray, np.ndarray]:
        t = float(np.clip(t, 0.0, 1.0))
        i = max(0, min(seg_i, len(self.lower) - 2))
        lower = (1.0 - t) * self.lower[i] + t * self.lower[i + 1]
        upper = (1.0 - t) * self.upper[i] + t * self.upper[i + 1]
        return lower, upper

    def clearances(self, point: np.ndarray) -> tuple[float, float, float]:
        """
        Return (clearance_to_lower, clearance_to_upper, signed_from_mid).

        Clearance > 0 means inside relative to that edge. signed_from_mid is
        the offset along the local lower→upper chord (positive toward upper).
        """
        _, _, _, seg_i, t = self.project(point)
        lower, upper = self.edge_points_at(seg_i, t)
        mid = 0.5 * (lower + upper)
        chord = upper - lower
        chord_len = float(np.linalg.norm(chord))
        if chord_len < 1e-6:
            return 0.0, 0.0, 0.0
        unit = chord / chord_len
        signed = float((np.asarray(point, dtype=float) - mid) @ unit)
        half = 0.5 * chord_len
        return half + signed, half - signed, signed

    def inside(self, point: np.ndarray, margin: float = 0.0) -> bool:
        c_lo, c_hi, _ = self.clearances(point)
        return c_lo >= margin and c_hi >= margin

    def xy_from_frenet(self, s: float, lateral: float) -> tuple[np.ndarray, np.ndarray]:
        """Map (s, lateral) → (xy, tangent)."""
        s = float(np.clip(s, 0.0, self.length))
        idx = int(np.searchsorted(self.cumulative_s, s, side="right") - 1)
        idx = max(0, min(idx, len(self.center) - 2))
        s0 = self.cumulative_s[idx]
        seg_len = max(float(self.cumulative_s[idx + 1] - s0), 1e-9)
        t = (s - s0) / seg_len
        a = self.center[idx]
        b = self.center[idx + 1]
        q = a + t * (b - a)
        tangent = self.tangents[idx]
        normal = np.array([-tangent[1], tangent[0]], dtype=float)
        return q + lateral * normal, tangent

    def path_error(self, point: np.ndarray, boundary_buffer: float = 1.5) -> float:
        """Same convention as calibration boundary_path_error (scalar)."""
        c_lo, c_hi, _ = self.clearances(point)
        clearance = min(c_lo, c_hi)
        if clearance >= 0.0:
            return float(max(0.0, boundary_buffer - clearance))
        return float(-clearance + boundary_buffer)


@lru_cache(maxsize=8)
def load_corridor(
    run_id: int = DEFAULT_RUN_ID,
    lane_kf: int = DEFAULT_LANE_KF,
    csv_path: str | None = None,
) -> HighwayCorridor:
    import pandas as pd

    path = Path(csv_path) if csv_path else DEFAULT_BOUNDARY_CSV
    df = pd.read_csv(path)
    mask = (df["run_id"] == int(run_id)) & (df["lane_kf"] == int(lane_kf))
    g = df.loc[mask].sort_values("point_index")
    if len(g) < 2:
        raise ValueError(f"No boundary polyline for run_id={run_id}, lane_kf={lane_kf} in {path}")

    center = g[["center_x", "center_y"]].to_numpy(float)
    lower = g[["lower_x", "lower_y"]].to_numpy(float)
    upper = g[["upper_x", "upper_y"]].to_numpy(float)
    seg = center[1:] - center[:-1]
    seg_lens = np.linalg.norm(seg, axis=1)
    cumulative_s = np.concatenate([[0.0], np.cumsum(seg_lens)])
    tangents = seg / np.maximum(seg_lens[:, None], 1e-12)
    return HighwayCorridor(
        run_id=int(run_id),
        lane_kf=int(lane_kf),
        center=center,
        lower=lower,
        upper=upper,
        cumulative_s=cumulative_s,
        tangents=tangents,
    )


def oriented_box_corners(
    pos: np.ndarray,
    heading: float,
    length: float = DEFAULT_VEHICLE_LENGTH,
    width: float = DEFAULT_VEHICLE_WIDTH,
) -> np.ndarray:
    """Return 4 corner points of an oriented rectangle (rear-left … clockwise)."""
    c, s = float(np.cos(heading)), float(np.sin(heading))
    # Local frame: +x forward, +y left
    hx, hy = 0.5 * length, 0.5 * width
    local = np.array(
        [
            [hx, hy],
            [hx, -hy],
            [-hx, -hy],
            [-hx, hy],
        ],
        dtype=float,
    )
    rot = np.array([[c, -s], [s, c]], dtype=float)
    return np.asarray(pos, dtype=float) + local @ rot.T


def oriented_box_corners_batch(
    positions: np.ndarray,
    headings: np.ndarray,
    length: float = DEFAULT_VEHICLE_LENGTH,
    width: float = DEFAULT_VEHICLE_WIDTH,
) -> np.ndarray:
    """Corners for many boxes at once: (m, 4, 2), same ordering as the scalar version."""
    positions = np.atleast_2d(np.asarray(positions, dtype=float))
    headings = np.atleast_1d(np.asarray(headings, dtype=float))
    hx, hy = 0.5 * length, 0.5 * width
    local = np.array([[hx, hy], [hx, -hy], [-hx, -hy], [-hx, hy]], dtype=float)
    c = np.cos(headings)
    s = np.sin(headings)
    # rot.T per box, applied as local @ rot.T
    rot_t = np.stack([np.stack([c, s], axis=-1), np.stack([-s, c], axis=-1)], axis=1)
    return positions[:, None, :] + np.einsum("kj,mjl->mkl", local, rot_t)


def boxes_overlap_any(
    pos_a: np.ndarray,
    heading_a: float,
    corners_b: np.ndarray,
    positions_b: np.ndarray,
    length: float = DEFAULT_VEHICLE_LENGTH,
    width: float = DEFAULT_VEHICLE_WIDTH,
) -> bool:
    """True if box A overlaps any of the pre-built boxes in ``corners_b``.

    Same separating-axis test as :func:`boxes_overlap`, evaluated for all
    candidates at once. Boxes farther apart than the circumscribed diameter are
    rejected first, which cannot change the result.
    """
    corners_b = np.asarray(corners_b, dtype=float)
    if corners_b.size == 0:
        return False
    pos_a = np.asarray(pos_a, dtype=float)
    positions_b = np.atleast_2d(np.asarray(positions_b, dtype=float))

    diameter = float(np.hypot(length, width))
    offset = positions_b - pos_a
    near = np.einsum("ij,ij->i", offset, offset) <= diameter * diameter
    if not near.any():
        return False
    cb = corners_b[near]

    ca = oriented_box_corners(pos_a, heading_a, length, width)
    separated = np.zeros(len(cb), dtype=bool)
    for axis in (ca[0] - ca[1], ca[1] - ca[2]):
        unit = axis / (float(np.linalg.norm(axis)) + 1e-12)
        proj_a = ca @ unit
        proj_b = cb @ unit
        separated |= (proj_a.max() < proj_b.min(axis=1)) | (proj_b.max(axis=1) < proj_a.min())
    for axis in (cb[:, 0] - cb[:, 1], cb[:, 1] - cb[:, 2]):
        unit = axis / (np.linalg.norm(axis, axis=1)[:, None] + 1e-12)
        proj_a = ca @ unit.T  # (4, m)
        proj_b = np.einsum("mij,mj->mi", cb, unit)
        separated |= (proj_a.max(axis=0) < proj_b.min(axis=1)) | (proj_b.max(axis=1) < proj_a.min(axis=0))
    return bool((~separated).any())


def _axis_separate(corners_a: np.ndarray, corners_b: np.ndarray, axis: np.ndarray) -> bool:
    axis = axis / (np.linalg.norm(axis) + 1e-12)
    proj_a = corners_a @ axis
    proj_b = corners_b @ axis
    return float(proj_a.max()) < float(proj_b.min()) or float(proj_b.max()) < float(proj_a.min())


def boxes_overlap(
    pos_a: np.ndarray,
    heading_a: float,
    pos_b: np.ndarray,
    heading_b: float,
    length: float = DEFAULT_VEHICLE_LENGTH,
    width: float = DEFAULT_VEHICLE_WIDTH,
) -> bool:
    """2D OBB overlap via separating-axis theorem."""
    ca = oriented_box_corners(pos_a, heading_a, length, width)
    cb = oriented_box_corners(pos_b, heading_b, length, width)
    axes = [
        ca[0] - ca[1],
        ca[1] - ca[2],
        cb[0] - cb[1],
        cb[1] - cb[2],
    ]
    for axis in axes:
        if _axis_separate(ca, cb, axis):
            return False
    return True


def corridor_sim_defaults(corridor: HighwayCorridor) -> dict[str, Any]:
    """sim_config fields for utility_model + RL env using the selected corridor."""
    xs = np.concatenate([corridor.lower[:, 0], corridor.upper[:, 0], corridor.center[:, 0]])
    ys = np.concatenate([corridor.lower[:, 1], corridor.upper[:, 1], corridor.center[:, 1]])
    return {
        "run_id": corridor.run_id,
        "lane_kf": corridor.lane_kf,
        "path_mode": "polyline",
        "utility_frame": "corridor",
        "boundary_buffer": 1.5,
        "road_x_min": float(xs.min()),
        "road_x_max": float(xs.max()),
        "road_y_min": float(ys.min()),
        "road_y_max": float(ys.max()),
        "vehicle_length": DEFAULT_VEHICLE_LENGTH,
        "vehicle_width": DEFAULT_VEHICLE_WIDTH,
        "corridor_length": corridor.length,
    }
