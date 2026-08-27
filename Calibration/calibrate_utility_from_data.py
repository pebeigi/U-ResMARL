#!/usr/bin/env python
"""Calibrate utility-parameter ranges from observed trajectories.

The main target is short-horizon closed-loop tracking: start from an observed
state, repeatedly let the utility model choose controls, and fit parameters that
keep the simulated rollout close to the observed trajectory. A one-step choice
likelihood is kept as a diagnostic and optional regularizer.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from heapq import heappop, heappush
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import shapely
from shapely.geometry import Polygon
from shapely.ops import unary_union

import Calibration._paths  # noqa: F401 — repo root on sys.path
from Calibration._paths import REPO_ROOT
from RL.traffic_env import EnvConfig
from utility_model import (
    UTILITY_PARAM_KEYS,
    CorridorGeometry,
    TrafficAgent,
    collision_variances,
    generate_candidate_actions,
)


PARAM_BOUNDS: dict[str, tuple[float, float]] = {
    "S_theta": (0.05, 10.0),
    "S_v": (0.05, 10.0),
    "xi_i": (1.1, 10.0),
    "S_d": (0.05, 10.0),
    "gamma": (0.1, 10.0),
    "w_x": (0.1, 10.0),
    "w_y": (0.1, 10.0),
    "w_c": (0.01, 1000.0),
    "w_ell": (0.1, 1000.0),
    "beta": (0.01, 10.0),
    "sigma_long": (0.5, 5.0),
    "sigma_lat": (0.3, 2.5),
    "beta": (0.01, 10.0),
    "sigma_long": (0.5, 5.0),
    "sigma_lat": (0.3, 2.5),
}


@dataclass
class ChoiceSample:
    run_id: int
    lane_kf: int
    vehicle_id: int
    time: float
    dir_cos: np.ndarray
    cand_speed: np.ndarray
    dist_abs: np.ndarray
    collision_prob: np.ndarray
    path_error: np.ndarray
    current_speed: float
    desired_speed: float
    target_index: int
    target_match_cost: float
    path_mode: str = "boundary"


@dataclass
class RolloutWindow:
    run_id: int
    lane_kf: int
    vehicle_id: int
    rows: list[dict[str, Any]]


# typing.Dict/Tuple: this is a runtime alias (not postponed by __future__ annotations).
# Required for Python 3.8 compatibility.
BoundaryMap = Dict[Tuple[int, int], pd.DataFrame]


def log_progress(message: str, verbose: bool) -> None:
    if verbose:
        print(message, flush=True)


def angle_diff(a: np.ndarray, b: float) -> np.ndarray:
    return np.arctan2(np.sin(a - b), np.cos(a - b))


def pca_axis(xy: np.ndarray) -> np.ndarray:
    centered = xy - xy.mean(axis=0)
    if len(centered) < 3:
        return np.array([1.0, 0.0], dtype=float)
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    axis = vt[0].astype(float)
    return axis / max(np.linalg.norm(axis), 1e-12)


def load_boundary_map(boundary_csv: Path | None) -> BoundaryMap:
    if boundary_csv is None or not boundary_csv.exists():
        return {}
    df = pd.read_csv(boundary_csv)
    need = {
        "run_id",
        "lane_kf",
        "point_index",
        "center_x",
        "center_y",
        "lower_x",
        "lower_y",
        "upper_x",
        "upper_y",
    }
    if not need.issubset(df.columns):
        return {}
    return {
        (int(run_id), int(lane_kf)): group.sort_values("point_index").reset_index(drop=True)
        for (run_id, lane_kf), group in df.groupby(["run_id", "lane_kf"], sort=True)
    }


def _closed_xy(group: pd.DataFrame) -> np.ndarray:
    xy = group.sort_values("vertex_index")[["x", "y"]].to_numpy(float)
    if len(xy) and not np.allclose(xy[0], xy[-1]):
        xy = np.vstack([xy, xy[0]])
    return xy


def load_site_roadway(path: Path | None):
    """Closed roadway polygon from Jounieh (outer−islands) or TGSIM curb CSV."""
    if path is None or not path.exists():
        return None
    df = pd.read_csv(path)
    if {"kind", "polygon_id", "x", "y", "vertex_index"}.issubset(df.columns):
        outers = []
        holes = []
        for (kind, pid), g in df.groupby(["kind", "polygon_id"], sort=True):
            poly = Polygon(_closed_xy(g))
            if not poly.is_valid:
                poly = poly.buffer(0)
            if str(kind) == "outer":
                outers.append(poly)
            else:
                holes.append(poly)
        if not outers:
            return None
        road = unary_union(outers)
        for hole in holes:
            road = road.difference(hole)
        return road if not road.is_empty else None
    if {"x", "y", "part_index", "vertex_index"}.issubset(df.columns):
        rings = [
            _closed_xy(g)
            for _, g in df.sort_values(["part_index", "vertex_index"]).groupby("part_index")
        ]
        if not rings:
            return None
        poly = Polygon(rings[0], rings[1:]) if len(rings) > 1 else Polygon(rings[0])
        if not poly.is_valid:
            poly = poly.buffer(0)
        return poly if not poly.is_empty else None
    return None


def polygon_path_error(points: np.ndarray, roadway, boundary_buffer: float = 1.5) -> np.ndarray:
    """Inside: near-edge cost in the last ``buffer`` m. Outside: distance + buffer."""
    pts = np.atleast_2d(np.asarray(points, dtype=float))
    if roadway is None or len(pts) == 0:
        return np.zeros(len(pts), dtype=float)
    geoms = shapely.points(pts)
    d_edge = np.asarray(shapely.distance(geoms, roadway.boundary), dtype=float)
    inside = np.asarray(shapely.contains(roadway, geoms), dtype=bool)
    return np.where(inside, np.maximum(0.0, boundary_buffer - d_edge), d_edge + boundary_buffer)


class OnRoadDestField:
    """Geodesic-to-dest on a site polygon. Cached per destination cell."""

    def __init__(self, roadway, cell: float | None = None):
        self.roadway = roadway
        minx, miny, maxx, maxy = roadway.bounds
        span = max(maxx - minx, maxy - miny)
        self.cell = float(cell if cell is not None else (1.0 if span < 80.0 else 2.0))
        pad = 2.0 * self.cell
        self.x0 = float(minx - pad)
        self.y0 = float(miny - pad)
        nx = int(np.ceil((maxx + pad - self.x0) / self.cell)) + 1
        ny = int(np.ceil((maxy + pad - self.y0) / self.cell)) + 1
        xs = self.x0 + self.cell * np.arange(nx)
        ys = self.y0 + self.cell * np.arange(ny)
        xx, yy = np.meshgrid(xs, ys)
        pts = np.column_stack([xx.ravel(), yy.ravel()])
        self.inside = np.asarray(shapely.contains(roadway, shapely.points(pts)), dtype=bool).reshape((ny, nx))
        self.nx = nx
        self.ny = ny
        self._dist: dict[tuple[int, int], np.ndarray] = {}
        offsets = [
            (-1, 0, self.cell),
            (1, 0, self.cell),
            (0, -1, self.cell),
            (0, 1, self.cell),
            (-1, -1, self.cell * np.sqrt(2.0)),
            (-1, 1, self.cell * np.sqrt(2.0)),
            (1, -1, self.cell * np.sqrt(2.0)),
            (1, 1, self.cell * np.sqrt(2.0)),
        ]
        self._nbrs = [(di, dj, float(c)) for di, dj, c in offsets if di or dj]

    def _snap(self, xy: np.ndarray) -> tuple[int, int]:
        j = int(np.clip(round((float(xy[0]) - self.x0) / self.cell), 0, self.nx - 1))
        i = int(np.clip(round((float(xy[1]) - self.y0) / self.cell), 0, self.ny - 1))
        if self.inside[i, j]:
            return i, j
        best = None
        best_d = 1e18
        for r in range(1, max(self.nx, self.ny)):
            for di in range(-r, r + 1):
                for dj in range(-r, r + 1):
                    if max(abs(di), abs(dj)) != r:
                        continue
                    ni, nj = i + di, j + dj
                    if 0 <= ni < self.ny and 0 <= nj < self.nx and self.inside[ni, nj]:
                        d = di * di + dj * dj
                        if d < best_d:
                            best_d = d
                            best = (ni, nj)
            if best is not None:
                return best
        return i, j

    def _field(self, dest: np.ndarray) -> np.ndarray:
        key = self._snap(dest)
        cached = self._dist.get(key)
        if cached is not None:
            return cached
        inf = 1.0e9
        dist = np.full((self.ny, self.nx), inf, dtype=float)
        si, sj = key
        if not self.inside[si, sj]:
            self._dist[key] = dist
            return dist
        dist[si, sj] = 0.0
        heap = [(0.0, si, sj)]
        while heap:
            d, i, j = heappop(heap)
            if d > dist[i, j] + 1e-9:
                continue
            for di, dj, step in self._nbrs:
                ni, nj = i + di, j + dj
                if not (0 <= ni < self.ny and 0 <= nj < self.nx and self.inside[ni, nj]):
                    continue
                nd = d + step
                if nd + 1e-9 < dist[ni, nj]:
                    dist[ni, nj] = nd
                    heappush(heap, (nd, ni, nj))
        self._dist[key] = dist
        return dist

    def remaining_and_dir(self, points: np.ndarray, dest: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        pts = np.atleast_2d(np.asarray(points, dtype=float))
        field = self._field(dest)
        remaining = np.zeros(len(pts), dtype=float)
        dest_dir = np.zeros((len(pts), 2), dtype=float)
        geoms = shapely.points(pts)
        d_edge = np.asarray(shapely.distance(geoms, self.roadway.boundary), dtype=float)
        inside = np.asarray(shapely.contains(self.roadway, geoms), dtype=bool)
        for k, xy in enumerate(pts):
            i, j = self._snap(xy)
            rem = float(field[i, j])
            if not np.isfinite(rem) or rem >= 1.0e8:
                rem = float(np.linalg.norm(np.asarray(dest, dtype=float) - xy)) + 50.0
            remaining[k] = rem if inside[k] else rem + float(d_edge[k])
            best = None
            best_val = rem
            for di, dj, _ in self._nbrs:
                ni, nj = i + di, j + dj
                if 0 <= ni < self.ny and 0 <= nj < self.nx and field[ni, nj] + 1e-9 < best_val:
                    best_val = field[ni, nj]
                    best = (di, dj)
            if best is None:
                vec = np.asarray(dest, dtype=float) - xy
            else:
                vec = np.array([best[1] * self.cell, best[0] * self.cell], dtype=float)
            n = float(np.linalg.norm(vec))
            dest_dir[k] = vec / n if n > 1e-6 else np.array([1.0, 0.0])
        return remaining, dest_dir


def destination_choice_features(
    ego_pos: np.ndarray,
    cand_pos: np.ndarray,
    dest_pos: np.ndarray,
    heading: float | None = None,
    on_road: OnRoadDestField | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Dest alignment + remaining distance (Euclidean, or on-road geodesic if given)."""
    ego_pos = np.asarray(ego_pos, dtype=float)
    dest_pos = np.asarray(dest_pos, dtype=float)
    move = cand_pos - ego_pos
    move_norm = np.linalg.norm(move, axis=1)
    if on_road is not None:
        remaining, dest_dir = on_road.remaining_and_dir(cand_pos, dest_pos)
        geoms = shapely.points(np.atleast_2d(cand_pos))
        lat = np.asarray(shapely.distance(geoms, on_road.roadway.boundary), dtype=float)
        inside = np.asarray(shapely.contains(on_road.roadway, geoms), dtype=bool)
        lat = np.where(inside, 0.0, lat)
        dest_norm = np.linalg.norm(dest_dir, axis=1)
        dir_cos = np.ones(len(cand_pos), dtype=float)
        valid = (move_norm > 1e-6) & (dest_norm > 1e-6)
        dir_cos[valid] = np.einsum(
            "ij,ij->i",
            move[valid] / move_norm[valid, None],
            dest_dir[valid],
        )
        stagnant = move_norm <= 1e-6
        if np.any(stagnant) and heading is not None:
            tangent = np.array([np.cos(heading), np.sin(heading)], dtype=float)
            dir_cos[stagnant] = dest_dir[stagnant] @ tangent
        dist_abs = np.column_stack([remaining, lat])
        return dir_cos, dist_abs
    dest_vec = dest_pos - cand_pos
    dest_norm = np.linalg.norm(dest_vec, axis=1)
    dir_cos = np.ones(len(cand_pos), dtype=float)
    valid = (move_norm > 1e-6) & (dest_norm > 1e-6)
    dir_cos[valid] = np.einsum(
        "ij,ij->i",
        move[valid] / move_norm[valid, None],
        dest_vec[valid] / dest_norm[valid, None],
    )
    stagnant = (move_norm <= 1e-6) & (dest_norm > 1e-6)
    if np.any(stagnant) and heading is not None:
        dest_from_ego = dest_pos - ego_pos
        n = float(np.linalg.norm(dest_from_ego))
        if n > 1e-6:
            tangent = np.array([np.cos(heading), np.sin(heading)], dtype=float)
            dir_cos[stagnant] = float(tangent @ (dest_from_ego / n))
    dist_abs = np.abs(cand_pos - dest_pos)
    return dir_cos, dist_abs


def heading_choice_features(
    ego_pos: np.ndarray,
    heading: float,
    cand_pos: np.ndarray,
    reference_pos: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Deprecated heading-frame features; kept for older result JSON replay."""
    return destination_choice_features(ego_pos, cand_pos, reference_pos, heading=heading)


def corridor_geometry_from_map(
    run_id: int,
    lane_kf: int,
    boundary_map: BoundaryMap,
) -> CorridorGeometry | None:
    boundary = boundary_map.get((int(run_id), int(lane_kf)))
    if boundary is None:
        return None
    center = boundary[["center_x", "center_y"]].to_numpy(float)
    return CorridorGeometry.from_center(center)


_BOUNDARY_ARRAY_CACHE: Dict[int, Tuple[np.ndarray, ...]] = {}


def project_points_on_corridor(
    corridor: CorridorGeometry,
    points: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    s_vals = np.zeros(len(points), dtype=float)
    n_vals = np.zeros(len(points), dtype=float)
    tangents = np.zeros((len(points), 2), dtype=float)
    for idx, point in enumerate(points):
        s_vals[idx], n_vals[idx], tangents[idx] = corridor.project(point)
    return s_vals, n_vals, tangents


def corridor_choice_features(
    ego_pos: np.ndarray,
    cand_pos: np.ndarray,
    reference_pos: np.ndarray,
    dest_pos: np.ndarray,
    corridor: CorridorGeometry | None,
) -> tuple[np.ndarray, np.ndarray]:
    """Direction + Frenet distance features for each candidate."""
    move = cand_pos - ego_pos
    move_norm = np.linalg.norm(move, axis=1)
    if corridor is not None:
        _, _, tangent_at_ego = corridor.project(ego_pos)
        dir_cos = np.zeros(len(cand_pos), dtype=float)
        valid = move_norm > 1e-6
        dir_cos[valid] = (move[valid] / move_norm[valid, None]) @ tangent_at_ego
        dir_cos[~valid] = 1.0
        ref_s, ref_n, _ = corridor.project(reference_pos)
        cand_s, cand_n, _ = project_points_on_corridor(corridor, cand_pos)
        dist_abs = np.column_stack([np.abs(cand_s - ref_s), np.abs(cand_n - ref_n)])
        return dir_cos, dist_abs

    dest_vec = dest_pos - cand_pos
    dest_norm = np.linalg.norm(dest_vec, axis=1)
    valid = (move_norm > 1e-6) & (dest_norm > 1e-6)
    dir_cos = np.zeros(len(cand_pos), dtype=float)
    dir_cos[valid] = np.einsum(
        "ij,ij->i",
        move[valid] / move_norm[valid, None],
        dest_vec[valid] / dest_norm[valid, None],
    )
    dir_cos[~valid] = 1.0
    dist_abs = np.abs(cand_pos - dest_pos)
    return dir_cos, dist_abs


def _boundary_arrays(boundary: pd.DataFrame) -> tuple[np.ndarray, ...]:
    """Cached numpy view of one run/lane boundary plus precomputed segment terms."""
    key = id(boundary)
    cached = _BOUNDARY_ARRAY_CACHE.get(key)
    if cached is not None:
        return cached
    center = boundary[["center_x", "center_y"]].to_numpy(float)
    lower = boundary[["lower_x", "lower_y"]].to_numpy(float)
    upper = boundary[["upper_x", "upper_y"]].to_numpy(float)
    if len(center) >= 2:
        ab = center[1:] - center[:-1]
        denom = np.einsum("ij,ij->i", ab, ab) + 1e-12
    else:
        ab = np.zeros((0, 2), dtype=float)
        denom = np.zeros(0, dtype=float)
    cached = (center, lower, upper, ab, denom)
    _BOUNDARY_ARRAY_CACHE[key] = cached
    return cached


def closest_boundary_points(points: np.ndarray, boundary: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Interpolate lower/upper boundary points at closest centerline segment for each point.

    Vectorized over points and segments: the scalar double loop dominated the
    closed-loop calibration objective (~88% of rollout time).
    """
    center, lower, upper, ab, denom = _boundary_arrays(boundary)
    points = np.atleast_2d(np.asarray(points, dtype=float))
    if len(center) < 2:
        return np.repeat(lower[:1], len(points), axis=0), np.repeat(upper[:1], len(points), axis=0)

    # (N, S) projection parameter of each point onto each centerline segment.
    ap = points[:, None, :] - center[None, :-1, :]
    t = np.clip(np.einsum("nsk,sk->ns", ap, ab) / denom[None, :], 0.0, 1.0)
    q = center[None, :-1, :] + t[:, :, None] * ab[None, :, :]
    d2 = np.sum((points[:, None, :] - q) ** 2, axis=2)
    best_i = np.argmin(d2, axis=1)
    rows = np.arange(len(points))
    best_t = t[rows, best_i][:, None]

    out_lower = (1.0 - best_t) * lower[best_i] + best_t * lower[best_i + 1]
    out_upper = (1.0 - best_t) * upper[best_i] + best_t * upper[best_i + 1]
    return out_lower, out_upper


def boundary_path_error(
    candidate_pos: np.ndarray,
    run_id: int,
    lane_kf: int,
    boundary_map: BoundaryMap,
    boundary_buffer: float = 1.5,
) -> np.ndarray:
    """
    Boundary-aware path error for calibration.

    Inside the corridor, the error is only positive within boundary_buffer meters
    from either edge. Outside the corridor, it grows with the outside distance.
    """
    boundary = boundary_map.get((int(run_id), int(lane_kf)))
    if boundary is None:
        return np.zeros(len(candidate_pos), dtype=float)
    lower, upper = closest_boundary_points(candidate_pos, boundary)
    mid = 0.5 * (lower + upper)
    lateral_vec = upper - lower
    width = np.linalg.norm(lateral_vec, axis=1)
    unit = lateral_vec / np.maximum(width[:, None], 1e-9)
    signed = np.einsum("ij,ij->i", candidate_pos - mid, unit)
    half_width = 0.5 * width
    clearance = half_width - np.abs(signed)
    return np.where(clearance >= 0.0, np.maximum(0.0, boundary_buffer - clearance), -clearance + boundary_buffer)


def boundary_summary(df: pd.DataFrame) -> dict[str, Any]:
    """Robust run/lane highway envelope in PCA Frenet coordinates."""
    out: dict[str, Any] = {}
    for (run_id, lane_kf), group in df.groupby(["run_id", "lane_kf"], sort=True):
        xy = group[["xloc_kf", "yloc_kf"]].to_numpy(float)
        origin = xy.mean(axis=0)
        tangent = pca_axis(xy)
        normal = np.array([-tangent[1], tangent[0]], dtype=float)
        st = (xy - origin) @ np.column_stack([tangent, normal])
        out[f"run_{int(run_id)}_lane_{int(lane_kf)}"] = {
            "run_id": int(run_id),
            "lane_kf": int(lane_kf),
            "origin_xy": origin.tolist(),
            "tangent_xy": tangent.tolist(),
            "normal_xy": normal.tolist(),
            "s_min_p01": float(np.quantile(st[:, 0], 0.01)),
            "s_max_p99": float(np.quantile(st[:, 0], 0.99)),
            "lateral_min_p01": float(np.quantile(st[:, 1], 0.01)),
            "lateral_max_p99": float(np.quantile(st[:, 1], 0.99)),
            "lateral_min_p005": float(np.quantile(st[:, 1], 0.005)),
            "lateral_max_p995": float(np.quantile(st[:, 1], 0.995)),
        }
    return out


def load_and_prepare(
    csv_path: Path,
    class_id: float | None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    cols = [
        "id",
        "time",
        "xloc_kf",
        "yloc_kf",
        "lane_kf",
        "speed_kf",
        "acceleration_kf",
        "class",
        "run_id",
    ]
    df = pd.read_csv(csv_path, usecols=cols)
    df = df.dropna(subset=["id", "time", "xloc_kf", "yloc_kf", "speed_kf", "run_id"])
    df = df.sort_values(["run_id", "id", "time"]).reset_index(drop=True)
    group = df.groupby(["run_id", "id"], sort=False)

    for col in ("xloc_kf", "yloc_kf", "speed_kf", "time"):
        df[f"next_{col}"] = group[col].shift(-1)
    df["final_x"] = group["xloc_kf"].transform("last")
    df["final_y"] = group["yloc_kf"].transform("last")
    df["nominal_y"] = group["yloc_kf"].transform("median")

    def _desired(s: pd.Series) -> float:
        moving = s[s.to_numpy(float) > 1.0]
        src = moving if len(moving) >= 5 else s
        return float(np.quantile(src.to_numpy(float), 0.85))

    df["desired_speed"] = group["speed_kf"].transform(_desired)

    dx = df["next_xloc_kf"] - df["xloc_kf"]
    dy = df["next_yloc_kf"] - df["yloc_kf"]
    df["dt"] = df["next_time"] - df["time"]
    step = np.hypot(dx.to_numpy(float), dy.to_numpy(float))
    heading = np.arctan2(dy.to_numpy(float), dx.to_numpy(float))
    heading = np.where(step > 0.05, heading, np.nan)
    df["heading"] = heading
    df["heading"] = df.groupby(["run_id", "id"], sort=False)["heading"].transform(
        lambda s: s.ffill().bfill()
    )
    df["vx"] = df["speed_kf"] * np.cos(df["heading"])
    df["vy"] = df["speed_kf"] * np.sin(df["heading"])

    time_ok = (
        df["next_xloc_kf"].notna()
        & df["next_yloc_kf"].notna()
        & df["dt"].between(0.05, 0.2)
        & np.isfinite(df["heading"])
    )
    scene = df[time_ok].copy()
    ego = scene
    if class_id is not None:
        ego = ego[ego["class"] == class_id]
    ego = ego[ego["speed_kf"] > 0.5].copy()
    return ego, scene


def make_agent(row: pd.Series, agent_id: int) -> TrafficAgent:
    vel = np.array([row["vx"], row["vy"]], dtype=float)
    return TrafficAgent(
        agent_id=agent_id,
        pos=np.array([row["xloc_kf"], row["yloc_kf"]], dtype=float),
        vel=vel,
        dest=np.array([row["final_x"], row["final_y"]], dtype=float),
        desired_speed=float(row["desired_speed"]),
        nominal_y=float(row["nominal_y"]),
        run_id=int(row["run_id"]),
        lane_kf=int(row["lane_kf"]),
        utility_reference_pos=np.array([row["next_xloc_kf"], row["next_yloc_kf"]], dtype=float),
        heading_angle=float(row["heading"]),
    )


def collision_probability(
    cand_pos: np.ndarray,
    neighbors: list[TrafficAgent],
    dt: float,
    variances: np.ndarray,
    heading: float | np.ndarray | None = None,
    vehicle_length: float = 4.5,
    vehicle_width: float = 1.8,
) -> np.ndarray:
    """Soft footprint presence matching ``utility_model.collision_penalty``.

    Uses ego-body-frame surface gaps (beyond L×W) so OBB contact has unit
    presence; anisotropic ``variances`` are σ² along body long/lat axes.
    """
    if not neighbors:
        return np.zeros(len(cand_pos), dtype=float)
    cand_pos = np.asarray(cand_pos, dtype=float)
    n = len(cand_pos)
    sig_long = float(np.sqrt(max(float(variances[0]), 1e-12)))
    sig_lat = float(np.sqrt(max(float(variances[1]), 1e-12)))
    if heading is None:
        headings = np.zeros(n, dtype=float)
    else:
        headings = np.broadcast_to(np.asarray(heading, dtype=float), (n,))
    c = np.cos(headings)
    s = np.sin(headings)
    p_sum = np.zeros(n, dtype=float)
    for other in neighbors:
        mu = np.asarray(other.pos, dtype=float) + np.asarray(other.vel, dtype=float) * dt
        diff = cand_pos - mu
        d_long = c * diff[:, 0] + s * diff[:, 1]
        d_lat = -s * diff[:, 0] + c * diff[:, 1]
        gap_long = np.maximum(0.0, np.abs(d_long) - float(vehicle_length))
        gap_lat = np.maximum(0.0, np.abs(d_lat) - float(vehicle_width))
        mahal_sq = (gap_long / sig_long) ** 2 + (gap_lat / sig_lat) ** 2
        p_sum += np.exp(-0.5 * mahal_sq)
    return np.minimum(p_sum, 1.0)


def observed_neighbors(
    same_time: pd.DataFrame | None,
    ego_pos: np.ndarray,
    ego_id: int | float,
    neighbor_radius: float,
    max_neighbors: int,
) -> list[TrafficAgent]:
    if same_time is None or len(same_time) < 2:
        return []
    dx = same_time["xloc_kf"].to_numpy(float) - float(ego_pos[0])
    dy = same_time["yloc_kf"].to_numpy(float) - float(ego_pos[1])
    dist = np.hypot(dx, dy)
    mask = (same_time["id"].to_numpy() != ego_id) & (dist <= neighbor_radius)
    neighbor_rows = same_time.loc[mask].assign(_dist=dist[mask]).sort_values("_dist").head(max_neighbors)
    return [make_agent(nrow, i + 1) for i, (_, nrow) in enumerate(neighbor_rows.iterrows())]


def build_choice_sample(
    row: pd.Series,
    same_time: pd.DataFrame,
    sim_config: dict[str, Any],
    neighbor_radius: float,
    max_neighbors: int,
    boundary_map: BoundaryMap,
) -> ChoiceSample | None:
    ego = make_agent(row, 0)
    dt = float(row["dt"])
    sim_config = dict(sim_config)
    sim_config["dt"] = dt

    neighbors = observed_neighbors(
        same_time,
        ego.pos,
        ego_id=row["id"],
        neighbor_radius=neighbor_radius,
        max_neighbors=max_neighbors,
    )

    candidates = generate_candidate_actions(ego, dt, sim_config)
    cand_pos = np.vstack([c["pos"] for c in candidates])
    cand_speed = np.array([float(c["speed"]) for c in candidates], dtype=float)
    cand_heading = np.array([float(c["heading"]) for c in candidates], dtype=float)

    actual_next = np.array([row["next_xloc_kf"], row["next_yloc_kf"]], dtype=float)
    actual_speed = float(row["next_speed_kf"])
    actual_heading = float(row["heading"])
    match_cost = (
        np.linalg.norm(cand_pos - actual_next, axis=1)
        + 0.5 * np.abs(cand_speed - actual_speed)
        + 0.5 * np.abs(angle_diff(cand_heading, actual_heading))
    )
    target_index = int(np.argmin(match_cost))
    reference_pos = actual_next

    corridor = None
    if sim_config.get("utility_frame") == "destination":
        dir_cos, dist_abs = destination_choice_features(
            ego.pos,
            cand_pos,
            ego.dest,
            heading=float(ego.heading_angle),
            on_road=sim_config.get("_on_road_dest"),
        )
    else:
        corridor = corridor_geometry_from_map(int(row["run_id"]), int(row["lane_kf"]), boundary_map)
        dir_cos, dist_abs = corridor_choice_features(
            ego.pos,
            cand_pos,
            reference_pos,
            ego.dest,
            corridor,
        )

    variances = np.array(sim_config["collision_pred_variances"], dtype=float)
    p_collision = collision_probability(
        cand_pos,
        neighbors,
        dt,
        variances,
        heading=cand_heading,
        vehicle_length=float(sim_config.get("vehicle_length", 4.5)),
        vehicle_width=float(sim_config.get("vehicle_width", 1.8)),
    )
    roadway = sim_config.get("_site_roadway")
    if roadway is not None:
        path_error = polygon_path_error(
            cand_pos,
            roadway,
            boundary_buffer=float(sim_config.get("boundary_buffer", 1.5)),
        )
    else:
        path_error = boundary_path_error(
            cand_pos,
            run_id=int(row["run_id"]),
            lane_kf=int(row["lane_kf"]),
            boundary_map=boundary_map,
            boundary_buffer=float(sim_config.get("boundary_buffer", 1.5)),
        )
    if not np.any(np.isfinite(path_error)) or np.allclose(path_error, 0.0):
        path_error = np.abs(cand_pos[:, 1] - float(row["nominal_y"]))

    return ChoiceSample(
        run_id=int(row["run_id"]),
        lane_kf=int(row["lane_kf"]),
        vehicle_id=int(row["id"]),
        time=float(row["time"]),
        dir_cos=dir_cos,
        cand_speed=cand_speed,
        dist_abs=dist_abs,
        collision_prob=p_collision,
        path_error=path_error,
        current_speed=float(row["speed_kf"]),
        desired_speed=float(row["desired_speed"]),
        target_index=target_index,
        target_match_cost=float(match_cost[target_index]),
        path_mode=str(sim_config.get("path_mode", "boundary")),
    )


def sample_choices(
    df: pd.DataFrame,
    args: argparse.Namespace,
    sim_config: dict[str, Any],
    boundary_map: BoundaryMap,
    scene_df: pd.DataFrame | None = None,
) -> list[ChoiceSample]:
    rng = np.random.default_rng(args.seed)
    if len(df) > args.n_samples * 20:
        candidate_rows = df.sample(n=args.n_samples * 20, random_state=args.seed)
    else:
        candidate_rows = df

    neighbor_src = scene_df if scene_df is not None else df
    grouped = {key: group for key, group in neighbor_src.groupby(["run_id", "time"], sort=False)}
    samples: list[ChoiceSample] = []
    log_progress(
        f"Building up to {args.n_samples} choice samples from {len(candidate_rows)} candidate rows...",
        args.verbose,
    )
    next_report = max(args.n_samples // 10, 1)
    for _, row in candidate_rows.sample(frac=1.0, random_state=args.seed).iterrows():
        key = (row["run_id"], row["time"])
        same_time = grouped.get(key)
        if same_time is None or len(same_time) < 2:
            continue
        sample = build_choice_sample(
            row,
            same_time,
            sim_config,
            neighbor_radius=args.neighbor_radius,
            max_neighbors=args.max_neighbors,
            boundary_map=boundary_map,
        )
        if sample is not None:
            samples.append(sample)
            if args.verbose and len(samples) % next_report == 0:
                log_progress(f"  choice samples: {len(samples)}/{args.n_samples}", args.verbose)
        if len(samples) >= args.n_samples:
            break
    rng.shuffle(samples)
    log_progress(f"Built {len(samples)} choice samples.", args.verbose)
    return samples


def utility_values(sample: ChoiceSample, params: dict[str, float]) -> np.ndarray:
    v_cand = np.maximum(sample.cand_speed, 0.0)
    v_des = max(sample.desired_speed, 1e-6)
    rho = np.minimum(v_cand, v_des) / np.maximum(v_cand, v_des)
    rho = np.maximum(rho, 1e-6)
    exponent = (params["xi_i"] - 1.0) / 2.0
    speed_term = rho / (1.0 + rho**exponent)

    d_eff = params["w_x"] * sample.dist_abs[:, 0] + params["w_y"] * sample.dist_abs[:, 1]
    h_p = max(2.0 * sample.current_speed, 5.0)
    distance_term = 1.0 / (1.0 + (d_eff / h_p) ** params["gamma"])
    err = np.maximum(sample.path_error, 0.0)
    if getattr(sample, "path_mode", "boundary") == "site_polygon":
        cap = 1.5
        path_penalty = 1.0 - np.exp(-params["beta"] * np.minimum(err, cap) ** 2) + np.maximum(0.0, err - cap)
    else:
        path_penalty = 1.0 - np.exp(-params["beta"] * err**2)

    return (
        params["S_theta"] * sample.dir_cos
        + params["S_v"] * speed_term
        + params["S_d"] * distance_term
        - params["w_c"] * sample.collision_prob
        - params["w_ell"] * path_penalty
    )


def candidate_features_for_state(
    agent: TrafficAgent,
    neighbors: list[TrafficAgent],
    sim_config: dict[str, Any],
    run_id: int,
    lane_kf: int,
    boundary_map: BoundaryMap,
    reference_pos: np.ndarray | None = None,
    params: dict[str, float] | None = None,
) -> tuple[list[dict[str, Any]], ChoiceSample]:
    candidates = generate_candidate_actions(agent, sim_config["dt"], sim_config)
    cand_pos = np.vstack([c["pos"] for c in candidates])
    cand_speed = np.array([float(c["speed"]) for c in candidates], dtype=float)
    cand_heading = np.array([float(c["heading"]) for c in candidates], dtype=float)

    if reference_pos is None:
        reference_pos = agent.utility_reference_pos
    if reference_pos is None:
        reference_pos = agent.dest
    if sim_config.get("utility_frame") == "destination":
        dir_cos, dist_abs = destination_choice_features(
            agent.pos,
            cand_pos,
            np.asarray(agent.dest, dtype=float),
            heading=float(agent.heading_angle),
            on_road=sim_config.get("_on_road_dest"),
        )
    else:
        corridor = corridor_geometry_from_map(run_id, lane_kf, boundary_map)
        dir_cos, dist_abs = corridor_choice_features(
            agent.pos,
            cand_pos,
            np.asarray(reference_pos, dtype=float),
            agent.dest,
            corridor,
        )

    variances = collision_variances(params, sim_config)
    p_collision = collision_probability(
        cand_pos,
        neighbors,
        float(sim_config["dt"]),
        variances,
        heading=cand_heading,
        vehicle_length=float(sim_config.get("vehicle_length", 4.5)),
        vehicle_width=float(sim_config.get("vehicle_width", 1.8)),
    )
    roadway = sim_config.get("_site_roadway")
    if roadway is not None:
        path_error = polygon_path_error(
            cand_pos,
            roadway,
            boundary_buffer=float(sim_config.get("boundary_buffer", 1.5)),
        )
    else:
        path_error = boundary_path_error(
            cand_pos,
            run_id=run_id,
            lane_kf=lane_kf,
            boundary_map=boundary_map,
            boundary_buffer=float(sim_config.get("boundary_buffer", 1.5)),
        )
    if not np.any(np.isfinite(path_error)) or np.allclose(path_error, 0.0):
        path_error = np.abs(cand_pos[:, 1] - float(agent.nominal_y))

    sample = ChoiceSample(
        run_id=run_id,
        lane_kf=lane_kf,
        vehicle_id=agent.agent_id,
        time=0.0,
        dir_cos=dir_cos,
        cand_speed=cand_speed,
        dist_abs=dist_abs,
        collision_prob=p_collision,
        path_error=path_error,
        current_speed=agent.speed,
        desired_speed=agent.desired_speed,
        target_index=0,
        target_match_cost=0.0,
        path_mode=str(sim_config.get("path_mode", "boundary")),
    )
    return candidates, sample


def select_best_candidate_with_boundary(
    agent: TrafficAgent,
    neighbors: list[TrafficAgent],
    params: dict[str, float],
    sim_config: dict[str, Any],
    run_id: int,
    lane_kf: int,
    boundary_map: BoundaryMap,
    reference_pos: np.ndarray | None = None,
) -> dict[str, Any]:
    candidates, sample = candidate_features_for_state(
        agent,
        neighbors,
        sim_config,
        run_id,
        lane_kf,
        boundary_map,
        reference_pos=reference_pos,
        params=params,
    )
    utilities = utility_values(sample, params)
    return candidates[int(np.argmax(utilities))]


TrajectoryGroup = Tuple[int, int, List[Dict[str, Any]]]


def enumerate_trajectory_groups(df: pd.DataFrame) -> list[TrajectoryGroup]:
    """One entry per (run_id, vehicle_id) trajectory, sorted by time."""
    groups: list[TrajectoryGroup] = []
    for (run_id, vehicle_id), group in df.groupby(["run_id", "id"], sort=True):
        rows = group.sort_values("time").to_dict("records")
        if len(rows) < 2:
            continue
        groups.append((int(run_id), int(vehicle_id), rows))
    return groups


def windows_from_groups(
    groups: list[TrajectoryGroup],
    group_indices: np.ndarray | list[int],
    n_windows: int,
    horizon_steps: int,
    rng: np.random.Generator,
) -> list[RolloutWindow]:
    """Sample short windows drawn only from the given trajectories."""
    if n_windows <= 0 or horizon_steps <= 0:
        return []
    starts: list[tuple[int, int]] = []
    for gi in group_indices:
        rows = groups[int(gi)][2]
        starts.extend((int(gi), start) for start in range(len(rows) - 1))
    if not starts:
        return []

    selected = rng.choice(len(starts), size=min(n_windows, len(starts)), replace=False)
    windows: list[RolloutWindow] = []
    for idx in selected:
        group_idx, start = starts[int(idx)]
        run_id, vehicle_id, rows = groups[group_idx]
        end = min(start + horizon_steps + 1, len(rows))
        window_rows = rows[start:end]
        if len(window_rows) < 2:
            continue
        windows.append(
            RolloutWindow(
                run_id=run_id,
                lane_kf=int(window_rows[0]["lane_kf"]),
                vehicle_id=vehicle_id,
                rows=window_rows,
            )
        )
    return windows


def sample_rollout_windows(
    df: pd.DataFrame,
    n_windows: int,
    horizon_steps: int,
    seed: int,
) -> list[RolloutWindow]:
    if n_windows <= 0 or horizon_steps <= 0:
        return []
    groups = enumerate_trajectory_groups(df)
    if not groups:
        return []
    rng = np.random.default_rng(seed)
    return windows_from_groups(groups, range(len(groups)), n_windows, horizon_steps, rng)


def split_rollout_windows(
    df: pd.DataFrame,
    horizon_steps: int,
    seed: int,
    n_train: int,
    n_val: int,
    n_test: int,
    group_fractions: tuple[float, float] = (0.6, 0.2),
) -> tuple[list[RolloutWindow], list[RolloutWindow], list[RolloutWindow], dict[str, Any]]:
    """Split windows into train/validation/test by *vehicle*, not by window.

    Windows starting at consecutive indices of one trajectory overlap almost
    entirely, so splitting windows directly would leak calibration data into the
    held-out sets. Partitioning whole trajectories keeps the splits independent.
    """
    groups = enumerate_trajectory_groups(df)
    if not groups:
        return [], [], [], {"n_vehicles": 0}

    rng = np.random.default_rng(seed)
    order = rng.permutation(len(groups))
    n_groups = len(order)
    n_train_groups = int(round(group_fractions[0] * n_groups))
    n_val_groups = int(round(group_fractions[1] * n_groups))
    n_train_groups = max(1, min(n_train_groups, n_groups - 2))
    n_val_groups = max(1, min(n_val_groups, n_groups - n_train_groups - 1))
    train_groups = order[:n_train_groups]
    val_groups = order[n_train_groups : n_train_groups + n_val_groups]
    test_groups = order[n_train_groups + n_val_groups :]

    train = windows_from_groups(groups, train_groups, n_train, horizon_steps, rng)
    val = windows_from_groups(groups, val_groups, n_val, horizon_steps, rng)
    test = windows_from_groups(groups, test_groups, n_test, horizon_steps, rng)
    info = {
        "n_vehicles": int(n_groups),
        "n_vehicles_train": int(len(train_groups)),
        "n_vehicles_val": int(len(val_groups)),
        "n_vehicles_test": int(len(test_groups)),
        "n_windows_train": len(train),
        "n_windows_val": len(val),
        "n_windows_test": len(test),
        "horizon_steps": int(horizon_steps),
        "split_by": "vehicle_trajectory",
    }
    return train, val, test, info


def rollout_loss(
    windows: list[RolloutWindow],
    grouped_by_time: dict[tuple[float, float], pd.DataFrame],
    params: dict[str, float],
    sim_config: dict[str, Any],
    args: argparse.Namespace,
    boundary_map: BoundaryMap,
) -> float:
    if not windows:
        return 0.0

    window_losses: list[float] = []
    for window in windows:
        agent = make_agent(window.rows[0], 0)
        step_losses: list[float] = []
        for row in window.rows[:-1]:
            dt = float(row["dt"])
            if not np.isfinite(dt) or dt <= 0:
                continue
            local_config = dict(sim_config)
            local_config["dt"] = dt
            same_time = grouped_by_time.get((row["run_id"], row["time"]))
            neighbors = observed_neighbors(
                same_time,
                agent.pos,
                ego_id=row["id"],
                neighbor_radius=args.neighbor_radius,
                max_neighbors=args.max_neighbors,
            )
            chosen = select_best_candidate_with_boundary(
                agent,
                neighbors,
                params,
                local_config,
                run_id=int(row["run_id"]),
                lane_kf=int(row["lane_kf"]),
                boundary_map=boundary_map,
            )
            agent.update_state_from_candidate(chosen, dt, local_config["destination_threshold"])

            observed_pos = np.array([row["next_xloc_kf"], row["next_yloc_kf"]], dtype=float)
            pos_error = float(np.linalg.norm(agent.pos - observed_pos))
            speed_error = abs(float(agent.speed) - float(row["next_speed_kf"]))
            heading_error = abs(float(angle_diff(np.array([agent.heading]), float(row["heading"]))[0]))
            step_losses.append(
                pos_error
                + args.closed_loop_speed_weight * speed_error
                + args.closed_loop_heading_weight * heading_error
            )
        if step_losses:
            window_losses.append(float(np.mean(step_losses)))

    if not window_losses:
        return 0.0
    return float(np.mean(window_losses))


def mean_target_rank(samples: list[ChoiceSample], params: dict[str, float]) -> float:
    if not samples:
        return 0.0
    ranks: list[float] = []
    for sample in samples:
        utilities = utility_values(sample, params)
        order = np.argsort(utilities)[::-1]
        rank = int(np.where(order == sample.target_index)[0][0]) + 1
        ranks.append(float(rank))
    return float(np.mean(ranks))


def calibration_objective(
    samples: list[ChoiceSample],
    windows: list[RolloutWindow],
    grouped_by_time: dict[tuple[float, float], pd.DataFrame],
    params: dict[str, float],
    sim_config: dict[str, Any],
    args: argparse.Namespace,
    boundary_map: BoundaryMap,
) -> tuple[float, float, float, float]:
    one_step_loss = nll(samples, params, args.temperature) if samples and args.one_step_weight > 0 else 0.0
    closed_loop_loss = (
        rollout_loss(windows, grouped_by_time, params, sim_config, args, boundary_map)
        if windows and args.closed_loop_weight > 0
        else 0.0
    )
    tracking_loss = (
        mean_target_rank(samples, params) / max(args.tracking_rank_normalizer, 1.0)
        if samples and args.tracking_weight > 0
        else 0.0
    )
    objective = (
        args.one_step_weight * one_step_loss
        + args.closed_loop_weight * closed_loop_loss
        + args.tracking_weight * tracking_loss
    )
    return objective, one_step_loss, closed_loop_loss, tracking_loss


def nll(samples: list[ChoiceSample], params: dict[str, float], temperature: float) -> float:
    losses = []
    temp = max(temperature, 1e-6)
    for sample in samples:
        logits = utility_values(sample, params) / temp
        logits = logits - np.max(logits)
        log_denom = np.log(np.exp(logits).sum() + 1e-12)
        losses.append(float(-(logits[sample.target_index] - log_denom)))
    return float(np.mean(losses))


def random_params(rng: np.random.Generator) -> dict[str, float]:
    return {
        key: float(rng.uniform(low, high))
        for key, (low, high) in PARAM_BOUNDS.items()
    }


def screen_params_by_one_step_nll(
    rng: np.random.Generator,
    samples: list[ChoiceSample],
    n_trials: int,
    args: argparse.Namespace,
    keep_count: int,
    verbose: bool = False,
    label: str = "one-step screening",
) -> list[tuple[float, dict[str, float]]]:
    total_trials = max(n_trials, 1)
    log_progress(
        f"{label}: scoring {total_trials} random parameter sets on {len(samples)} samples...",
        verbose,
    )
    pre_scored: list[tuple[float, dict[str, float]]] = []
    report_every = max(total_trials // 10, 1)
    best_so_far = float("inf")
    for trial_idx in range(total_trials):
        params = random_params(rng)
        one_step_loss = nll(samples, params, args.temperature) if samples else 0.0
        tracking_loss = (
            mean_target_rank(samples, params) / max(args.tracking_rank_normalizer, 1.0)
            if samples
            else 0.0
        )
        score = args.one_step_weight * one_step_loss + args.tracking_weight * tracking_loss
        pre_scored.append((score, params))
        best_so_far = min(best_so_far, score)
        if verbose and ((trial_idx + 1) % report_every == 0 or trial_idx + 1 == total_trials):
            log_progress(
                f"  {label}: {trial_idx + 1}/{total_trials} trials, best screen={best_so_far:.4f}",
                verbose,
            )
    if keep_count <= 0:
        return pre_scored
    pre_scored.sort(key=lambda x: x[0])
    kept = pre_scored[: min(keep_count, len(pre_scored))]
    log_progress(
        f"{label}: kept top {len(kept)} / {total_trials} by one-step + tracking screen",
        verbose,
    )
    return kept


def summarize_scored_trials(
    scored: list[tuple[float, float, float, float, dict[str, float], int, int]],
    args: argparse.Namespace,
) -> dict[str, Any]:
    """Build best/robust summaries and identifiability metrics from scored trials."""
    if not scored:
        raise ValueError("No scored calibration trials to summarize")

    scored = sorted(scored, key=lambda x: x[0])
    best_objective, best_nll, best_closed_loop_loss, best_tracking_loss, best_params, _, _ = scored[0]
    rank_norm = max(args.tracking_rank_normalizer, 1.0)

    # Near-optimal cloud: trials within relative/absolute tolerance of the best objective.
    rel_tol = float(getattr(args, "near_optimal_rel_tol", 0.05))
    abs_tol = float(getattr(args, "near_optimal_abs_tol", 0.05))
    obj_cut = best_objective + max(abs_tol, rel_tol * max(abs(best_objective), 1e-9))
    near = [row for row in scored if row[0] <= obj_cut]
    used_fallback_top = False
    if len(near) < 3:
        # Fall back to at least a small top set so robust stats are defined.
        near = scored[: max(3, min(10, len(scored)))]
        used_fallback_top = True

    top_n = max(5, int(len(scored) * args.top_fraction))
    top = scored[:top_n]

    def _matrix(
        rows: list[tuple[float, float, float, float, dict[str, float], int, int]]
    ) -> dict[str, np.ndarray]:
        return {
            key: np.array([row[4][key] for row in rows], dtype=float)
            for key in UTILITY_PARAM_KEYS
        }

    def _quantile_ranges(matrix: dict[str, np.ndarray]) -> dict[str, dict[str, float]]:
        return {
            key: {
                "p05": float(np.quantile(values, 0.05)),
                "p50": float(np.quantile(values, 0.50)),
                "p95": float(np.quantile(values, 0.95)),
                "p025": float(np.quantile(values, 0.025)),
                "p975": float(np.quantile(values, 0.975)),
            }
            for key, values in matrix.items()
        }

    top_matrix = _matrix(top)
    near_matrix = _matrix(near)
    ranges = _quantile_ranges(top_matrix)
    near_ranges = _quantile_ranges(near_matrix)

    robust_params = {key: float(np.median(near_matrix[key])) for key in UTILITY_PARAM_KEYS}

    # Re-score robust params is done by caller if needed; store placeholder metrics from medoids.
    objectives = np.array([row[0] for row in scored], dtype=float)
    near_objectives = np.array([row[0] for row in near], dtype=float)
    param_cv = {}
    for key in UTILITY_PARAM_KEYS:
        vals = near_matrix[key]
        param_cv[key] = float(np.std(vals, ddof=1) / max(abs(np.mean(vals)), 1e-9))

    # Correlation among near-optimal params (identifiability / tradeoffs).
    corr = np.corrcoef(np.vstack([near_matrix[k] for k in UTILITY_PARAM_KEYS]))
    corr = np.nan_to_num(corr, nan=0.0)
    strong_pairs = []
    for i, ki in enumerate(UTILITY_PARAM_KEYS):
        for j, kj in enumerate(UTILITY_PARAM_KEYS):
            if j <= i:
                continue
            rho = float(corr[i, j])
            if abs(rho) >= 0.5:
                strong_pairs.append({"param_a": ki, "param_b": kj, "corr": rho})
    strong_pairs.sort(key=lambda d: -abs(d["corr"]))

    trials_out = []
    for objective, one_step_loss, closed_loop_loss, tracking_loss, params, seed, restart in scored:
        row_tuple = (objective, one_step_loss, closed_loop_loss, tracking_loss, params, seed, restart)
        # Mark membership in the set actually used for robust_params.
        in_near = any(
            abs(objective - n[0]) < 1e-15
            and seed == n[5]
            and restart == n[6]
            and all(abs(params[k] - n[4][k]) < 1e-12 for k in UTILITY_PARAM_KEYS)
            for n in near
        )
        trials_out.append(
            {
                "objective": float(objective),
                "nll": float(one_step_loss),
                "closed_loop_loss": float(closed_loop_loss),
                "tracking_rank": float(tracking_loss * rank_norm),
                "seed": int(seed),
                "restart": int(restart),
                "near_optimal": bool(in_near),
                **{key: float(params[key]) for key in UTILITY_PARAM_KEYS},
            }
        )

    flatness_ratio = float(
        (near_objectives.max() - near_objectives.min()) / max(abs(best_objective), 1e-9)
    )
    return {
        "best_objective": float(best_objective),
        "best_nll": float(best_nll),
        "best_closed_loop_loss": float(best_closed_loop_loss),
        "best_tracking_rank": float(best_tracking_loss * rank_norm),
        "best_params": best_params,
        "robust_params": robust_params,
        "recommended_ranges_from_top_trials": ranges,
        "near_optimal_ranges": near_ranges,
        "top_trials": trials_out,
        "identifiability": {
            "n_scored_trials": len(scored),
            "n_near_optimal": len(near),
            "used_top_fallback_for_robust": used_fallback_top,
            "objective_cutoff": float(obj_cut),
            "near_optimal_rel_tol": rel_tol,
            "near_optimal_abs_tol": abs_tol,
            "best_objective": float(best_objective),
            "near_optimal_objective_min": float(near_objectives.min()),
            "near_optimal_objective_max": float(near_objectives.max()),
            "near_optimal_objective_range_rel": flatness_ratio,
            "all_objective_p50": float(np.median(objectives)),
            "all_objective_p95": float(np.quantile(objectives, 0.95)),
            "param_cv_near_optimal": param_cv,
            "strong_param_correlations": strong_pairs[:15],
            "verdict": (
                "non_identifiable_flat_ridge"
                if flatness_ratio <= 0.10 and len(near) >= 5
                else (
                    "weakly_identified"
                    if flatness_ratio <= 0.20
                    else "more_peaked_or_under_explored"
                )
            ),
        },
        "n_samples": 0,  # filled by caller
        "n_rollout_windows": 0,
        "closed_loop_candidates": len(scored),
        "closed_loop_horizon_steps": args.closed_loop_horizon_steps,
        "one_step_weight": args.one_step_weight,
        "closed_loop_weight": args.closed_loop_weight,
        "tracking_weight": args.tracking_weight,
        "utility_frame": "corridor",
        "n_trials": args.n_trials,
        "top_fraction": args.top_fraction,
        "temperature": args.temperature,
        "n_restarts": int(getattr(args, "n_restarts", 1)),
    }


def calibrate_once(
    samples: list[ChoiceSample],
    windows: list[RolloutWindow],
    grouped_by_time: dict[tuple[float, float], pd.DataFrame],
    sim_config: dict[str, Any],
    args: argparse.Namespace,
    boundary_map: BoundaryMap,
    seed: int,
    restart: int,
) -> list[tuple[float, float, float, float, dict[str, float], int, int]]:
    """Run one random-search calibration restart; return scored closed-loop trials."""
    rng = np.random.default_rng(seed)
    closed_loop_candidates = min(max(args.closed_loop_candidates, 1), max(args.n_trials, 1))
    candidates = screen_params_by_one_step_nll(
        rng,
        samples,
        n_trials=args.n_trials,
        args=args,
        keep_count=closed_loop_candidates,
        verbose=args.verbose,
        label=f"Global calibration (restart {restart + 1})",
    )
    scored: list[tuple[float, float, float, float, dict[str, float], int, int]] = []
    log_progress(
        f"Restart {restart + 1}: evaluating {len(candidates)} candidates with "
        f"{len(windows)} closed-loop windows x {args.closed_loop_horizon_steps} steps...",
        args.verbose,
    )
    for i, (_, params) in enumerate(candidates):
        objective, one_step_loss, closed_loop_loss, tracking_loss = calibration_objective(
            samples,
            windows,
            grouped_by_time,
            params,
            sim_config,
            args,
            boundary_map,
        )
        scored.append(
            (objective, one_step_loss, closed_loop_loss, tracking_loss, params, seed, restart)
        )
        if args.verbose and (
            (i + 1) % max(len(candidates) // 10, 1) == 0 or i + 1 == len(candidates)
        ):
            best = min(scored, key=lambda x: x[0])
            log_progress(
                f"  restart {restart + 1}: {i + 1}/{len(candidates)} candidates, "
                f"best_objective={best[0]:.4f}, nll={best[1]:.4f}, "
                f"rollout={best[2]:.4f}, tracking={best[3]:.4f}",
                args.verbose,
            )
    return scored


def evaluate_params_objective(
    samples: list[ChoiceSample],
    windows: list[RolloutWindow],
    grouped_by_time: dict[tuple[float, float], pd.DataFrame],
    params: dict[str, float],
    sim_config: dict[str, Any],
    args: argparse.Namespace,
    boundary_map: BoundaryMap,
) -> tuple[float, float, float, float]:
    return calibration_objective(
        samples,
        windows,
        grouped_by_time,
        params,
        sim_config,
        args,
        boundary_map,
    )


def _held_out_selection(
    result: dict[str, Any],
    all_scored: list[tuple[float, float, float, float, dict[str, float], int, int]],
    grouped_by_time: dict[tuple[float, float], pd.DataFrame],
    sim_config: dict[str, Any],
    args: argparse.Namespace,
    boundary_map: BoundaryMap,
    val_windows: list[RolloutWindow],
    test_windows: list[RolloutWindow],
) -> dict[str, Any]:
    """Select the working parameters on validation windows and score them on test."""
    out: dict[str, Any] = {}
    if not val_windows:
        return out

    def closed_loop(params: dict[str, float], w: list[RolloutWindow]) -> float:
        if not w:
            return float("nan")
        return float(rollout_loss(w, grouped_by_time, params, sim_config, args, boundary_map))

    candidates = {
        "best": result["best_params"],
        "robust": result["robust_params"],
    }
    val_losses = {name: closed_loop(params, val_windows) for name, params in candidates.items()}
    out["val_closed_loop_loss"] = {k: float(v) for k, v in val_losses.items()}

    # Validation spread across the fit-acceptable cloud: how much does the choice
    # of point inside the near-optimal region matter out of sample?
    top_k = max(int(getattr(args, "val_top_k", 0)), 0)
    if top_k:
        near_sorted = sorted(all_scored, key=lambda row: row[0])[:top_k]
        cloud = [closed_loop(row[4], val_windows) for row in near_sorted]
        cloud = [c for c in cloud if np.isfinite(c)]
        if cloud:
            out["val_closed_loop_cloud"] = {
                "n": len(cloud),
                "min": float(np.min(cloud)),
                "p50": float(np.median(cloud)),
                "max": float(np.max(cloud)),
                "std": float(np.std(cloud)),
            }

    selected = min(val_losses, key=lambda k: val_losses[k])
    out["working_params"] = candidates[selected]
    out["working_params_source"] = f"{selected}_selected_on_validation_windows"
    out["test_closed_loop_loss"] = float(closed_loop(candidates[selected], test_windows))
    return out


def calibrate(
    samples: list[ChoiceSample],
    windows: list[RolloutWindow],
    grouped_by_time: dict[tuple[float, float], pd.DataFrame],
    sim_config: dict[str, Any],
    args: argparse.Namespace,
    boundary_map: BoundaryMap,
    val_windows: list[RolloutWindow] | None = None,
    test_windows: list[RolloutWindow] | None = None,
    split_info: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Multi-restart random search with robust near-optimal parameter aggregation.

    Fitting uses ``windows``; ``val_windows`` selects the working parameter vector
    among fit-acceptable candidates; ``test_windows`` is touched once, only for the
    selected vector, so the reported closed-loop number is genuinely held out.
    """
    n_restarts = max(int(getattr(args, "n_restarts", 1)), 1)
    all_scored: list[tuple[float, float, float, float, dict[str, float], int, int]] = []
    for restart in range(n_restarts):
        seed = int(args.seed) + restart
        all_scored.extend(
            calibrate_once(
                samples,
                windows,
                grouped_by_time,
                sim_config,
                args,
                boundary_map,
                seed=seed,
                restart=restart,
            )
        )

    result = summarize_scored_trials(all_scored, args)
    result["n_samples"] = len(samples)
    result["n_rollout_windows"] = len(windows)

    # Evaluate the robust (median near-optimal) parameter vector on the true objective.
    robust_objective, robust_nll, robust_closed, robust_track = evaluate_params_objective(
        samples,
        windows,
        grouped_by_time,
        result["robust_params"],
        sim_config,
        args,
        boundary_map,
    )
    result["robust_objective"] = float(robust_objective)
    result["robust_nll"] = float(robust_nll)
    result["robust_closed_loop_loss"] = float(robust_closed)
    result["robust_tracking_rank"] = float(robust_track * max(args.tracking_rank_normalizer, 1.0))
    result["working_params"] = result["robust_params"]
    result["working_params_source"] = "median_of_near_optimal_trials"
    if split_info:
        result["window_split"] = dict(split_info)

    result.update(
        _held_out_selection(
            result,
            all_scored,
            grouped_by_time,
            sim_config,
            args,
            boundary_map,
            val_windows or [],
            test_windows or [],
        )
    )

    idinfo = result["identifiability"]
    log_progress(
        "Identifiability: "
        f"near-optimal={idinfo['n_near_optimal']}/{idinfo['n_scored_trials']}, "
        f"obj_range_rel={idinfo['near_optimal_objective_range_rel']:.4f}, "
        f"verdict={idinfo['verdict']}",
        args.verbose,
    )
    log_progress(
        f"Robust params objective={result['robust_objective']:.4f} "
        f"(best={result['best_objective']:.4f})",
        args.verbose,
    )
    return result


def calibrate_local_params(
    samples: list[ChoiceSample],
    windows: list[RolloutWindow],
    grouped_by_time: dict[tuple[float, float], pd.DataFrame],
    sim_config: dict[str, Any],
    args: argparse.Namespace,
    boundary_map: BoundaryMap,
    n_trials: int,
    seed: int,
) -> tuple[float, float, float, float, dict[str, float]]:
    """Small random-search calibration for one vehicle ID diagnostic plot."""
    if not samples and not windows:
        raise ValueError("Cannot calibrate local parameters without samples or rollout windows")
    rng = np.random.default_rng(seed)
    keep_count = min(max(args.per_id_closed_loop_candidates, 1), max(n_trials, 1))
    candidates = screen_params_by_one_step_nll(
        rng,
        samples,
        n_trials=n_trials,
        args=args,
        keep_count=keep_count,
        verbose=False,
    )
    best_objective = float("inf")
    best_nll = float("inf")
    best_closed_loop_loss = float("inf")
    best_tracking_loss = float("inf")
    best_params: dict[str, float] | None = None
    for _, params in candidates:
        objective, one_step_loss, closed_loop_loss, tracking_loss = calibration_objective(
            samples,
            windows,
            grouped_by_time,
            params,
            sim_config,
            args,
            boundary_map,
        )
        if objective < best_objective:
            best_objective = objective
            best_nll = one_step_loss
            best_closed_loop_loss = closed_loop_loss
            best_tracking_loss = tracking_loss
            best_params = params
    assert best_params is not None
    return best_objective, best_nll, best_closed_loop_loss, best_tracking_loss, best_params


def plot_site_polygon(ax, path: Path) -> None:
    if path is None or not path.exists():
        return
    df = pd.read_csv(path)
    if {"kind", "polygon_id", "x", "y", "vertex_index"}.issubset(df.columns):
        for (kind, pid), g in df.groupby(["kind", "polygon_id"], sort=True):
            xy = _closed_xy(g)
            ax.plot(
                xy[:, 0],
                xy[:, 1],
                lw=2.2 if str(kind) == "outer" else 1.8,
                color="C0" if str(kind) == "outer" else "C3",
                label=f"{kind} {pid}",
            )
        return
    if {"part_index", "x", "y", "vertex_index"}.issubset(df.columns):
        for i, (_, g) in enumerate(df.groupby("part_index", sort=True)):
            xy = _closed_xy(g)
            ax.plot(xy[:, 0], xy[:, 1], lw=2.0, color="C0", label="site curb" if i == 0 else None)


def plot_all_trajectories(
    df: pd.DataFrame,
    out_path: Path,
    boundary_csv: Path | None = None,
    site_polygon_csv: Path | None = None,
    max_points: int = 250_000,
) -> None:
    """XY overview of all trajectory points, colored by run/lane."""
    plot_df = df
    if len(plot_df) > max_points:
        plot_df = plot_df.sample(n=max_points, random_state=0)

    fig, ax = plt.subplots(figsize=(9, 8), constrained_layout=True)
    for (run_id, lane_kf), group in plot_df.groupby(["run_id", "lane_kf"], sort=True):
        ax.scatter(
            group["xloc_kf"],
            group["yloc_kf"],
            s=1,
            alpha=0.18,
            label=f"run {int(run_id)}, lane {int(lane_kf)}",
        )
    if site_polygon_csv is not None:
        plot_site_polygon(ax, site_polygon_csv)
    elif boundary_csv is not None and boundary_csv.exists():
        bdf = pd.read_csv(boundary_csv)
        group_cols = ["run_id", "lane_kf"] if "lane_kf" in bdf.columns else ["run_id"]
        for key, group in bdf.groupby(group_cols, sort=True):
            if isinstance(key, tuple):
                run_id, lane_kf = key
                label = f"run {int(run_id)}, lane {int(lane_kf)}"
            else:
                label = f"run {int(key)}"
            ax.plot(group["lower_x"], group["lower_y"], lw=2, label=f"{label} lower")
            ax.plot(group["upper_x"], group["upper_y"], lw=2, label=f"{label} upper")
    ax.set_title("All vehicle trajectory points")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.25)
    ax.legend(markerscale=6, fontsize=8)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def samples_for_vehicle(
    group: pd.DataFrame,
    grouped_by_time: dict[tuple[float, float], pd.DataFrame],
    sim_config: dict[str, Any],
    args: argparse.Namespace,
    boundary_map: BoundaryMap,
) -> list[ChoiceSample]:
    """Build observed-choice samples for a single vehicle."""
    rows = group.sort_values("time")
    if len(rows) > args.per_id_samples:
        rows = rows.iloc[np.linspace(0, len(rows) - 1, args.per_id_samples).astype(int)]
    samples: list[ChoiceSample] = []
    for _, row in rows.iterrows():
        same_time = grouped_by_time.get((row["run_id"], row["time"]))
        if same_time is None or len(same_time) < 2:
            continue
        sample = build_choice_sample(
            row,
            same_time,
            sim_config,
            neighbor_radius=args.neighbor_radius,
            max_neighbors=args.max_neighbors,
            boundary_map=boundary_map,
        )
        if sample is not None:
            samples.append(sample)
    return samples


def simulate_vehicle(
    group: pd.DataFrame,
    grouped_by_time: dict[tuple[float, float], pd.DataFrame],
    params: dict[str, float],
    sim_config: dict[str, Any],
    neighbor_radius: float,
    max_neighbors: int,
    boundary_map: BoundaryMap,
) -> pd.DataFrame:
    """Closed-loop utility simulation for one vehicle against observed neighbors."""
    group = group.sort_values("time")
    rows = group.to_dict("records")
    if not rows:
        return pd.DataFrame()
    first = pd.Series(rows[0])
    agent = make_agent(first, 0)
    sim_rows = [
        {
            "time": float(first["time"]),
            "x": float(agent.pos[0]),
            "y": float(agent.pos[1]),
            "vx": float(agent.vel[0]),
            "vy": float(agent.vel[1]),
            "speed": float(agent.speed),
        }
    ]
    for row_dict in rows[:-1]:
        row = pd.Series(row_dict)
        dt = float(row["dt"])
        if not np.isfinite(dt) or dt <= 0:
            continue
        local_config = dict(sim_config)
        local_config["dt"] = dt
        same_time = grouped_by_time.get((row["run_id"], row["time"]))
        neighbors = observed_neighbors(
            same_time,
            agent.pos,
            ego_id=row["id"],
            neighbor_radius=neighbor_radius,
            max_neighbors=max_neighbors,
        )
        agent.run_id = int(row["run_id"])
        agent.lane_kf = int(row["lane_kf"])
        if local_config.get("utility_frame") == "destination":
            reference_pos = np.asarray(agent.dest, dtype=float)
        else:
            reference_pos = np.array([row["next_xloc_kf"], row["next_yloc_kf"]], dtype=float)
        agent.utility_reference_pos = reference_pos
        chosen = select_best_candidate_with_boundary(
            agent,
            neighbors,
            params,
            local_config,
            run_id=int(row["run_id"]),
            lane_kf=int(row["lane_kf"]),
            boundary_map=boundary_map,
            reference_pos=reference_pos,
        )
        agent.update_state_from_candidate(chosen, dt, local_config["destination_threshold"])
        next_time = float(row["next_time"])
        sim_rows.append(
            {
                "time": next_time,
                "x": float(agent.pos[0]),
                "y": float(agent.pos[1]),
                "vx": float(agent.vel[0]),
                "vy": float(agent.vel[1]),
                "speed": float(agent.speed),
            }
        )
    return pd.DataFrame(sim_rows)


def params_text(params: dict[str, float]) -> str:
    rows = [f"{k}={params[k]:.2g}" for k in UTILITY_PARAM_KEYS]
    return "\n".join(rows)


def plot_vehicle_simulated_vs_observed(
    group: pd.DataFrame,
    sim_df: pd.DataFrame,
    params: dict[str, float],
    local_nll: float,
    out_path: Path,
    boundary_map: BoundaryMap,
    site_polygon_csv: Path | None = None,
) -> None:
    """Per-ID observed vs simulated x-y, x(t), y(t), vx(t), vy(t), speed(t)."""
    group = group.sort_values("time")
    fig, axes = plt.subplots(2, 3, figsize=(16, 9), constrained_layout=True)
    t_obs = group["time"].to_numpy(float)
    t_sim = sim_df["time"].to_numpy(float)

    run_id = int(group["run_id"].iloc[0])
    lane_kf = int(group["lane_kf"].iloc[0])
    boundary = boundary_map.get((run_id, lane_kf))
    obs_points = group[["xloc_kf", "yloc_kf"]].to_numpy(float)
    sim_points = sim_df[["x", "y"]].to_numpy(float)
    obs_lower = obs_upper = sim_lower = sim_upper = None
    use_lane_tubes = site_polygon_csv is None and boundary is not None
    if use_lane_tubes:
        obs_lower, obs_upper = closest_boundary_points(obs_points, boundary)
        sim_lower, sim_upper = closest_boundary_points(sim_points, boundary)
        axes[0, 0].plot(boundary["lower_x"], boundary["lower_y"], color="0.55", lw=2, label="boundary lower")
        axes[0, 0].plot(boundary["upper_x"], boundary["upper_y"], color="0.25", lw=2, label="boundary upper")
    if site_polygon_csv is not None:
        plot_site_polygon(axes[0, 0], site_polygon_csv)

    axes[0, 0].plot(group["xloc_kf"], group["yloc_kf"], lw=2, label="observed")
    axes[0, 0].plot(sim_df["x"], sim_df["y"], "--", lw=2, label="simulated")
    dest = group[["final_x", "final_y"]].iloc[-1].to_numpy(float)
    axes[0, 0].scatter(
        dest[0],
        dest[1],
        s=80,
        marker="*",
        c="k",
        zorder=6,
        label="destination",
    )
    axes[0, 0].set_xlabel("x (m)")
    axes[0, 0].set_ylabel("y (m)")
    axes[0, 0].set_aspect("equal", adjustable="box")
    axes[0, 0].legend()

    axes[0, 1].plot(t_obs, group["xloc_kf"], lw=1.5, label="observed")
    axes[0, 1].plot(t_sim, sim_df["x"], "--", lw=1.5, label="simulated")
    if obs_lower is not None and obs_upper is not None:
        axes[0, 1].plot(t_obs, obs_lower[:, 0], ":", color="0.55", lw=1.2, label="obs lower/upper")
        axes[0, 1].plot(t_obs, obs_upper[:, 0], ":", color="0.55", lw=1.2)
    if sim_lower is not None and sim_upper is not None:
        axes[0, 1].plot(t_sim, sim_lower[:, 0], "-.", color="0.25", lw=1.0, label="sim lower/upper")
        axes[0, 1].plot(t_sim, sim_upper[:, 0], "-.", color="0.25", lw=1.0)
    axes[0, 1].set_ylabel("x (m)")

    axes[0, 2].plot(t_obs, group["yloc_kf"], lw=1.5, label="observed")
    axes[0, 2].plot(t_sim, sim_df["y"], "--", lw=1.5, label="simulated")
    if obs_lower is not None and obs_upper is not None:
        axes[0, 2].plot(t_obs, obs_lower[:, 1], ":", color="0.55", lw=1.2, label="obs lower/upper")
        axes[0, 2].plot(t_obs, obs_upper[:, 1], ":", color="0.55", lw=1.2)
    if sim_lower is not None and sim_upper is not None:
        axes[0, 2].plot(t_sim, sim_lower[:, 1], "-.", color="0.25", lw=1.0, label="sim lower/upper")
        axes[0, 2].plot(t_sim, sim_upper[:, 1], "-.", color="0.25", lw=1.0)
    axes[0, 2].set_ylabel("y (m)")

    axes[1, 0].plot(t_obs, group["vx"], lw=1.5, label="observed")
    axes[1, 0].plot(t_sim, sim_df["vx"], "--", lw=1.5, label="simulated")
    axes[1, 0].set_ylabel("x speed vx (m/s)")
    axes[1, 0].set_xlabel("time (s)")

    axes[1, 1].plot(t_obs, group["vy"], lw=1.5, label="observed")
    axes[1, 1].plot(t_sim, sim_df["vy"], "--", lw=1.5, label="simulated")
    axes[1, 1].set_ylabel("y speed vy (m/s)")
    axes[1, 1].set_xlabel("time (s)")

    axes[1, 2].plot(t_obs, group["speed_kf"], lw=1.5, label="observed")
    axes[1, 2].plot(t_sim, sim_df["speed"], "--", lw=1.5, label="simulated")
    axes[1, 2].set_ylabel("speed (m/s)")
    axes[1, 2].set_xlabel("time (s)")

    for ax in axes.ravel():
        ax.grid(True, alpha=0.3)

    vehicle_id = int(group["id"].iloc[0])
    fig.suptitle(
        f"Observed vs simulated, ID {vehicle_id}, run {run_id}, lane_kf {lane_kf}, "
        f"local NLL={local_nll:.3f}",
        fontsize=13,
    )
    axes[0, 0].text(
        1.04,
        0.98,
        "Best parameters for this ID:\n" + params_text(params),
        transform=axes[0, 0].transAxes,
        va="top",
        ha="left",
        fontsize=8,
        family="monospace",
        bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.75},
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def infer_site_polygon_csv(traj_csv: Path | None) -> Path | None:
    """Jounieh/TGSIM use one shared curb polygon for every vehicle ID."""
    if traj_csv is None:
        return None
    text = str(traj_csv).replace("\\", "/").lower()
    if "lebanon_jounieh" in text:
        path = REPO_ROOT / "data" / "Lebanon_Jounieh" / "Jounieh_Road_Boundaries.csv"
        return path if path.exists() else None
    if "tgsim" in text:
        path = REPO_ROOT / "data" / "TGSIM FB" / "derived_boundaries" / "street_boundaries.csv"
        return path if path.exists() else None
    return None


def apply_urban_site_config(cfg: EnvConfig, args: argparse.Namespace, ego_df: pd.DataFrame) -> None:
    """Site curb + paper dest utility: U_dir and U_d use each ID's destination."""
    if getattr(args, "site_polygon_csv", None) is None:
        args.site_polygon_csv = infer_site_polygon_csv(getattr(args, "csv", None))
    roadway = load_site_roadway(getattr(args, "site_polygon_csv", None))
    if roadway is None:
        cfg.sim_config["utility_frame"] = "corridor"
        return
    cfg.sim_config["_site_roadway"] = roadway
    cfg.sim_config["_on_road_dest"] = OnRoadDestField(roadway)
    cfg.sim_config["utility_frame"] = "destination"
    cfg.sim_config["path_mode"] = "site_polygon"
    cfg.sim_config["steer_from_rest"] = True
    cfg.sim_config["min_steer_speed"] = 0.5
    cfg.sim_config["candidate_accel_grid"] = [-5.0, -4.0, -3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0]
    cfg.sim_config["max_accel"] = 5.0
    vmax = getattr(args, "max_agent_speed", None)
    if vmax is None and len(ego_df):
        vmax = max(4.0, float(np.quantile(ego_df["speed_kf"].to_numpy(float), 0.99)) * 1.15)
    if vmax is not None:
        cfg.sim_config["max_agent_speed"] = float(vmax)
    # Do not use per-lane PCA lower/upper for these sites.
    args.boundary_csv = None


def plot_id_timeseries(
    df: pd.DataFrame,
    out_dir: Path,
    max_ids: int,
    plot_all_ids: bool,
    global_params: dict[str, float],
    args: argparse.Namespace,
    scene_df: pd.DataFrame | None = None,
) -> int:
    """Create per-ID observed-vs-simulated plots; bounded by default."""
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg = EnvConfig()
    apply_urban_site_config(cfg, args, df)
    boundary_map = {}
    if cfg.sim_config.get("_site_roadway") is None:
        boundary_map = load_boundary_map(args.boundary_csv)
    neighbor_src = scene_df if scene_df is not None else df
    grouped_by_time = {key: group for key, group in neighbor_src.groupby(["run_id", "time"], sort=False)}
    groups = list(df.groupby(["run_id", "id"], sort=True))
    if not plot_all_ids:
        groups = groups[:max_ids]
    param_rows = []
    n_groups = len(groups)
    log_progress(f"Per-ID diagnostics: processing {n_groups} vehicle(s)...", args.verbose)
    report_every = max(n_groups // 10, 1)
    for plot_idx, ((run_id, vehicle_id), group) in enumerate(groups, start=1):
        local_samples = samples_for_vehicle(
            group, grouped_by_time, cfg.sim_config, args, boundary_map
        )
        local_windows = sample_rollout_windows(
            group,
            n_windows=args.per_id_rollout_windows,
            horizon_steps=args.closed_loop_horizon_steps,
            seed=args.seed + int(vehicle_id),
        )
        if local_samples or local_windows:
            (
                local_objective,
                local_nll,
                local_closed_loop_loss,
                local_tracking_loss,
                local_params,
            ) = calibrate_local_params(
                local_samples,
                local_windows,
                grouped_by_time,
                cfg.sim_config,
                args,
                boundary_map,
                n_trials=args.per_id_trials,
                seed=args.seed + int(vehicle_id),
            )
        else:
            local_objective = float("nan")
            local_nll = float("nan")
            local_closed_loop_loss = float("nan")
            local_tracking_loss = float("nan")
            local_params = global_params
        sim_df = simulate_vehicle(
            group,
            grouped_by_time,
            local_params,
            cfg.sim_config,
            neighbor_radius=args.neighbor_radius,
            max_neighbors=args.max_neighbors,
            boundary_map=boundary_map,
        )
        plot_vehicle_simulated_vs_observed(
            group,
            sim_df,
            local_params,
            local_nll,
            out_dir / f"run_{int(run_id):02d}_id_{int(vehicle_id):06d}_sim_vs_obs.png",
            boundary_map=boundary_map,
            site_polygon_csv=getattr(args, "site_polygon_csv", None),
        )
        row = {
            "run_id": int(run_id),
            "id": int(vehicle_id),
            "lane_kf": int(group["lane_kf"].iloc[0]),
            "local_objective": local_objective,
            "local_nll": local_nll,
            "local_closed_loop_loss": local_closed_loop_loss,
            "local_tracking_rank": local_tracking_loss * max(args.tracking_rank_normalizer, 1.0),
            "n_local_samples": len(local_samples),
            "n_local_rollout_windows": len(local_windows),
        }
        row.update(local_params)
        param_rows.append(row)
        if args.verbose and (plot_idx % report_every == 0 or plot_idx == n_groups):
            log_progress(f"  per-ID plots: {plot_idx}/{n_groups}", args.verbose)
    pd.DataFrame(param_rows).to_csv(out_dir / "per_id_best_params.csv", index=False)
    return len(groups)


def calibration_quality_frame(
    samples: list[ChoiceSample],
    params: dict[str, float],
    temperature: float,
) -> pd.DataFrame:
    rows = []
    temp = max(temperature, 1e-6)
    for sample in samples:
        u = utility_values(sample, params)
        logits = (u - np.max(u)) / temp
        probs = np.exp(logits)
        probs = probs / max(float(probs.sum()), 1e-12)
        order = np.argsort(u)[::-1]
        rank = int(np.where(order == sample.target_index)[0][0]) + 1
        best_non_target = float(np.max(np.delete(u, sample.target_index)))
        rows.append(
            {
                "run_id": sample.run_id,
                "id": sample.vehicle_id,
                "time": sample.time,
                "target_rank": rank,
                "target_probability": float(probs[sample.target_index]),
                "target_margin": float(u[sample.target_index] - best_non_target),
                "target_match_cost": sample.target_match_cost,
            }
        )
    return pd.DataFrame(rows)


def plot_calibration_quality(qdf: pd.DataFrame, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), constrained_layout=True)
    axes[0].hist(qdf["target_rank"], bins=np.arange(1, qdf["target_rank"].max() + 2) - 0.5)
    axes[0].set_title("Observed-like candidate rank")
    axes[0].set_xlabel("rank (1 is best)")
    axes[0].set_ylabel("count")
    axes[1].hist(qdf["target_probability"], bins=30)
    axes[1].set_title("Softmax probability of observed-like candidate")
    axes[1].set_xlabel("probability")
    axes[2].hist(qdf["target_margin"], bins=30)
    axes[2].axvline(0.0, color="k", lw=1, ls="--")
    axes[2].set_title("Utility margin vs best alternative")
    axes[2].set_xlabel("U_target - max U_other")
    for ax in axes:
        ax.grid(True, alpha=0.25)
    fig.savefig(out_dir / "calibration_quality.png", dpi=180)
    plt.close(fig)


def plot_parameter_ranges(result: dict[str, Any], out_dir: Path) -> None:
    ranges = result["recommended_ranges_from_top_trials"]
    keys = list(UTILITY_PARAM_KEYS)
    med = np.array([ranges[k]["p50"] for k in keys], dtype=float)
    lo = np.array([ranges[k]["p05"] for k in keys], dtype=float)
    hi = np.array([ranges[k]["p95"] for k in keys], dtype=float)
    fig, ax = plt.subplots(figsize=(11, 5), constrained_layout=True)
    x = np.arange(len(keys))
    ax.errorbar(x, med, yerr=[med - lo, hi - med], fmt="o", capsize=4)
    ax.set_xticks(x)
    ax.set_xticklabels(keys, rotation=35, ha="right")
    ax.set_ylabel("parameter value")
    ax.set_title("Recommended utility parameter ranges from top calibration trials")
    ax.grid(True, axis="y", alpha=0.3)
    fig.savefig(out_dir / "parameter_ranges.png", dpi=180)
    plt.close(fig)


def plot_identifiability_diagnostics(result: dict[str, Any], out_dir: Path) -> None:
    """Objective flatness, near-optimal parameter spreads, and tradeoff correlations."""
    trials = result.get("top_trials") or []
    if not trials:
        return
    out_dir.mkdir(parents=True, exist_ok=True)
    tdf = pd.DataFrame(trials)
    near = tdf[tdf["near_optimal"]].copy() if "near_optimal" in tdf.columns else tdf.nsmallest(max(5, len(tdf) // 10), "objective")
    if near.empty:
        near = tdf.nsmallest(min(10, len(tdf)), "objective")

    fig, axes = plt.subplots(2, 2, figsize=(12.5, 8.5), constrained_layout=True)

    ax = axes[0, 0]
    objs = np.sort(tdf["objective"].to_numpy(dtype=float))
    ax.plot(np.arange(1, len(objs) + 1), objs, color="#4C78A8", lw=1.8)
    cut = float(result.get("identifiability", {}).get("objective_cutoff", objs[0]))
    ax.axhline(cut, color="#54A24B", ls="--", lw=1.3, label="near-optimal cutoff")
    ax.axhline(objs[0], color="#E45756", ls=":", lw=1.2, label="best")
    ax.set_xlabel("trial rank (best → worst)")
    ax.set_ylabel("objective")
    ax.set_title("Closed-loop objective spectrum")
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False)

    ax = axes[0, 1]
    keys = list(UTILITY_PARAM_KEYS)
    cvs = [float(result["identifiability"]["param_cv_near_optimal"].get(k, np.nan)) for k in keys]
    ax.bar(np.arange(len(keys)), cvs, color="#F58518")
    ax.set_xticks(np.arange(len(keys)))
    ax.set_xticklabels(keys, rotation=35, ha="right")
    ax.set_ylabel("CV = std/|mean|")
    ax.set_title("Parameter dispersion inside near-optimal set")
    ax.grid(True, axis="y", alpha=0.25)

    ax = axes[1, 0]
    # Pair with strongest |corr| if available, else S_v vs S_theta.
    pairs = result.get("identifiability", {}).get("strong_param_correlations") or []
    if pairs:
        ka, kb = pairs[0]["param_a"], pairs[0]["param_b"]
        title = f"Strongest tradeoff: {ka} vs {kb} (corr={pairs[0]['corr']:.2f})"
    else:
        ka, kb = "S_v", "S_theta"
        title = f"{ka} vs {kb} (near-optimal cloud)"
    ax.scatter(tdf[ka], tdf[kb], s=18, alpha=0.25, c="#9ecae1", label="all scored")
    ax.scatter(near[ka], near[kb], s=28, alpha=0.85, c="#E45756", label="near-optimal")
    if "robust_params" in result:
        ax.scatter(
            [result["robust_params"][ka]],
            [result["robust_params"][kb]],
            s=90,
            marker="D",
            c="#54A24B",
            label="robust",
            zorder=3,
        )
    if "best_params" in result:
        ax.scatter(
            [result["best_params"][ka]],
            [result["best_params"][kb]],
            s=80,
            marker="*",
            c="#4C78A8",
            label="best",
            zorder=3,
        )
    ax.set_xlabel(ka)
    ax.set_ylabel(kb)
    ax.set_title(title)
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False, fontsize=8)

    ax = axes[1, 1]
    mat = near[list(UTILITY_PARAM_KEYS)].to_numpy(dtype=float)
    if len(near) >= 3:
        corr = np.corrcoef(mat.T)
        corr = np.nan_to_num(corr, nan=0.0)
    else:
        corr = np.eye(len(UTILITY_PARAM_KEYS))
    im = ax.imshow(corr, vmin=-1, vmax=1, cmap="coolwarm", aspect="equal")
    ax.set_xticks(np.arange(len(UTILITY_PARAM_KEYS)))
    ax.set_yticks(np.arange(len(UTILITY_PARAM_KEYS)))
    ax.set_xticklabels(UTILITY_PARAM_KEYS, rotation=35, ha="right", fontsize=8)
    ax.set_yticklabels(UTILITY_PARAM_KEYS, fontsize=8)
    ax.set_title("Near-optimal parameter correlations")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.suptitle("Calibration identifiability diagnostics", y=1.02)
    fig.savefig(out_dir / "identifiability_diagnostics.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    # Also dump the trial table for paper / external analysis.
    tdf.to_csv(out_dir / "top_trials.csv", index=False)
    near.to_csv(out_dir / "near_optimal_trials.csv", index=False)


def working_params_from_result(result: dict[str, Any]) -> dict[str, float]:
    if result.get("working_params"):
        return result["working_params"]
    if result.get("robust_params"):
        return result["robust_params"]
    return result["best_params"]


def write_diagnostics(
    df: pd.DataFrame,
    samples: list[ChoiceSample],
    result: dict[str, Any],
    args: argparse.Namespace,
    scene_df: pd.DataFrame | None = None,
) -> None:
    diag_dir = args.diagnostics_dir
    diag_dir.mkdir(parents=True, exist_ok=True)
    work_params = working_params_from_result(result)
    log_progress("Diagnostics: writing all-agents XY plot...", args.verbose)
    plot_all_trajectories(
        df,
        diag_dir / "all_agents_xy.png",
        boundary_csv=args.boundary_csv,
        site_polygon_csv=getattr(args, "site_polygon_csv", None),
    )
    n_id_plots = plot_id_timeseries(
        df,
        diag_dir / "per_id",
        max_ids=args.max_id_plots,
        plot_all_ids=args.plot_all_ids,
        global_params=work_params,
        args=args,
        scene_df=scene_df,
    )
    log_progress("Diagnostics: writing calibration quality plots...", args.verbose)
    qdf = calibration_quality_frame(samples, work_params, args.temperature)
    qdf.to_csv(diag_dir / "calibration_quality_samples.csv", index=False)
    plot_calibration_quality(qdf, diag_dir)
    plot_parameter_ranges(result, diag_dir)
    log_progress("Diagnostics: writing identifiability plots...", args.verbose)
    plot_identifiability_diagnostics(result, diag_dir)
    log_progress(f"Wrote diagnostics to {diag_dir} ({n_id_plots} per-ID plots)", args.verbose)


def main() -> None:
    parser = argparse.ArgumentParser(description="Calibrate utility ranges from trajectory data")
    parser.add_argument(
        "--csv",
        type=Path,
        default=REPO_ROOT / "data" / "Lebanon_Highway" / "Final_Lebanon_Data.csv",
    )
    parser.add_argument("--output", type=Path, default=REPO_ROOT / "Calibration" / "utility_calibration.json")
    parser.add_argument("--n-samples", type=int, default=2000)
    parser.add_argument("--n-trials", type=int, default=400)
    parser.add_argument("--top-fraction", type=float, default=0.1)
    parser.add_argument("--temperature", type=float, default=0.25)
    parser.add_argument(
        "--n-restarts",
        type=int,
        default=3,
        help="Independent random-search restarts (different seeds) aggregated into one near-optimal cloud",
    )
    parser.add_argument(
        "--near-optimal-rel-tol",
        type=float,
        default=0.05,
        help="Relative objective tolerance for the near-optimal set used to build robust_params",
    )
    parser.add_argument(
        "--near-optimal-abs-tol",
        type=float,
        default=0.05,
        help="Absolute objective tolerance for the near-optimal set used to build robust_params",
    )
    parser.add_argument(
        "--closed-loop-windows",
        type=int,
        default=300,
        help="Number of short trajectory windows used in the closed-loop calibration objective",
    )
    parser.add_argument(
        "--closed-loop-val-windows",
        type=int,
        default=100,
        help="Held-out windows used to select the working parameter vector (0 disables)",
    )
    parser.add_argument(
        "--closed-loop-test-windows",
        type=int,
        default=100,
        help="Held-out windows scored once, only for the selected working vector",
    )
    parser.add_argument(
        "--window-split-fractions",
        type=float,
        nargs=2,
        default=(0.6, 0.2),
        metavar=("TRAIN", "VAL"),
        help="Fraction of vehicle trajectories assigned to train and validation (rest is test)",
    )
    parser.add_argument(
        "--val-top-k",
        type=int,
        default=10,
        help="Score this many best-on-train trials on validation to report the cloud spread",
    )
    parser.add_argument(
        "--closed-loop-horizon-steps",
        type=int,
        default=8,
        help="Number of simulation steps per closed-loop calibration window",
    )
    parser.add_argument(
        "--closed-loop-candidates",
        type=int,
        default=80,
        help="Number of one-step-screened global trials evaluated with closed-loop rollouts",
    )
    parser.add_argument(
        "--closed-loop-weight",
        type=float,
        default=1.0,
        help="Weight on closed-loop rollout tracking error in the calibration objective",
    )
    parser.add_argument(
        "--one-step-weight",
        type=float,
        default=0.1,
        help="Weight on one-step choice NLL regularization in the calibration objective",
    )
    parser.add_argument(
        "--tracking-weight",
        type=float,
        default=1.0,
        help="Weight on normalized mean target rank in the calibration objective",
    )
    parser.add_argument(
        "--tracking-rank-normalizer",
        type=int,
        default=63,
        help="Divide mean target rank by this value before adding to the objective",
    )
    parser.add_argument(
        "--closed-loop-speed-weight",
        type=float,
        default=0.5,
        help="Speed-error weight inside each closed-loop rollout step",
    )
    parser.add_argument(
        "--closed-loop-heading-weight",
        type=float,
        default=0.5,
        help="Heading-error weight inside each closed-loop rollout step",
    )
    parser.add_argument("--neighbor-radius", type=float, default=60.0)
    parser.add_argument("--max-neighbors", type=int, default=6)
    parser.add_argument("--class-id", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--diagnostics-dir", type=Path, default=REPO_ROOT / "Calibration" / "diagnostics")
    parser.add_argument(
        "--boundary-csv",
        type=Path,
        default=(
            REPO_ROOT
            / "data"
            / "Lebanon_Highway"
            / "derived_highway_boundaries"
            / "highway_boundaries.csv"
        ),
        help="Per-lane PCA corridor CSV (highway only). Cleared for Jounieh/TGSIM site-curb mode.",
    )
    parser.add_argument(
        "--site-polygon-csv",
        type=Path,
        default=None,
        help="Street curb for path cost. Jounieh/TGSIM also use each ID dest in U_dir/U_d.",
    )
    parser.add_argument(
        "--max-agent-speed",
        type=float,
        default=None,
        help="Cap on bicycle speed (m/s). Default 10 on highway; on sites, 1.15× p99 of ego speed.",
    )
    parser.add_argument("--no-plots", action="store_true", help="Skip calibration diagnostic plots")
    parser.add_argument(
        "--max-id-plots",
        type=int,
        default=200,
        help="Number of per-ID simulated-vs-observed plots unless --plot-all-ids is set",
    )
    parser.add_argument(
        "--per-id-samples",
        type=int,
        default=80,
        help="Maximum observed-choice samples used to calibrate each plotted ID",
    )
    parser.add_argument(
        "--per-id-trials",
        type=int,
        default=120,
        help="Random-search trials for each plotted ID's local calibration",
    )
    parser.add_argument(
        "--per-id-rollout-windows",
        type=int,
        default=1,
        help="Closed-loop trajectory windows used for each plotted ID's local calibration",
    )
    parser.add_argument(
        "--per-id-closed-loop-candidates",
        type=int,
        default=20,
        help="Number of one-step-screened per-ID trials evaluated with closed-loop rollouts",
    )
    parser.add_argument(
        "--plot-all-ids",
        action="store_true",
        help="Write one time-series plot for every vehicle ID (thousands of files)",
    )
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    log_progress(f"Loading trajectory data from {args.csv}...", args.verbose)
    df, scene_df = load_and_prepare(args.csv, args.class_id)
    log_progress(
        f"Loaded {len(df)} ego rows ({df['id'].nunique()} ids) and "
        f"{len(scene_df)} neighbor-scene rows.",
        args.verbose,
    )
    cfg = EnvConfig()
    apply_urban_site_config(cfg, args, df)
    if cfg.sim_config.get("utility_frame") == "destination":
        log_progress(
            "Site dest utility (on-road geodesic to each ID dest) + curb: "
            f"{args.site_polygon_csv}, v_max={cfg.sim_config['max_agent_speed']:.2f} m/s.",
            args.verbose,
        )
    boundary_map = {}
    if cfg.sim_config.get("_site_roadway") is None:
        boundary_map = load_boundary_map(args.boundary_csv)
        if not boundary_map:
            print(
                f"Warning: no run/lane boundaries loaded from {args.boundary_csv}; using nominal-y fallback",
                flush=True,
            )
    samples = sample_choices(df, args, cfg.sim_config, boundary_map, scene_df=scene_df)
    if not samples:
        raise RuntimeError("No calibration samples built from trajectory data")
    rollout_windows, val_windows, test_windows, split_info = split_rollout_windows(
        df,
        horizon_steps=args.closed_loop_horizon_steps,
        seed=args.seed,
        n_train=args.closed_loop_windows,
        n_val=args.closed_loop_val_windows,
        n_test=args.closed_loop_test_windows,
        group_fractions=tuple(args.window_split_fractions),
    )
    grouped_by_time = {key: group for key, group in scene_df.groupby(["run_id", "time"], sort=False)}
    log_progress(
        f"Prepared closed-loop windows (horizon={args.closed_loop_horizon_steps} steps): "
        f"{len(rollout_windows)} train / {len(val_windows)} val / {len(test_windows)} test "
        f"from {split_info.get('n_vehicles', 0)} vehicle trajectories "
        f"(disjoint vehicles per split).",
        args.verbose,
    )

    result = calibrate(
        samples,
        rollout_windows,
        grouped_by_time,
        cfg.sim_config,
        args,
        boundary_map,
        val_windows=val_windows,
        test_windows=test_windows,
        split_info=split_info,
    )
    log_progress("Global calibration finished.", args.verbose)
    result["utility_frame"] = cfg.sim_config.get("utility_frame", "corridor")
    result["max_agent_speed"] = cfg.sim_config.get("max_agent_speed")
    if getattr(args, "site_polygon_csv", None):
        result["site_polygon_csv"] = str(args.site_polygon_csv)
    result["boundary_summary_pca"] = boundary_summary(df)
    result["candidate_grid"] = {
        "acceleration": cfg.sim_config["candidate_accel_grid"],
        "steering": cfg.sim_config["candidate_steering_grid"],
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    # Keep JSON lean enough: trials are also written to diagnostics CSV.
    json_result = dict(result)
    # Embed only a short preview of trials in JSON; full table goes to CSV in diagnostics.
    if json_result.get("top_trials") and len(json_result["top_trials"]) > 40:
        json_result["top_trials_preview"] = json_result["top_trials"][:40]
        json_result["n_top_trials_saved_in_json_preview"] = 40
        # Keep full trials out of the main JSON to avoid huge files; CSV has all.
        del json_result["top_trials"]
    args.output.write_text(json.dumps(json_result, indent=2))
    # Always write full trial tables next to the JSON even if plots are skipped.
    trials_dir = args.diagnostics_dir
    trials_dir.mkdir(parents=True, exist_ok=True)
    if result.get("top_trials"):
        pd.DataFrame(result["top_trials"]).to_csv(trials_dir / "top_trials.csv", index=False)
        near_df = pd.DataFrame([t for t in result["top_trials"] if t.get("near_optimal")])
        if not near_df.empty:
            near_df.to_csv(trials_dir / "near_optimal_trials.csv", index=False)
    if not args.no_plots:
        write_diagnostics(df, samples, result, args, scene_df=scene_df)
    print(f"Wrote {args.output}", flush=True)
    print("Best objective:", f"{result['best_objective']:.4f}", flush=True)
    print("Robust objective:", f"{result['robust_objective']:.4f}", flush=True)
    print("Best closed-loop loss (train):", f"{result['best_closed_loop_loss']:.4f}", flush=True)
    print("Robust closed-loop loss (train):", f"{result['robust_closed_loop_loss']:.4f}", flush=True)
    if result.get("val_closed_loop_loss"):
        val = result["val_closed_loop_loss"]
        print(
            "Validation closed-loop loss:",
            ", ".join(f"{k}={v:.4f}" for k, v in val.items()),
            flush=True,
        )
        print(f"Working params: {result.get('working_params_source')}", flush=True)
    if np.isfinite(result.get("test_closed_loop_loss", float("nan"))):
        print("Test closed-loop loss:", f"{result['test_closed_loop_loss']:.4f}", flush=True)
    if result.get("val_closed_loop_cloud"):
        cloud = result["val_closed_loop_cloud"]
        print(
            "Validation spread over top trials: "
            f"n={cloud['n']}, min={cloud['min']:.4f}, p50={cloud['p50']:.4f}, "
            f"max={cloud['max']:.4f}, std={cloud['std']:.4f}",
            flush=True,
        )
    if result.get("window_split"):
        sp = result["window_split"]
        print(
            "Window split: "
            f"{sp['n_windows_train']}/{sp['n_windows_val']}/{sp['n_windows_test']} windows from "
            f"{sp['n_vehicles_train']}/{sp['n_vehicles_val']}/{sp['n_vehicles_test']} vehicles",
            flush=True,
        )
    print("Best mean target rank:", f"{result['best_tracking_rank']:.2f}", flush=True)
    print("Best NLL:", f"{result['best_nll']:.4f}", flush=True)
    idinfo = result.get("identifiability", {})
    print(
        "Identifiability:",
        f"near-optimal={idinfo.get('n_near_optimal')}/{idinfo.get('n_scored_trials')},",
        f"obj_range_rel={idinfo.get('near_optimal_objective_range_rel', float('nan')):.4f},",
        f"verdict={idinfo.get('verdict')}",
        flush=True,
    )
    print("Working params (robust median of near-optimal trials):", flush=True)
    for key in UTILITY_PARAM_KEYS:
        print(f"  {key}: {result['working_params'][key]:.4f}", flush=True)
    print("Best params (single lowest objective):", flush=True)
    for key in UTILITY_PARAM_KEYS:
        print(f"  {key}: {result['best_params'][key]:.4f}", flush=True)


if __name__ == "__main__":
    main()
