# -*- coding: utf-8 -*-
"""
Highway corridor geometry for PT boundary risk.

Loads envelope polylines produced by derive_boundaries_envelope.py:
  derived_boundaries/polylines_envelope.csv
  columns: run_id, lane_kf, point_index, cx, cy, lower_x, lower_y, upper_x, upper_y, abscissa

Boundary collision risk uses the minimum distance from a point to the lower or upper polyline
(nearest corridor edge).
"""

from __future__ import annotations

import math
import os
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_ENVELOPE_CSV = os.path.join(_SCRIPT_DIR, "derived_boundaries", "polylines_envelope.csv")
DEFAULT_HIGHWAY_BOUNDARY_CSV = os.path.join(
    _SCRIPT_DIR, "derived_highway_boundaries", "highway_boundaries.csv"
)

# (run_id, lane_kf) -> {"lower": Nx2, "upper": Nx2, "center": Nx2}
_BOUNDARIES: Dict[Tuple[int, int], Dict[str, np.ndarray]] = {}
# (run_id, lane_kf) -> {"lower": Nx2, "upper": Nx2, "center": Nx2}
_HIGHWAY_BOUNDARIES: Dict[Tuple[int, int], Dict[str, np.ndarray]] = {}
_LOADED_PATH: Optional[str] = None
_LOADED_HIGHWAY_PATH: Optional[str] = None


def load_envelope_boundaries(csv_path: Optional[str] = None) -> bool:
    """
    Load lower/upper/center polylines per (run_id, lane_kf). Returns True if file read OK.
    If file is missing, clears cache and returns False (PT runs with no boundary penalty).
    """
    global _BOUNDARIES, _LOADED_PATH
    path = csv_path or DEFAULT_ENVELOPE_CSV
    _BOUNDARIES = {}
    _LOADED_PATH = path
    if not os.path.isfile(path):
        return False
    df = pd.read_csv(path)
    need = {"run_id", "lane_kf", "point_index", "lower_x", "lower_y", "upper_x", "upper_y"}
    if not need.issubset(df.columns):
        return False
    for (rid, lk), g in df.groupby(["run_id", "lane_kf"], sort=True):
        g = g.sort_values("point_index")
        lower = np.column_stack(
            [g["lower_x"].to_numpy(dtype=np.float64), g["lower_y"].to_numpy(dtype=np.float64)]
        )
        upper = np.column_stack(
            [g["upper_x"].to_numpy(dtype=np.float64), g["upper_y"].to_numpy(dtype=np.float64)]
        )
        center = None
        if "cx" in g.columns and "cy" in g.columns:
            center = np.column_stack(
                [g["cx"].to_numpy(dtype=np.float64), g["cy"].to_numpy(dtype=np.float64)]
            )
        rid_i, lk_i = int(rid), int(lk)
        _BOUNDARIES[(rid_i, lk_i)] = {"lower": lower, "upper": upper, "center": center}
    return len(_BOUNDARIES) > 0


def load_highway_boundaries(csv_path: Optional[str] = None) -> bool:
    """
    Load run/lane lower/upper/center polylines produced by
    derive_highway_boundaries.py.
    """
    global _HIGHWAY_BOUNDARIES, _LOADED_HIGHWAY_PATH
    path = csv_path or DEFAULT_HIGHWAY_BOUNDARY_CSV
    _HIGHWAY_BOUNDARIES = {}
    _LOADED_HIGHWAY_PATH = path
    if not os.path.isfile(path):
        return False
    df = pd.read_csv(path)
    need = {"run_id", "lane_kf", "point_index", "lower_x", "lower_y", "upper_x", "upper_y"}
    if not need.issubset(df.columns):
        return False
    for (rid, lk), g in df.groupby(["run_id", "lane_kf"], sort=True):
        g = g.sort_values("point_index")
        lower = np.column_stack(
            [g["lower_x"].to_numpy(dtype=np.float64), g["lower_y"].to_numpy(dtype=np.float64)]
        )
        upper = np.column_stack(
            [g["upper_x"].to_numpy(dtype=np.float64), g["upper_y"].to_numpy(dtype=np.float64)]
        )
        center = None
        if "center_x" in g.columns and "center_y" in g.columns:
            center = np.column_stack(
                [g["center_x"].to_numpy(dtype=np.float64), g["center_y"].to_numpy(dtype=np.float64)]
            )
        _HIGHWAY_BOUNDARIES[(int(rid), int(lk))] = {
            "lower": lower,
            "upper": upper,
            "center": center,
        }
    return len(_HIGHWAY_BOUNDARIES) > 0


def _ensure_loaded() -> None:
    if not _BOUNDARIES and _LOADED_PATH is None:
        load_envelope_boundaries()
    if not _HIGHWAY_BOUNDARIES and _LOADED_HIGHWAY_PATH is None:
        load_highway_boundaries()


def _boundary_entry(
    run_id: Optional[float],
    lane_kf: Optional[float] = None,
) -> Optional[Dict[str, np.ndarray]]:
    """Prefer new run/lane highway boundary, fall back to older lane envelope."""
    _ensure_loaded()
    if run_id is None:
        return None
    try:
        rid = int(round(float(run_id)))
    except (TypeError, ValueError):
        return None
    if lane_kf is None:
        return None
    try:
        key = (rid, int(round(float(lane_kf))))
    except (TypeError, ValueError):
        return None
    entry = _HIGHWAY_BOUNDARIES.get(key)
    if entry:
        return entry
    return _BOUNDARIES.get(key)


def lane_boundary_polylines_m(run_id: Optional[float], lane_kf: Optional[float]) -> bool:
    """True if we have envelope lower/upper polylines for this (run_id, lane_kf)."""
    return _boundary_entry(run_id, lane_kf) is not None


def center_polyline(run_id: Optional[float], lane_kf: Optional[float]) -> Optional[np.ndarray]:
    """Return center polyline (N, 2) in meters for (run_id, lane_kf), if available."""
    entry = _boundary_entry(run_id, lane_kf)
    if not entry:
        return None
    center = entry.get("center")
    if center is None or len(center) < 2:
        lower, upper = entry["lower"], entry["upper"]
        if len(lower) < 2 or len(upper) < 2:
            return None
        return 0.5 * (lower + upper)
    return center


def envelope_lower_upper_polylines(
    run_id: Optional[float], lane_kf: Optional[float]
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """
    Return (lower, upper) polyline arrays (N, 2) in meters for (run_id, lane_kf) if envelope CSV is loaded.
    Arrays are shared with the internal cache (read-only for plotting).
    """
    entry = _boundary_entry(run_id, lane_kf)
    if not entry:
        return None
    return entry["lower"], entry["upper"]


def _point_to_segment_dist(px: float, py: float, ax: float, ay: float, bx: float, by: float) -> float:
    abx, aby = bx - ax, by - ay
    apx, apy = px - ax, py - ay
    denom = abx * abx + aby * aby + 1e-18
    t = max(0.0, min(1.0, (apx * abx + apy * aby) / denom))
    cx, cy = ax + t * abx, ay + t * aby
    return float(math.hypot(px - cx, py - cy))


def closest_point_on_polyline(
    px: float, py: float, poly: np.ndarray
) -> Tuple[float, float, int, float]:
    """
    Closest point on polyline to (px, py). Returns (qx, qy, segment_index, t) where t in [0,1]
    interpolates between poly[seg] and poly[seg+1].
    """
    if poly is None or len(poly) < 2:
        return float(px), float(py), 0, 0.0
    best_d = float("inf")
    best = (float(px), float(py), 0, 0.0)
    for i in range(len(poly) - 1):
        ax, ay = float(poly[i, 0]), float(poly[i, 1])
        bx, by = float(poly[i + 1, 0]), float(poly[i + 1, 1])
        abx, aby = bx - ax, by - ay
        apx, apy = px - ax, py - ay
        denom = abx * abx + aby * aby + 1e-18
        t = max(0.0, min(1.0, (apx * abx + apy * aby) / denom))
        qx, qy = ax + t * abx, ay + t * aby
        d = float(math.hypot(px - qx, py - qy))
        if d < best_d:
            best_d = d
            best = (qx, qy, i, t)
    return best  # type: ignore


def centerline_frenet_frame(
    px: float, py: float, run_id: Optional[float], lane_kf: Optional[float]
) -> Optional[Tuple[Tuple[float, float], Tuple[float, float], float, float, float]]:
    """
    Tangent / normal to the *center* polyline at the closest point (Delpiano et al. 2020, Sec. 2.3).

    Returns
        (t_hat, n_hat, qx, qy, d_lateral) or None.
        t_hat, n_hat: unit vectors. d_lateral = (p - q) · n_hat (signed).
        (qx, qy) is the closest point on the centerline.
    Or None if geometry is missing.
    """
    entry = _boundary_entry(run_id, lane_kf)
    if not entry:
        return None
    center = entry.get("center")
    lower, upper = entry["lower"], entry["upper"]
    if center is None or len(center) < 2:
        center = 0.5 * (lower + upper)
    if len(center) < 2:
        return None
    qx, qy, seg_i, t = closest_point_on_polyline(float(px), float(py), center)
    ax, ay = float(center[seg_i, 0]), float(center[seg_i, 1])
    bx, by = float(center[seg_i + 1, 0]), float(center[seg_i + 1, 1])
    tx, ty = bx - ax, by - ay
    tl = math.hypot(tx, ty)
    if tl < 1e-6:
        return None
    tx, ty = tx / tl, ty / tl
    # Lateral normal: 90° CCW from forward tangent (lateral x^+, longitudinal y^+ in paper’s local frame)
    nx, ny = -ty, tx
    pqx, pqy = float(px) - qx, float(py) - qy
    d_lat = pqx * nx + pqy * ny
    return ((tx, ty), (nx, ny), qx, qy, d_lat)


def clip_position_to_lane_envelope(
    px: float,
    py: float,
    run_id: Optional[float],
    lane_kf: Optional[float],
    margin_m: float = 0.35,
) -> Tuple[float, float, bool]:
    """
    Hard clamp: project (px, py) back into the corridor between lower and upper envelope polylines.

    Uses the closest point on the centerline (or (L+U)/2 if no center column) to pick the
    along-road segment, builds the local lower/upper chord, and clips lateral offset toward
    the chord midpoint. margin_m shrinks the feasible interval inward (meters).

    Returns (px_new, py_new, was_clipped). If no envelope for (run_id, lane_kf), returns input unchanged.
    """
    entry = _boundary_entry(run_id, lane_kf)
    if not entry:
        return px, py, False
    lower = entry["lower"]
    upper = entry["upper"]
    center = entry.get("center")
    if center is None or len(center) < 2:
        center = 0.5 * (lower + upper)
    if len(lower) < 2 or len(upper) < 2 or len(center) < 2:
        return px, py, False
    if not (len(lower) == len(upper) == len(center)):
        return px, py, False

    _, _, seg_i, t = closest_point_on_polyline(px, py, center)
    t = max(0.0, min(1.0, t))
    i = max(0, min(seg_i, len(lower) - 2))
    L0, L1 = lower[i], lower[i + 1]
    U0, U1 = upper[i], upper[i + 1]
    lbx = (1.0 - t) * L0[0] + t * L1[0]
    lby = (1.0 - t) * L0[1] + t * L1[1]
    ubx = (1.0 - t) * U0[0] + t * U1[0]
    uby = (1.0 - t) * U0[1] + t * U1[1]
    chord_x, chord_y = ubx - lbx, uby - lby
    chord_len = float(math.hypot(chord_x, chord_y))
    if chord_len < 0.15:
        return px, py, False
    half_w = 0.5 * chord_len
    nx, ny = chord_x / (chord_len + 1e-18), chord_y / (chord_len + 1e-18)
    cx_mid = 0.5 * (lbx + ubx)
    cy_mid = 0.5 * (lby + uby)
    s = (px - cx_mid) * nx + (py - cy_mid) * ny
    smax = half_w - margin_m
    smin = -half_w + margin_m
    if smax < smin:
        smin = smax = 0.0
    s2 = max(smin, min(smax, s))
    pnx = cx_mid + s2 * nx
    pny = cy_mid + s2 * ny
    was = (abs(s2 - s) > 1e-4) or (math.hypot(pnx - px, pny - py) > 1e-3)
    return pnx, pny, was


def min_distance_point_to_polyline(px: float, py: float, poly: np.ndarray) -> float:
    """Minimum Euclidean distance from (px,py) to polyline (segment chain)."""
    if poly is None or len(poly) < 2:
        return float("inf")
    dmin = float("inf")
    for i in range(len(poly) - 1):
        d = _point_to_segment_dist(
            px, py,
            float(poly[i, 0]), float(poly[i, 1]),
            float(poly[i + 1, 0]), float(poly[i + 1, 1]),
        )
        if d < dmin:
            dmin = d
    return dmin


def max_boundary_collision_prob(
    px: float,
    py: float,
    run_id: Optional[float],
    lane_kf: Optional[float],
    threshold_m: float,
    sigma: float,
) -> float:
    """
    Soft collision probability vs corridor: clearance = min(dist to lower, dist to upper polyline).
    Gaussian tail: high p when clearance < threshold_m.
    """
    entry = _boundary_entry(run_id, lane_kf)
    if not entry:
        return 0.0
    lower = entry["lower"]
    upper = entry["upper"]
    d_lo = min_distance_point_to_polyline(px, py, lower)
    d_hi = min_distance_point_to_polyline(px, py, upper)
    d_clear = min(d_lo, d_hi)
    if sigma <= 1e-12:
        sigma = 1e-6
    z = (d_clear - threshold_m) / sigma
    return 0.5 * (1.0 - math.erf(z / math.sqrt(2.0)))
