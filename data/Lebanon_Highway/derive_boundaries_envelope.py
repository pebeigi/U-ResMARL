# -*- coding: utf-8 -*-
"""
Envelope boundaries from trajectories (NEW method — does not replace fixed half-width).

Coordinates are the same world frame as xloc_kf / yloc_kf (meters). The data centroid is used only
to define the PCA axis; centerline vertices are median x,y per bin — not anchored at (0,0).

Single lane: by default, points farther than --max-lateral from the iterative smoothed centerline
are dropped so adjacent-lane or junk outliers do not widen the envelope.

Boundaries: bin along **PCA along-road coordinate s** (not raw x/y). Lateral offsets use the
global normal **t ⟂ u**. Half-open s-bins avoid double-counting edges; sparse bins get **limited**
gap interpolation only (no long artificial bridges). Stricter minimum counts stabilize percentiles.
Smoothing uses a small window on **both** x and y. Highway **start/end** = low/high s tails. Optional
single-lane filter uses distance to the binned centerline.

Outputs:
  - ./derived_boundaries/figures_envelope/run_{rid}_lane_{lk}.png
  - ./derived_boundaries/polylines_envelope.csv (abscissa = pca_s)
  - ./derived_boundaries/envelope_lane_meta.csv (PCA start/end, axis, per lane)
  - With --diagnostics: ./derived_boundaries/diagnostics_envelope/run_{rid}_lane_{lk}.png
    and centerline_diagnostics_summary.csv (per-lane counts / path length).

Run: python derive_boundaries_envelope.py
     python derive_boundaries_envelope.py --no-single-lane-filter
     python derive_boundaries_envelope.py --max-lateral 3.5 --buffer 1.0 --max-gap-bins 2 --smooth-window 5
     python derive_boundaries_envelope.py --diagnostics
"""

from __future__ import annotations

import argparse
import os
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CSV = os.path.join(_SCRIPT_DIR, "Final_Lebanon_Data.csv")
OUT_DIR = os.path.join(_SCRIPT_DIR, "derived_boundaries")

BIN_WIDTH_M = 8.0
CAR_CLASS = 1.0
MIN_POINTS_PER_BIN = 8
MIN_POINTS_PER_BIN_END = 3
N_END_BINS_RELAXED = 3
SMOOTH_WINDOW = 5
MAX_INTERP_GAP_BINS = 2
MIN_PERCENTILE_COUNT_INNER = 20
MIN_PERCENTILE_COUNT_END = 10
MAX_LATERAL_FROM_CENTERLINE_M = 4.0
CENTERLINE_FILTER_ITERS = 3
MIN_POINTS_AFTER_FILTER = 150


def pca_first_axis(xy: np.ndarray) -> np.ndarray:
    c = xy.mean(axis=0)
    xc = xy - c
    if len(xc) < 3:
        return np.array([1.0, 0.0])
    _, _, vt = np.linalg.svd(xc, full_matrices=False)
    u = vt[0].astype(np.float64)
    n = np.linalg.norm(u)
    if n < 1e-12:
        return np.array([1.0, 0.0])
    return u / n


def dist_points_to_segment(
    x: np.ndarray,
    y: np.ndarray,
    ax: float,
    ay: float,
    bx: float,
    by: float,
) -> np.ndarray:
    abx = bx - ax
    aby = by - ay
    apx = x - ax
    apy = y - ay
    denom = abx * abx + aby * aby + 1e-18
    t = np.clip((apx * abx + apy * aby) / denom, 0.0, 1.0)
    cx = ax + t * abx
    cy = ay + t * aby
    return np.hypot(x - cx, y - cy)


def min_distance_points_to_polyline(x: np.ndarray, y: np.ndarray, C: np.ndarray) -> np.ndarray:
    n = len(x)
    dmin = np.full(n, np.inf, dtype=np.float64)
    if C is None or len(C) < 2:
        return dmin
    for i in range(len(C) - 1):
        d = dist_points_to_segment(
            x, y, float(C[i, 0]), float(C[i, 1]), float(C[i + 1, 0]), float(C[i + 1, 1])
        )
        dmin = np.minimum(dmin, d)
    return dmin


def highway_start_end_pca(
    x: np.ndarray,
    y: np.ndarray,
    tip_frac: float = 0.02,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Mean (x,y) in low / high PCA-s quantiles; returns (start, end, unit tangent u)."""
    xy = np.column_stack([x, y])
    c = xy.mean(axis=0)
    u = pca_first_axis(xy)
    s = (xy - c) @ u
    lo = float(np.quantile(s, tip_frac))
    hi = float(np.quantile(s, 1.0 - tip_frac))
    m0 = s <= lo
    m1 = s >= hi
    start = np.array([float(x[m0].mean()), float(y[m0].mean())], dtype=np.float64)
    end = np.array([float(x[m1].mean()), float(y[m1].mean())], dtype=np.float64)
    return start, end, u


def lateral_unit(u: np.ndarray) -> np.ndarray:
    v = np.array([-float(u[1]), float(u[0])], dtype=np.float64)
    return v / (np.linalg.norm(v) + 1e-18)


def _min_count_for_bin(i: int, n_bins: int, min_count: int) -> int:
    if i < N_END_BINS_RELAXED or i >= n_bins - N_END_BINS_RELAXED:
        return MIN_POINTS_PER_BIN_END
    return min_count


def _percentile_min_count(i: int, n_bins: int) -> int:
    """Stable percentiles need enough samples (stricter in interior bins)."""
    if i < N_END_BINS_RELAXED or i >= n_bins - N_END_BINS_RELAXED:
        return max(MIN_POINTS_PER_BIN_END, MIN_PERCENTILE_COUNT_END)
    return max(MIN_POINTS_PER_BIN, MIN_PERCENTILE_COUNT_INNER)


def _bin_mask_halfopen(v: np.ndarray, lo: float, hi: float, i_bin: int, n_bins: int) -> np.ndarray:
    if i_bin == n_bins - 1:
        return (v >= lo) & (v <= hi)
    return (v >= lo) & (v < hi)


def fill_nan_linear_limited(v: np.ndarray, max_gap_bins: int) -> np.ndarray:
    """Interpolate NaNs only across short index gaps (avoids fake bridges over large sparse runs)."""
    v = np.asarray(v, dtype=np.float64).copy()
    n = len(v)
    idx = np.arange(n, dtype=np.float64)
    isnan = ~np.isfinite(v)
    if not np.any(isnan):
        return v
    good = np.where(~isnan)[0]
    if len(good) == 0:
        return v
    for i in range(len(good) - 1):
        start, end = int(good[i]), int(good[i + 1])
        gap = end - start - 1
        if gap > 0 and gap <= max_gap_bins:
            v[start : end + 1] = np.interp(
                idx[start : end + 1],
                np.array([float(start), float(end)], dtype=np.float64),
                np.array([v[start], v[end]], dtype=np.float64),
            )
    return v


def compute_pca_s_binned_envelope(
    x: np.ndarray,
    y: np.ndarray,
    bin_width_m: float,
    min_count: int,
    buffer_m: float,
    p_low: float,
    p_high: float,
    max_gap_bins: int,
) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """
    Bin along PCA along-road coordinate s; lateral offsets along t = (-u_y, u_x).
    Half-open s-intervals; limited-gap fill only; drop bins that stay NaN after fill.
    """
    xy = np.column_stack([x, y])
    c = xy.mean(axis=0)
    u = pca_first_axis(xy)
    v_perp = lateral_unit(u)
    s = (xy - c) @ u

    s_min, s_max = float(s.min()), float(s.max())
    span = s_max - s_min
    if span < bin_width_m * 2:
        return None

    n_bins = max(5, int(np.ceil(span / bin_width_m)))
    edges = np.linspace(s_min, s_max, n_bins + 1)

    cx = np.full(n_bins, np.nan, dtype=np.float64)
    cy = np.full(n_bins, np.nan, dtype=np.float64)
    lx = np.full(n_bins, np.nan, dtype=np.float64)
    ly = np.full(n_bins, np.nan, dtype=np.float64)
    ux = np.full(n_bins, np.nan, dtype=np.float64)
    uy = np.full(n_bins, np.nan, dtype=np.float64)

    for i in range(n_bins):
        lo, hi = float(edges[i]), float(edges[i + 1])
        mask = _bin_mask_halfopen(s, lo, hi, i, n_bins)
        cnt = int(mask.sum())
        mc = _min_count_for_bin(i, n_bins, min_count)
        pmc = _percentile_min_count(i, n_bins)
        if cnt < mc or cnt < pmc:
            continue

        xm = float(np.median(x[mask]))
        ym = float(np.median(y[mask]))
        center = np.array([xm, ym], dtype=np.float64)
        d_loc = (xy[mask] - center) @ v_perp
        d_lo = float(np.percentile(d_loc, p_low)) - buffer_m
        d_hi = float(np.percentile(d_loc, p_high)) + buffer_m
        if d_lo > d_hi:
            d_lo, d_hi = d_hi, d_lo
        lo_pt = center + d_lo * v_perp
        hi_pt = center + d_hi * v_perp
        cx[i] = xm
        cy[i] = ym
        lx[i], ly[i] = float(lo_pt[0]), float(lo_pt[1])
        ux[i], uy[i] = float(hi_pt[0]), float(hi_pt[1])

    if not np.any(np.isfinite(cx)):
        return None

    cx = fill_nan_linear_limited(cx, max_gap_bins)
    cy = fill_nan_linear_limited(cy, max_gap_bins)
    lx = fill_nan_linear_limited(lx, max_gap_bins)
    ly = fill_nan_linear_limited(ly, max_gap_bins)
    ux = fill_nan_linear_limited(ux, max_gap_bins)
    uy = fill_nan_linear_limited(uy, max_gap_bins)

    ok = (
        np.isfinite(cx)
        & np.isfinite(cy)
        & np.isfinite(lx)
        & np.isfinite(ly)
        & np.isfinite(ux)
        & np.isfinite(uy)
    )
    if int(ok.sum()) < 3:
        return None
    center = np.column_stack([cx[ok], cy[ok]])
    lower = np.column_stack([lx[ok], ly[ok]])
    upper = np.column_stack([ux[ok], uy[ok]])
    return center, lower, upper


def centerline_from_bins_pca_s(
    x: np.ndarray,
    y: np.ndarray,
    bin_width_m: float,
    min_count: int,
    max_gap_bins: int,
) -> Optional[np.ndarray]:
    """Provisional centerline for inlier filter (median per s-bin; p50 lateral ~ center)."""
    tri = compute_pca_s_binned_envelope(
        x, y, bin_width_m, min_count, 0.0, 50.0, 50.0, max_gap_bins
    )
    if tri is None:
        return None
    return tri[0]


def smooth_polyline_xy(C: np.ndarray, window: int) -> np.ndarray:
    """Moving average on x and y (same kernel); preserves endpoints."""
    if C is None or len(C) < 3:
        return C
    w = max(3, window | 1)
    if w > len(C):
        w = len(C) if len(C) % 2 == 1 else len(C) - 1
        if w < 3:
            return C
    ker = np.ones(w, dtype=np.float64) / w
    out = np.column_stack(
        [
            np.convolve(C[:, 0], ker, mode="same"),
            np.convolve(C[:, 1], ker, mode="same"),
        ]
    )
    if len(out) >= 2:
        out[0] = C[0]
        out[-1] = C[-1]
    return out


def cumulative_arc_length_along_polyline(C: np.ndarray) -> np.ndarray:
    """Cumulative geodesic distance along vertex chain from C[0]; shape (n,)."""
    n = len(C)
    if n == 0:
        return np.array([], dtype=np.float64)
    if n == 1:
        return np.zeros(1, dtype=np.float64)
    d = np.hypot(np.diff(C[:, 0]), np.diff(C[:, 1]))
    s = np.zeros(n, dtype=np.float64)
    s[1:] = np.cumsum(d)
    return s


def edge_steps_dx_ds(C: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Per edge i: from C[i] to C[i+1]. Returns (ds, dx, dy, dx_ds), each length n-1.
    dx_ds = NaN where ds is tiny.
    """
    dx = np.diff(C[:, 0])
    dy = np.diff(C[:, 1])
    ds = np.hypot(dx, dy)
    with np.errstate(divide="ignore", invalid="ignore"):
        dx_ds = np.where(ds > 1e-12, dx / ds, np.nan)
    return ds, dx, dy, dx_ds


def count_sign_changes_1d(
    values: np.ndarray,
    min_abs: float,
) -> Tuple[int, List[int]]:
    """
    Walk consecutive finite values; count when sign flips and |value| >= min_abs.
    Returns (count, edge_start_indices) where each index i means flip between edge i and i+1.
    """
    flips: List[int] = []
    if values.size < 2:
        return 0, flips
    prev_sign: Optional[int] = None
    prev_i: Optional[int] = None
    for i in range(len(values)):
        v = float(values[i])
        if not np.isfinite(v) or abs(v) < min_abs:
            continue
        sgn = 1 if v > 0 else -1
        if prev_sign is not None and prev_i is not None and sgn != prev_sign:
            flips.append(prev_i)
        prev_sign = sgn
        prev_i = i
    return len(flips), flips


def centerline_geometry_diagnostics(C: np.ndarray) -> Dict[str, object]:
    """Arc-length stats, dx / dx-ds sign flips, non-adjacent near-duplicate x along chain."""
    empty: List[int] = []
    if C is None or len(C) < 2:
        return {
            "n_vertices": int(len(C)) if C is not None else 0,
            "path_length_m": 0.0,
            "n_dx_sign_changes": 0,
            "n_dxds_sign_changes": 0,
            "dx_flip_edge_starts": list(empty),
            "max_step_m": 0.0,
            "mean_step_m": 0.0,
            "n_nonadj_close_x_pairs": 0,
        }
    s_along = cumulative_arc_length_along_polyline(C)
    ds, dx, _dy, dx_ds = edge_steps_dx_ds(C)
    path_len = float(s_along[-1])
    n_dx, dx_flips = count_sign_changes_1d(dx, DX_SIGN_EPS_M)
    n_dxds, _ = count_sign_changes_1d(dx_ds, DXDS_SIGN_EPS)
    max_step = float(np.max(ds)) if len(ds) else 0.0
    mean_step = float(np.mean(ds)) if len(ds) else 0.0
    n_pairs = count_nonadj_vertex_pairs_close_x(C, x_tol_m=1.0)
    return {
        "n_vertices": int(len(C)),
        "path_length_m": path_len,
        "n_dx_sign_changes": n_dx,
        "n_dxds_sign_changes": n_dxds,
        "dx_flip_edge_starts": dx_flips,
        "max_step_m": max_step,
        "mean_step_m": mean_step,
        "n_nonadj_close_x_pairs": n_pairs,
    }


def count_nonadj_vertex_pairs_close_x(C: np.ndarray, x_tol_m: float = 1.0) -> int:
    """Pairs (i, j) with j >= i+2 and abs(x_i - x_j) <= tol along the vertex chain."""
    n = len(C)
    if n < 4:
        return 0
    if n > 500:
        return -1
    xs = C[:, 0]
    c = 0
    for i in range(n):
        xi = xs[i]
        for j in range(i + 2, n):
            if abs(xi - xs[j]) <= x_tol_m:
                c += 1
    return c


def plot_centerline_diagnostics(
    C: np.ndarray,
    run_id: int,
    lane_kf: int,
    out_path: str,
    stats: Dict[str, object],
) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))
    s_v = cumulative_arc_length_along_polyline(C)
    _ds, _dx, _dy, dx_ds = edge_steps_dx_ds(C)
    s_edge = 0.5 * (s_v[:-1] + s_v[1:])

    axes[0].plot(s_v, C[:, 0], "o-", ms=3, lw=1.2, color="C0")
    axes[0].set_xlabel("Arc length along centerline (m)")
    axes[0].set_ylabel("x (m)")
    axes[0].set_title("x(s): single-valued along path")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(s_v, C[:, 1], "o-", ms=3, lw=1.2, color="C1")
    axes[1].set_xlabel("Arc length along centerline (m)")
    axes[1].set_ylabel("y (m)")
    axes[1].set_title("y(s): single-valued along path")
    axes[1].grid(True, alpha=0.3)

    valid = np.isfinite(dx_ds)
    axes[2].plot(s_edge[valid], dx_ds[valid], "o-", ms=3, lw=1.2, color="C2")
    axes[2].axhline(0.0, color="0.5", lw=0.9)
    axes[2].set_xlabel("s at edge midpoint (m)")
    axes[2].set_ylabel("dx/ds per edge")
    axes[2].set_title("Eastward component along path")
    axes[2].set_ylim(-1.05, 1.05)
    axes[2].grid(True, alpha=0.3)

    ndx = stats.get("n_dx_sign_changes", 0)
    ndxd = stats.get("n_dxds_sign_changes", 0)
    pairs = stats.get("n_nonadj_close_x_pairs", 0)
    pairs_note = f"{pairs}" if pairs >= 0 else "skipped (n>500)"
    fig.suptitle(
        f"Centerline diagnostics run_id={run_id} lane_kf={lane_kf}\n"
        f"path={stats.get('path_length_m', 0):.1f} m, vertices={stats.get('n_vertices', 0)}, "
        f"dx sign flips={ndx}, dx/ds flips={ndxd}, "
        f"non-adj. |dx|<=1 m pairs={pairs_note}",
        fontsize=11,
    )
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def iterative_single_lane_xy_filter(
    x: np.ndarray,
    y: np.ndarray,
    bin_width_m: float,
    min_count: int,
    smooth_window: int,
    max_lateral_m: float,
    max_iters: int,
    max_gap_bins: int,
) -> Tuple[np.ndarray, np.ndarray, str]:
    x_orig = np.asarray(x, dtype=np.float64).copy()
    y_orig = np.asarray(y, dtype=np.float64).copy()
    x = x_orig.copy()
    y = y_orig.copy()
    n0 = len(x)
    if n0 < MIN_POINTS_AFTER_FILTER or max_iters < 1 or max_lateral_m <= 0:
        return x, y, "filter skipped"
    msg_parts = [f"raw n={n0}"]
    for it in range(max_iters):
        C0 = centerline_from_bins_pca_s(
            x, y, bin_width_m=bin_width_m, min_count=min_count, max_gap_bins=max_gap_bins
        )
        if C0 is None or len(C0) < 2:
            msg_parts.append("no centerline; stop filter")
            break
        C = smooth_polyline_xy(C0, smooth_window)
        if C is None or len(C) < 2:
            msg_parts.append("smooth failed; stop filter")
            break
        d = min_distance_points_to_polyline(x, y, C)
        mask = d <= max_lateral_m
        n_kept = int(mask.sum())
        msg_parts.append(f"iter{it + 1}: keep {n_kept}/{len(x)} (<= {max_lateral_m} m)")
        if n_kept < MIN_POINTS_AFTER_FILTER:
            msg_parts.append("too few after filter; revert to raw")
            return x_orig, y_orig, "; ".join(msg_parts)
        if n_kept == len(x):
            break
        x, y = x[mask], y[mask]
    return x, y, "; ".join(msg_parts)


FIG_DIR_ENVELOPE = os.path.join(OUT_DIR, "figures_envelope")
DIAG_DIR_ENVELOPE = os.path.join(OUT_DIR, "diagnostics_envelope")
DIAG_SUMMARY_CSV = os.path.join(OUT_DIR, "centerline_diagnostics_summary.csv")
POLY_CSV_ENVELOPE = os.path.join(OUT_DIR, "polylines_envelope.csv")
META_CSV_ENVELOPE = os.path.join(OUT_DIR, "envelope_lane_meta.csv")

# Ignore sub-centimeter edge steps when counting sign changes (noise after smoothing).
DX_SIGN_EPS_M = 0.05
DXDS_SIGN_EPS = 0.05

# Lateral extent from data + margin beyond observed spread
ENVELOPE_BUFFER_M = 1.0
P_LOW = 2.0
P_HIGH = 98.0


def plot_envelope_combo(
    traj_x: np.ndarray,
    traj_y: np.ndarray,
    center: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    run_id: int,
    lane_kf: int,
    out_path: str,
    bin_width_m: float,
    smooth_window: int,
    buffer_m: float,
    p_low: float,
    p_high: float,
    n_inliers: int,
    filter_note: str,
    envelope_mode: str,
    hwy_start: np.ndarray,
    hwy_end: np.ndarray,
) -> None:
    fig, ax = plt.subplots(figsize=(11, 10))
    ax.scatter(
        traj_x,
        traj_y,
        s=1,
        alpha=0.12,
        c="0.35",
        linewidths=0,
        label=f"Trajectory (all {len(traj_x):,} points)",
        rasterized=True,
        zorder=1,
    )
    ax.scatter(
        lower[:, 0],
        lower[:, 1],
        s=28,
        c="C2",
        marker="o",
        edgecolors="black",
        linewidths=0.5,
        label=f"Lower bound (p{p_low:g} - {buffer_m} m)",
        zorder=3,
    )
    ax.scatter(
        upper[:, 0],
        upper[:, 1],
        s=28,
        c="C3",
        marker="o",
        edgecolors="black",
        linewidths=0.5,
        label=f"Upper bound (p{p_high:g} + {buffer_m} m)",
        zorder=3,
    )
    ax.scatter(
        center[:, 0],
        center[:, 1],
        s=36,
        c="C1",
        marker="o",
        edgecolors="black",
        linewidths=0.6,
        label="Centerline",
        zorder=4,
    )
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ds = hwy_end - hwy_start
    ax.scatter(
        [hwy_start[0]],
        [hwy_start[1]],
        s=120,
        c="lime",
        edgecolors="black",
        linewidths=0.8,
        zorder=6,
        label="PCA start (low s)",
    )
    ax.scatter(
        [hwy_end[0]],
        [hwy_end[1]],
        s=120,
        c="red",
        edgecolors="black",
        linewidths=0.8,
        zorder=6,
        label="PCA end (high s)",
    )
    ax.set_title(
        f"run_id={run_id}, lane_kf={lane_kf} — {envelope_mode} (PCA s bins, lateral along t)\n"
        f"Highway PCA start ({hwy_start[0]:.1f},{hwy_start[1]:.1f}) -> end ({hwy_end[0]:.1f},{hwy_end[1]:.1f}), "
        f"Delta=({ds[0]:.1f},{ds[1]:.1f}) m\n"
        f"{filter_note} (envelope fit: {n_inliers:,} inlier points)\n"
        f"p{p_low:g}/p{p_high:g} lateral, {bin_width_m} m bins along s, smooth w={smooth_window}"
    )
    ax.legend(loc="best", fontsize=9)
    ax.grid(True, alpha=0.25)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def polylines_envelope_to_rows(
    run_id: int,
    lane_kf: int,
    center: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    envelope_mode: str,
) -> List[dict]:
    rows = []
    for i in range(len(center)):
        rows.append(
            {
                "run_id": run_id,
                "lane_kf": lane_kf,
                "abscissa": envelope_mode,
                "point_index": i,
                "cx": center[i, 0],
                "cy": center[i, 1],
                "lower_x": lower[i, 0],
                "lower_y": lower[i, 1],
                "upper_x": upper[i, 0],
                "upper_y": upper[i, 1],
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Envelope boundaries from lateral spread + buffer.")
    parser.add_argument("--csv", type=str, default=DEFAULT_CSV, help="Input trajectory CSV")
    parser.add_argument("--bin-width", type=float, default=BIN_WIDTH_M, help="Along-road bin width (m)")
    parser.add_argument("--buffer", type=float, default=ENVELOPE_BUFFER_M, help="Meters beyond p_low/p_high lateral")
    parser.add_argument("--p-low", type=float, default=P_LOW, help="Lower lateral percentile (0–100)")
    parser.add_argument("--p-high", type=float, default=P_HIGH, help="Upper lateral percentile (0–100)")
    parser.add_argument(
        "--no-single-lane-filter",
        action="store_true",
        help="Disable centerline-distance outlier filter",
    )
    parser.add_argument(
        "--max-lateral",
        type=float,
        default=MAX_LATERAL_FROM_CENTERLINE_M,
        help="Single-lane gate: max distance (m) to provisional centerline",
    )
    parser.add_argument(
        "--filter-iters",
        type=int,
        default=CENTERLINE_FILTER_ITERS,
        help="Iterations of fit + inlier mask",
    )
    parser.add_argument(
        "--diagnostics",
        action="store_true",
        help="Write per-lane diagnostic figures (x(s), y(s), dx/ds) and summary CSV",
    )
    parser.add_argument(
        "--diagnostics-dir",
        type=str,
        default="",
        help="Directory for diagnostic PNGs (default: derived_boundaries/diagnostics_envelope)",
    )
    parser.add_argument(
        "--max-gap-bins",
        type=int,
        default=MAX_INTERP_GAP_BINS,
        help="Max consecutive empty bins to fill by interpolation (larger gaps stay dropped)",
    )
    parser.add_argument(
        "--smooth-window",
        type=int,
        default=SMOOTH_WINDOW,
        help="Moving-average window for x,y (odd; endpoints fixed)",
    )
    args = parser.parse_args()
    bin_w = float(args.bin_width)
    buf = float(args.buffer)
    p_lo = float(args.p_low)
    p_hi = float(args.p_high)
    max_gap = int(args.max_gap_bins)
    sm_w = int(args.smooth_window)

    csv_path = args.csv
    if not os.path.isfile(csv_path):
        raise FileNotFoundError(csv_path)

    os.makedirs(FIG_DIR_ENVELOPE, exist_ok=True)
    diag_dir = os.path.normpath(args.diagnostics_dir) if args.diagnostics_dir else DIAG_DIR_ENVELOPE
    if args.diagnostics:
        os.makedirs(diag_dir, exist_ok=True)

    print("Loading:", csv_path)
    df = pd.read_csv(
        csv_path,
        usecols=["id", "time", "xloc_kf", "yloc_kf", "lane_kf", "run_id", "class"],
    )
    df = df[df["class"] == CAR_CLASS].copy()
    for c in ("lane_kf", "run_id", "xloc_kf", "yloc_kf", "time"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["lane_kf", "run_id", "xloc_kf", "yloc_kf"])
    df = df.sort_values(["id", "time"], kind="mergesort")

    all_rows: List[dict] = []
    diag_summary_rows: List[dict] = []
    meta_rows: List[dict] = []
    combos = (
        df.groupby(["run_id", "lane_kf"], sort=True)
        .size()
        .reset_index()[["run_id", "lane_kf"]]
        .values.tolist()
    )

    for run_id, lane_kf in combos:
        run_id = int(run_id)
        lane_kf = int(lane_kf)
        sub = df[(df["run_id"] == run_id) & (df["lane_kf"] == lane_kf)]
        x_raw = sub["xloc_kf"].to_numpy(dtype=np.float64)
        y_raw = sub["yloc_kf"].to_numpy(dtype=np.float64)
        n_raw = len(x_raw)
        print(f"run_id={run_id} lane_kf={lane_kf}: n_points={n_raw:,} (raw, time-sorted per id)")
        _s0, _e0, u0 = highway_start_end_pca(x_raw, y_raw)
        print(
            f"  PCA axis u=({u0[0]:.3f},{u0[1]:.3f}); envelope bins along PCA s (not raw x/y)"
        )
        if args.no_single_lane_filter:
            xf, yf = x_raw, y_raw
            flt_msg = "single-lane filter off"
        else:
            xf, yf, flt_msg = iterative_single_lane_xy_filter(
                x_raw,
                y_raw,
                bin_w,
                MIN_POINTS_PER_BIN,
                sm_w,
                float(args.max_lateral),
                int(args.filter_iters),
                max_gap,
            )
        n_inliers = len(xf)
        print(f"  {flt_msg}")
        cx_m = float(np.mean(xf))
        cy_m = float(np.mean(yf))
        print(
            f"  mean(x), mean(y) after filter = ({cx_m:.2f}, {cy_m:.2f}) m "
            "(centroid for PCA only; geometry in absolute track coords)"
        )
        x, y = xf, yf

        hwy_start, hwy_end, u_fit = highway_start_end_pca(xf, yf)
        print(
            f"  highway PCA start=({hwy_start[0]:.1f},{hwy_start[1]:.1f}) "
            f"end=({hwy_end[0]:.1f},{hwy_end[1]:.1f}) m (low/high s tails, inliers)"
        )

        triple = compute_pca_s_binned_envelope(
            x, y, bin_w, MIN_POINTS_PER_BIN, buf, p_lo, p_hi, max_gap
        )
        if triple is None:
            print("  skip: not enough structure")
            continue
        C0, L0, U0 = triple
        C = smooth_polyline_xy(C0, sm_w)
        L = smooth_polyline_xy(L0, sm_w)
        U = smooth_polyline_xy(U0, sm_w)

        tag_r = str(run_id).replace(".", "_")
        tag_l = str(lane_kf).replace(".", "_")

        if args.diagnostics and C is not None and len(C) >= 2:
            dstats = centerline_geometry_diagnostics(C)
            diag_path = os.path.join(diag_dir, f"run_{tag_r}_lane_{tag_l}.png")
            plot_centerline_diagnostics(C, run_id, lane_kf, diag_path, dstats)
            print(
                "  diagnostics:",
                f"path={dstats['path_length_m']:.1f} m",
                f"vertices={dstats['n_vertices']}",
                f"dx_sign_flips={dstats['n_dx_sign_changes']}",
                f"dx/ds_flips={dstats['n_dxds_sign_changes']}",
                f"nonadj_|dx|<=1m_pairs={dstats['n_nonadj_close_x_pairs']}",
            )
            print("  diagnostic figure:", diag_path)
            diag_summary_rows.append(
                {
                    "run_id": run_id,
                    "lane_kf": lane_kf,
                    "n_vertices": dstats["n_vertices"],
                    "path_length_m": dstats["path_length_m"],
                    "n_dx_sign_changes": dstats["n_dx_sign_changes"],
                    "n_dxds_sign_changes": dstats["n_dxds_sign_changes"],
                    "max_step_m": dstats["max_step_m"],
                    "mean_step_m": dstats["mean_step_m"],
                    "n_nonadj_close_x_pairs": dstats["n_nonadj_close_x_pairs"],
                }
            )
        fig_path = os.path.join(FIG_DIR_ENVELOPE, f"run_{tag_r}_lane_{tag_l}.png")
        plot_envelope_combo(
            x_raw,
            y_raw,
            C,
            L,
            U,
            run_id,
            lane_kf,
            fig_path,
            bin_w,
            sm_w,
            buf,
            p_lo,
            p_hi,
            n_inliers,
            flt_msg,
            "pca_s",
            hwy_start,
            hwy_end,
        )
        print("  saved:", fig_path)
        all_rows.extend(polylines_envelope_to_rows(run_id, lane_kf, C, L, U, "pca_s"))
        meta_rows.append(
            {
                "run_id": run_id,
                "lane_kf": lane_kf,
                "abscissa": "pca_s",
                "pca_u_x": float(u_fit[0]),
                "pca_u_y": float(u_fit[1]),
                "hwy_start_x": float(hwy_start[0]),
                "hwy_start_y": float(hwy_start[1]),
                "hwy_end_x": float(hwy_end[0]),
                "hwy_end_y": float(hwy_end[1]),
                "delta_end_start_x": float(hwy_end[0] - hwy_start[0]),
                "delta_end_start_y": float(hwy_end[1] - hwy_start[1]),
                "n_vertices": len(C),
            }
        )

    if all_rows:
        pd.DataFrame(all_rows).to_csv(POLY_CSV_ENVELOPE, index=False)
        print("\nWrote:", POLY_CSV_ENVELOPE, f"({len(all_rows)} points)")
    if meta_rows:
        pd.DataFrame(meta_rows).to_csv(META_CSV_ENVELOPE, index=False)
        print("Wrote:", META_CSV_ENVELOPE, f"({len(meta_rows)} lanes)")
    if diag_summary_rows:
        pd.DataFrame(diag_summary_rows).to_csv(DIAG_SUMMARY_CSV, index=False)
        print("Wrote:", DIAG_SUMMARY_CSV, f"({len(diag_summary_rows)} lanes)")
    print("Done.")


if __name__ == "__main__":
    main()
