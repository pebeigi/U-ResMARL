#!/usr/bin/env python
"""Derive highway boundaries from trajectory data per (run_id, lane_kf).

Here lane_kf is treated as the direction/corridor identifier within each run.
With 2 run_id values and 2 lane_kf values, this produces 4 boundary envelopes.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DEFAULT_CSV = Path(__file__).with_name("Final_Lebanon_Data.csv")
DEFAULT_OUT_DIR = Path(__file__).with_name("derived_highway_boundaries")


def pca_axis(xy: np.ndarray) -> np.ndarray:
    centered = xy - xy.mean(axis=0)
    if len(centered) < 3:
        return np.array([1.0, 0.0], dtype=float)
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    axis = vt[0].astype(float)
    return axis / max(np.linalg.norm(axis), 1e-12)


def rolling_median(values: np.ndarray, window: int) -> np.ndarray:
    if window <= 1:
        return values
    window = max(3, window | 1)
    series = pd.Series(values)
    return (
        series.rolling(window=window, center=True, min_periods=1)
        .median()
        .to_numpy(dtype=float)
    )


def derive_boundary(
    group: pd.DataFrame,
    bin_width: float,
    p_low: float,
    p_high: float,
    buffer_m: float,
    min_count: int,
    smooth_window: int,
) -> tuple[pd.DataFrame, dict[str, float | int]]:
    xy = group[["xloc_kf", "yloc_kf"]].to_numpy(float)
    origin = xy.mean(axis=0)
    tangent = pca_axis(xy)

    # Orient the tangent in the dominant observed travel direction.
    g = group.sort_values(["id", "time"])
    dx = g.groupby("id")["xloc_kf"].diff().to_numpy(float)
    dy = g.groupby("id")["yloc_kf"].diff().to_numpy(float)
    direction_score = np.nanmedian(dx * tangent[0] + dy * tangent[1])
    if np.isfinite(direction_score) and direction_score < 0:
        tangent = -tangent

    normal = np.array([-tangent[1], tangent[0]], dtype=float)
    rel = xy - origin
    s = rel @ tangent
    l = rel @ normal
    s_lo, s_hi = np.quantile(s, [0.005, 0.995])
    edges = np.arange(s_lo, s_hi + bin_width, bin_width)
    rows = []
    for i in range(len(edges) - 1):
        lo, hi = float(edges[i]), float(edges[i + 1])
        mask = (s >= lo) & (s < hi if i < len(edges) - 2 else s <= hi)
        if int(mask.sum()) < min_count:
            continue
        s_mid = 0.5 * (lo + hi)
        l_low = float(np.percentile(l[mask], p_low)) - buffer_m
        l_high = float(np.percentile(l[mask], p_high)) + buffer_m
        center_l = float(np.median(l[mask]))
        rows.append(
            {
                "point_index": len(rows),
                "s": s_mid,
                "center_l": center_l,
                "lower_l": min(l_low, l_high),
                "upper_l": max(l_low, l_high),
                "n_points": int(mask.sum()),
            }
        )
    if len(rows) < 3:
        raise RuntimeError("Not enough populated bins to derive boundary")

    out = pd.DataFrame(rows)
    for col in ("center_l", "lower_l", "upper_l"):
        out[col] = rolling_median(out[col].to_numpy(float), smooth_window)

    for label, lateral_col in (
        ("center", "center_l"),
        ("lower", "lower_l"),
        ("upper", "upper_l"),
    ):
        pts = origin + np.outer(out["s"].to_numpy(float), tangent) + np.outer(
            out[lateral_col].to_numpy(float), normal
        )
        out[f"{label}_x"] = pts[:, 0]
        out[f"{label}_y"] = pts[:, 1]

    meta = {
        "n_points": int(len(group)),
        "n_vertices": int(len(out)),
        "origin_x": float(origin[0]),
        "origin_y": float(origin[1]),
        "tangent_x": float(tangent[0]),
        "tangent_y": float(tangent[1]),
        "normal_x": float(normal[0]),
        "normal_y": float(normal[1]),
        "s_min": float(out["s"].min()),
        "s_max": float(out["s"].max()),
        "lateral_width_median": float(np.median(out["upper_l"] - out["lower_l"])),
    }
    return out, meta


def plot_boundary(
    group: pd.DataFrame,
    boundary: pd.DataFrame,
    run_id: int,
    lane_kf: int,
    out_path: Path,
) -> None:
    plot_df = group
    if len(plot_df) > 200_000:
        plot_df = plot_df.sample(n=200_000, random_state=0)
    fig, ax = plt.subplots(figsize=(9, 8), constrained_layout=True)
    ax.scatter(plot_df["xloc_kf"], plot_df["yloc_kf"], s=1, alpha=0.12, color="0.35")
    ax.plot(boundary["center_x"], boundary["center_y"], color="C0", lw=2, label="center")
    ax.plot(boundary["lower_x"], boundary["lower_y"], color="C3", lw=2, label="lower")
    ax.plot(boundary["upper_x"], boundary["upper_y"], color="C2", lw=2, label="upper")
    ax.set_title(f"Boundary envelope, run {run_id}, lane_kf {lane_kf}")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.25)
    ax.legend()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Derive boundary polylines per run_id/lane_kf")
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--class-id", type=float, default=1.0)
    parser.add_argument("--bin-width", type=float, default=8.0)
    parser.add_argument("--p-low", type=float, default=1.0)
    parser.add_argument("--p-high", type=float, default=99.0)
    parser.add_argument("--buffer", type=float, default=1.0)
    parser.add_argument("--min-count", type=int, default=40)
    parser.add_argument("--smooth-window", type=int, default=7)
    args = parser.parse_args()

    df = pd.read_csv(
        args.csv,
        usecols=["id", "time", "xloc_kf", "yloc_kf", "lane_kf", "class", "run_id"],
    ).dropna()
    if args.class_id is not None:
        df = df[df["class"] == args.class_id].copy()

    out_dir = args.out_dir
    fig_dir = out_dir / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    all_boundaries = []
    meta_rows = []
    for (run_id, lane_kf), group in df.groupby(["run_id", "lane_kf"], sort=True):
        boundary, meta = derive_boundary(
            group,
            bin_width=args.bin_width,
            p_low=args.p_low,
            p_high=args.p_high,
            buffer_m=args.buffer,
            min_count=args.min_count,
            smooth_window=args.smooth_window,
        )
        boundary.insert(0, "lane_kf", int(lane_kf))
        boundary.insert(0, "run_id", int(run_id))
        all_boundaries.append(boundary)
        meta["run_id"] = int(run_id)
        meta["lane_kf"] = int(lane_kf)
        meta_rows.append(meta)
        plot_boundary(
            group,
            boundary,
            int(run_id),
            int(lane_kf),
            fig_dir / f"run_{int(run_id):02d}_lane_{int(lane_kf):02d}_boundary.png",
        )

    boundary_df = pd.concat(all_boundaries, ignore_index=True)
    meta_df = pd.DataFrame(meta_rows)
    boundary_path = out_dir / "highway_boundaries.csv"
    meta_path = out_dir / "highway_boundary_meta.csv"
    boundary_df.to_csv(boundary_path, index=False)
    meta_df.to_csv(meta_path, index=False)
    print(f"Wrote {boundary_path}")
    print(f"Wrote {meta_path}")


if __name__ == "__main__":
    main()
