"""Shared helpers for Jounieh / TGSIM calibration data prep."""

from __future__ import annotations

import numpy as np
import pandas as pd


def snap_to_time_grid(df: pd.DataFrame, dt: float) -> pd.DataFrame:
    """Keep one sample per (run, id, grid time) so simultaneous vehicles share a clock.

    Independent per-vehicle subsampling leaves neighbors on slightly different
    timestamps, which breaks exact-time neighbor joins during calibration.
    """
    out = df.copy()
    t = out["time"].to_numpy(float)
    grid = np.round(t / dt) * dt
    out["_t_grid"] = grid
    out["_err"] = np.abs(t - grid)
    out = out.sort_values(["run_id", "id", "_t_grid", "_err"], kind="mergesort")
    out = out.drop_duplicates(["run_id", "id", "_t_grid"], keep="first")
    out["time_raw"] = out["time"]
    out["time"] = out["_t_grid"]
    return out.drop(columns=["_t_grid", "_err"]).reset_index(drop=True)


def assign_modal_class(df: pd.DataFrame) -> pd.DataFrame:
    """One class label per vehicle (mode of its rows)."""
    mode = (
        df.groupby(["run_id", "id"], sort=False)["class"]
        .agg(lambda s: s.mode().iloc[0] if len(s.mode()) else s.iloc[0])
        .rename("class")
        .reset_index()
    )
    out = df.drop(columns=["class"]).merge(mode, on=["run_id", "id"], how="left")
    return out


def desired_speed_from_track(speed: np.ndarray) -> float:
    """Per-ID desired speed from that vehicle's own trajectory.

    Uses the 95th percentile of samples faster than 1 m/s so stopped/queued
    frames do not pull v_des down. Falls back to the track max if needed.
    """
    v = np.asarray(speed, dtype=float)
    v = v[np.isfinite(v)]
    if len(v) == 0:
        return 1.0
    moving = v[v > 1.0]
    src = moving if len(moving) >= 3 else v
    return float(max(np.quantile(src, 0.95), 1.0))


def max_speed_from_track(speed: np.ndarray) -> float:
    """Per-ID kinematic speed cap = max observed speed on that track."""
    v = np.asarray(speed, dtype=float)
    v = v[np.isfinite(v)]
    if len(v) == 0:
        return 1.0
    return float(max(float(np.max(v)), 1.0))


def attach_desired_speed(df: pd.DataFrame) -> pd.DataFrame:
    """Stamp every row with that ID's desired_speed and max_speed from its track."""
    out = df.copy()
    mapped = (
        out.groupby(["run_id", "id"], sort=False)["speed_kf"]
        .agg(
            desired_speed=lambda s: desired_speed_from_track(s.to_numpy(float)),
            max_speed=lambda s: max_speed_from_track(s.to_numpy(float)),
        )
        .reset_index()
    )
    for col in ("desired_speed", "max_speed"):
        if col in out.columns:
            out = out.drop(columns=[col])
    return out.merge(mapped, on=["run_id", "id"], how="left").assign(
        max_speed=lambda d: np.maximum(d["max_speed"].to_numpy(float), d["desired_speed"].to_numpy(float))
    )


def origin_dest_table(df: pd.DataFrame) -> pd.DataFrame:
    """First-sample kinematics and last-sample destination for every ID."""
    rows = []
    for (run_id, vid), g in df.sort_values("time").groupby(["run_id", "id"], sort=False):
        g = g.sort_values("time")
        first = g.iloc[0]
        last = g.iloc[-1]
        x = g["xloc_kf"].to_numpy(float)
        y = g["yloc_kf"].to_numpy(float)
        path_m = float(np.sum(np.hypot(np.diff(x), np.diff(y)))) if len(g) > 1 else 0.0
        dx = float(last["xloc_kf"] - first["xloc_kf"])
        dy = float(last["yloc_kf"] - first["yloc_kf"])
        heading0 = float("nan")
        if len(g) > 1:
            step = np.hypot(np.diff(x), np.diff(y))
            moving = np.where(step > 0.05)[0]
            if len(moving):
                k = int(moving[0])
                heading0 = float(np.arctan2(y[k + 1] - y[k], x[k + 1] - x[k]))
        keep = bool(first["keep_ego"]) if "keep_ego" in g.columns else True
        rows.append(
            {
                "run_id": int(run_id),
                "id": int(vid),
                "class": float(first["class"]) if "class" in g.columns else np.nan,
                "keep_ego": keep,
                "n_rows": int(len(g)),
                "t0": float(first["time"]),
                "t_end": float(last["time"]),
                "duration_s": float(last["time"] - first["time"]),
                "x0": float(first["xloc_kf"]),
                "y0": float(first["yloc_kf"]),
                "speed0": float(first["speed_kf"]),
                "desired_speed": desired_speed_from_track(g["speed_kf"].to_numpy(float)),
                "max_speed": max_speed_from_track(g["speed_kf"].to_numpy(float)),
                "heading0": heading0,
                "dest_x": float(last["xloc_kf"]),
                "dest_y": float(last["yloc_kf"]),
                "chord_m": float(np.hypot(dx, dy)),
                "path_m": path_m,
                "lane_kf": int(first["lane_kf"]) if "lane_kf" in g.columns else -1,
            }
        )
    return pd.DataFrame(rows)
