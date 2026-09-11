#!/usr/bin/env python
"""Normalize TGSIM Foggy Bottom trajectories for calibration.

Site curb boundaries are built separately by ``build_street_boundaries.py``
(union of provided lane polygons → outer curb rings).

All IDs stay in the CSV. ``keep_ego`` marks vehicles used as calibration
targets (typically class 3); parked / short / other-class tracks remain as
neighbors. Each ID's first sample is the initial state and last sample is
the destination.

Outputs:
  prepared/trajectories_calibration.csv
  prepared/type_code_note.csv
  prepared/vehicle_filter_report.csv
  prepared/per_id_origin_dest.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO = _SCRIPT_DIR.parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from data.filter_calibration_vehicles import tag_ego_vehicles
from data.site_prep import attach_desired_speed, origin_dest_table, snap_to_time_grid

DEFAULT_CSV = (
    _SCRIPT_DIR / "Third_Generation_Simulation_Data__TGSIM__Foggy_Bottom_Trajectories.csv"
)
DEFAULT_OUT = _SCRIPT_DIR / "prepared"
TARGET_DT = 0.1


def normalize(df: pd.DataFrame, target_dt: float) -> pd.DataFrame:
    out = df.copy()
    vx = out["speed_kf_x"].to_numpy(float)
    vy = out["speed_kf_y"].to_numpy(float)
    ax = out["acceleration_kf_x"].to_numpy(float)
    ay = out["acceleration_kf_y"].to_numpy(float)
    speed = np.hypot(vx, vy)
    with np.errstate(invalid="ignore", divide="ignore"):
        accel = np.where(speed > 0.05, (vx * ax + vy * ay) / speed, np.hypot(ax, ay))
    out["speed_kf"] = speed
    out["acceleration_kf"] = accel
    out["run_id"] = 1
    out["class"] = out["type_most_common"].astype(float)
    out["lane_kf"] = out["lane_kf"].astype(int)
    out = snap_to_time_grid(out, target_dt)
    cols = [
        "id",
        "time",
        "time_raw",
        "xloc_kf",
        "yloc_kf",
        "lane_kf",
        "speed_kf",
        "acceleration_kf",
        "length_smoothed",
        "width_smoothed",
        "class",
        "run_id",
    ]
    return out[cols].sort_values(["run_id", "id", "time"]).reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--target-dt", type=float, default=TARGET_DT)
    args = parser.parse_args()

    print(f"Loading {args.csv}...")
    raw = pd.read_csv(args.csv)
    traj = normalize(raw, args.target_dt)
    n_before = traj.groupby(["run_id", "id"]).ngroups
    traj, report = tag_ego_vehicles(traj)
    traj = attach_desired_speed(traj)
    od = origin_dest_table(traj)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    traj_path = args.out_dir / "trajectories_calibration.csv"
    report_path = args.out_dir / "vehicle_filter_report.csv"
    od_path = args.out_dir / "per_id_origin_dest.csv"
    traj.to_csv(traj_path, index=False)
    report.to_csv(report_path, index=False)
    od.to_csv(od_path, index=False)
    n_ego = int(report["keep_ego"].sum())
    print(
        f"Wrote {traj_path} ({len(traj):,} rows, {n_before} ids, "
        f"{n_ego} ego / {n_before - n_ego} scene-only neighbors)"
    )
    print(f"Wrote {report_path}")
    print(f"Wrote {od_path}")

    type_note = (
        traj.groupby("class")
        .agg(
            n_rows=("id", "size"),
            n_ids=("id", "nunique"),
            length_med=("length_smoothed", "median"),
            width_med=("width_smoothed", "median"),
            speed_med=("speed_kf", "median"),
        )
        .reset_index()
    )
    ego_ids = traj.loc[traj["keep_ego"]].groupby("class")["id"].nunique()
    type_note["n_ego_ids"] = type_note["class"].map(ego_ids).fillna(0).astype(int)
    type_path = args.out_dir / "type_code_note.csv"
    type_note.to_csv(type_path, index=False)
    print(f"Wrote {type_path}")
    print(type_note.to_string(index=False))
    print(
        "Boundaries: run build_street_boundaries.py -> "
        "derived_boundaries/street_boundaries.csv. "
        "Plot via data/_plot_site_boundaries.py"
    )


if __name__ == "__main__":
    main()
