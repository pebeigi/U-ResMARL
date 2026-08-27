#!/usr/bin/env python
"""Normalize TGSIM Foggy Bottom trajectories for calibration.

Site curb boundaries are built separately by ``build_street_boundaries.py``
(union of provided lane polygons → outer curb rings).

Outputs:
  prepared/trajectories_calibration.csv
  prepared/type_code_note.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO = _SCRIPT_DIR.parent.parent  # data/<site>/script.py → repo
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from data.filter_calibration_vehicles import vehicle_keep_mask

DEFAULT_CSV = (
    _SCRIPT_DIR / "Third_Generation_Simulation_Data__TGSIM__Foggy_Bottom_Trajectories.csv"
)
DEFAULT_OUT = _SCRIPT_DIR / "prepared"


def normalize(df: pd.DataFrame) -> pd.DataFrame:
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
    cols = [
        "id",
        "time",
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
    args = parser.parse_args()

    print(f"Loading {args.csv}...")
    raw = pd.read_csv(args.csv)
    traj = normalize(raw)
    n_before = traj.groupby(["run_id", "id"]).ngroups
    traj, report = vehicle_keep_mask(traj)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    traj_path = args.out_dir / "trajectories_calibration.csv"
    report_path = args.out_dir / "vehicle_filter_report.csv"
    traj.to_csv(traj_path, index=False)
    report.to_csv(report_path, index=False)
    print(
        f"Wrote {traj_path} ({len(traj):,} rows, {traj['id'].nunique()} ids; "
        f"dropped {n_before - int(report['keep'].sum())} vehicles "
        f"as short/stationary)"
    )
    print(f"Wrote {report_path}")

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
    type_path = args.out_dir / "type_code_note.csv"
    type_note.to_csv(type_path, index=False)
    print(f"Wrote {type_path}")
    print(type_note.to_string(index=False))
    print(
        "Boundaries: run build_street_boundaries.py → "
        "derived_boundaries/street_boundaries.csv. "
        "Plot via data/_plot_site_boundaries.py"
    )


if __name__ == "__main__":
    main()
