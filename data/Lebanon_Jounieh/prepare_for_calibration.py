#!/usr/bin/env python
"""Normalize Lebanon_Jounieh trajectories for calibration.

Site boundaries are the provided outer/island polygons in
``Jounieh_Road_Boundaries.csv`` (not PCA envelopes).

All IDs stay in the CSV. ``keep_ego`` marks vehicles used as calibration
targets; parked / short tracks remain as neighbors. Times are snapped to a
shared 0.1 s grid so agents around an ego ID join on the same clock.
Each ID's first sample is the initial state and last sample is the destination.

Outputs:
  prepared/trajectories_calibration.csv
  prepared/lane_code_map.csv
  prepared/vehicle_filter_report.csv
  prepared/per_id_origin_dest.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO = _SCRIPT_DIR.parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from data.filter_calibration_vehicles import tag_ego_vehicles
from data.site_prep import assign_modal_class, attach_desired_speed, origin_dest_table, snap_to_time_grid

DEFAULT_CSV = _SCRIPT_DIR / "Final_Jounieh.csv"
DEFAULT_OUT = _SCRIPT_DIR / "prepared"
TARGET_DT = 0.1
COORDINATE_SCALE = 3.0


def normalize(df: pd.DataFrame, target_dt: float, coordinate_scale: float = COORDINATE_SCALE) -> tuple[pd.DataFrame, pd.DataFrame]:
    out = df.copy()
    if coordinate_scale <= 0:
        raise ValueError("Coordinate scale must be positive")
    for column in ("xloc_kf", "yloc_kf", "speed_kf", "acceleration_kf",
                   "length_smoothed", "width_smoothed"):
        out[column] = out[column].astype(float) * coordinate_scale
    out["lane_kf_raw"] = out["lane_kf"].astype(str)
    codes = sorted(out["lane_kf_raw"].unique())
    code_map = {c: i + 1 for i, c in enumerate(codes)}
    out["lane_kf"] = out["lane_kf_raw"].map(code_map).astype(int)
    out["run_id"] = out["run_id"].astype(int)
    out["class"] = out["class"].astype(float)
    out = assign_modal_class(out)
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
        "lane_kf_raw",
    ]
    out = out[cols].sort_values(["run_id", "id", "time"]).reset_index(drop=True)
    map_df = pd.DataFrame(
        {"lane_kf": [code_map[c] for c in codes], "lane_kf_raw": codes}
    )
    return out, map_df


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--target-dt", type=float, default=TARGET_DT)
    parser.add_argument("--coordinate-scale", type=float, default=COORDINATE_SCALE,
                        help="Metre-coordinate factor applied to the original Jounieh CSV")
    args = parser.parse_args()

    print(f"Loading {args.csv}...")
    raw = pd.read_csv(args.csv)
    traj, lane_map = normalize(raw, args.target_dt, args.coordinate_scale)
    n_before = traj.groupby(["run_id", "id"]).ngroups
    traj, report = tag_ego_vehicles(traj)
    traj = attach_desired_speed(traj)
    od = origin_dest_table(traj)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    traj_path = args.out_dir / "trajectories_calibration.csv"
    map_path = args.out_dir / "lane_code_map.csv"
    report_path = args.out_dir / "vehicle_filter_report.csv"
    od_path = args.out_dir / "per_id_origin_dest.csv"
    traj.to_csv(traj_path, index=False)
    lane_map.to_csv(map_path, index=False)
    report.to_csv(report_path, index=False)
    od.to_csv(od_path, index=False)
    n_ego = int(report["keep_ego"].sum())
    print(
        f"Wrote {traj_path} ({len(traj):,} rows, {n_before} ids, "
        f"{n_ego} ego / {n_before - n_ego} scene-only neighbors)"
    )
    print(f"Wrote {map_path}")
    print(f"Wrote {report_path}")
    print(f"Wrote {od_path}")
    print(lane_map.to_string(index=False))
    print("Boundaries: use Jounieh_Road_Boundaries.csv. Plot via data/_plot_site_boundaries.py")


if __name__ == "__main__":
    main()
