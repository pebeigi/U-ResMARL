#!/usr/bin/env python
"""Normalize Lebanon_Jounieh trajectories for calibration.

Site boundaries are the provided outer/island polygons in
``Jounieh_Road_Boundaries.csv`` (not PCA envelopes).

Outputs:
  prepared/trajectories_calibration.csv
  prepared/lane_code_map.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO = _SCRIPT_DIR.parent.parent  # data/<site>/script.py → repo
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from data.filter_calibration_vehicles import vehicle_keep_mask

DEFAULT_CSV = _SCRIPT_DIR / "Final_Jounieh.csv"
DEFAULT_OUT = _SCRIPT_DIR / "prepared"
TARGET_DT = 0.1


def subsample_to_dt(df: pd.DataFrame, target_dt: float) -> pd.DataFrame:
    """Keep roughly one sample every ``target_dt`` seconds within each vehicle."""
    parts = []
    for (_, _), g in df.groupby(["run_id", "id"], sort=False):
        g = g.sort_values("time")
        if g.empty:
            continue
        t0 = float(g["time"].iloc[0])
        keep = []
        next_t = t0
        for idx, t in zip(g.index, g["time"].to_numpy(float)):
            if t + 1e-9 >= next_t:
                keep.append(idx)
                next_t = t + target_dt
        parts.append(g.loc[keep])
    return pd.concat(parts, ignore_index=True) if parts else df.iloc[0:0].copy()


def normalize(df: pd.DataFrame, target_dt: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    out = df.copy()
    out["lane_kf_raw"] = out["lane_kf"].astype(str)
    codes = sorted(out["lane_kf_raw"].unique())
    code_map = {c: i + 1 for i, c in enumerate(codes)}
    out["lane_kf"] = out["lane_kf_raw"].map(code_map).astype(int)
    out["run_id"] = out["run_id"].astype(int)
    out["class"] = out["class"].astype(float)
    out = subsample_to_dt(out, target_dt)

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
    args = parser.parse_args()

    print(f"Loading {args.csv}...")
    raw = pd.read_csv(args.csv)
    traj, lane_map = normalize(raw, args.target_dt)
    n_before = traj.groupby(["run_id", "id"]).ngroups
    traj, report = vehicle_keep_mask(traj)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    traj_path = args.out_dir / "trajectories_calibration.csv"
    map_path = args.out_dir / "lane_code_map.csv"
    report_path = args.out_dir / "vehicle_filter_report.csv"
    traj.to_csv(traj_path, index=False)
    lane_map.to_csv(map_path, index=False)
    report.to_csv(report_path, index=False)
    print(
        f"Wrote {traj_path} ({len(traj):,} rows, {traj['id'].nunique()} ids; "
        f"dropped {n_before - int(report['keep'].sum())} vehicles "
        f"as short/stationary)"
    )
    print(f"Wrote {map_path}")
    print(f"Wrote {report_path}")
    print(lane_map.to_string(index=False))
    print("Boundaries: use Jounieh_Road_Boundaries.csv (provided). Plot via data/_plot_site_boundaries.py")


if __name__ == "__main__":
    main()
