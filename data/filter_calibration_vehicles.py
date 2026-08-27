#!/usr/bin/env python
"""Drop parked / short tracks from Jounieh and TGSIM calibration CSVs.

A vehicle (run_id, id) is kept only if:
  * it is present for at least ``min_duration_s`` seconds, and
  * it is not stationary (85th-percentile speed >= ``min_speed_p85`` and
    travelled path length >= ``min_path_m``).

Dropped IDs are removed entirely so they are neither ego nor neighbors.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

MIN_DURATION_S = 10.0
MIN_SPEED_P85 = 1.0  # m/s — 85% of samples below this → parked / idle
MIN_PATH_M = 8.0


def vehicle_keep_mask(
    df: pd.DataFrame,
    *,
    min_duration_s: float = MIN_DURATION_S,
    min_speed_p85: float = MIN_SPEED_P85,
    min_path_m: float = MIN_PATH_M,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (kept_traj, drop_report)."""
    rows = []
    keep_keys = set()
    for (run_id, vid), g in df.groupby(["run_id", "id"], sort=False):
        t = g["time"].to_numpy(float)
        x = g["xloc_kf"].to_numpy(float)
        y = g["yloc_kf"].to_numpy(float)
        v = g["speed_kf"].to_numpy(float)
        duration = float(t.max() - t.min()) if len(t) else 0.0
        path_m = float(np.sum(np.hypot(np.diff(x), np.diff(y)))) if len(t) > 1 else 0.0
        speed_p85 = float(np.quantile(v, 0.85)) if len(v) else 0.0
        too_short = duration < min_duration_s
        stationary = (speed_p85 < min_speed_p85) or (path_m < min_path_m)
        keep = (not too_short) and (not stationary)
        if keep:
            keep_keys.add((int(run_id), int(vid)))
        rows.append(
            {
                "run_id": int(run_id),
                "id": int(vid),
                "n_rows": int(len(g)),
                "duration_s": duration,
                "speed_p85": speed_p85,
                "path_m": path_m,
                "too_short": bool(too_short),
                "stationary": bool(stationary),
                "keep": bool(keep),
            }
        )
    report = pd.DataFrame(rows)
    key = list(zip(df["run_id"].astype(int), df["id"].astype(int)))
    mask = [(int(r), int(i)) in keep_keys for r, i in key]
    kept = df.loc[mask].copy().reset_index(drop=True)
    return kept, report


def apply_and_write(
    traj_path: Path,
    *,
    min_duration_s: float = MIN_DURATION_S,
    min_speed_p85: float = MIN_SPEED_P85,
    min_path_m: float = MIN_PATH_M,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = pd.read_csv(traj_path)
    n_ids_before = df.groupby(["run_id", "id"]).ngroups
    kept, report = vehicle_keep_mask(
        df,
        min_duration_s=min_duration_s,
        min_speed_p85=min_speed_p85,
        min_path_m=min_path_m,
    )
    kept.to_csv(traj_path, index=False)
    report_path = traj_path.with_name("vehicle_filter_report.csv")
    report.to_csv(report_path, index=False)
    dropped = report[~report["keep"]]
    n_short = int((dropped["too_short"] & ~dropped["stationary"]).sum())
    n_stat = int((~dropped["too_short"] & dropped["stationary"]).sum())
    n_both = int((dropped["too_short"] & dropped["stationary"]).sum())
    print(
        f"{traj_path}: {n_ids_before} → {int(report['keep'].sum())} vehicles, "
        f"{len(df):,} → {len(kept):,} rows "
        f"(too_short_only={n_short}, stationary_only={n_stat}, both={n_both})"
    )
    print(f"  wrote {report_path}")
    return kept, report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("traj_csv", type=Path, nargs="+")
    parser.add_argument("--min-duration", type=float, default=MIN_DURATION_S)
    parser.add_argument("--min-speed-p85", type=float, default=MIN_SPEED_P85)
    parser.add_argument("--min-path-m", type=float, default=MIN_PATH_M)
    args = parser.parse_args()
    for path in args.traj_csv:
        apply_and_write(
            path,
            min_duration_s=args.min_duration,
            min_speed_p85=args.min_speed_p85,
            min_path_m=args.min_path_m,
        )


if __name__ == "__main__":
    main()

