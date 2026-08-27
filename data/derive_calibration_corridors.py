#!/usr/bin/env python
"""Derive per-(run,lane) corridor envelopes for the calibration pipeline.

These are *utility corridor* polylines (center/lower/upper), not the site curb
polygons used for QA. Same schema as Lebanon_Highway
``derived_highway_boundaries/highway_boundaries.csv``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from data.Lebanon_Highway.derive_highway_boundaries import derive_boundary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--class-id", type=float, default=None)
    parser.add_argument("--bin-width", type=float, default=4.0)
    parser.add_argument("--p-low", type=float, default=1.0)
    parser.add_argument("--p-high", type=float, default=99.0)
    parser.add_argument("--buffer", type=float, default=0.75)
    parser.add_argument("--min-count", type=int, default=30)
    parser.add_argument("--smooth-window", type=int, default=5)
    args = parser.parse_args()

    usecols = ["id", "time", "xloc_kf", "yloc_kf", "lane_kf", "class", "run_id"]
    df = pd.read_csv(args.csv, usecols=usecols).dropna()
    if args.class_id is not None:
        df = df[df["class"] == args.class_id].copy()
        print(f"class={args.class_id}: {len(df):,} rows, {df['id'].nunique()} ids")
    else:
        print(f"all classes: {len(df):,} rows, {df['id'].nunique()} ids")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    all_boundaries = []
    meta_rows = []
    for (run_id, lane_kf), group in df.groupby(["run_id", "lane_kf"], sort=True):
        if len(group) < max(args.min_count * 3, 150):
            print(f"  skip run={run_id} lane={lane_kf}: only {len(group)} points")
            continue
        try:
            boundary, meta = derive_boundary(
                group,
                bin_width=args.bin_width,
                p_low=args.p_low,
                p_high=args.p_high,
                buffer_m=args.buffer,
                min_count=args.min_count,
                smooth_window=args.smooth_window,
            )
        except RuntimeError as exc:
            print(f"  skip run={run_id} lane={lane_kf}: {exc}")
            continue
        boundary.insert(0, "lane_kf", int(lane_kf))
        boundary.insert(0, "run_id", int(run_id))
        all_boundaries.append(boundary)
        meta["run_id"] = int(run_id)
        meta["lane_kf"] = int(lane_kf)
        meta_rows.append(meta)
        print(
            f"  run={run_id} lane={lane_kf}: {len(boundary)} verts, "
            f"width≈{meta['lateral_width_median']:.2f} m"
        )

    if not all_boundaries:
        raise SystemExit("No corridors derived")
    bound_path = args.out_dir / "highway_boundaries.csv"
    meta_path = args.out_dir / "highway_boundary_meta.csv"
    pd.concat(all_boundaries, ignore_index=True).to_csv(bound_path, index=False)
    pd.DataFrame(meta_rows).to_csv(meta_path, index=False)
    print(f"Wrote {bound_path} ({len(all_boundaries)} corridors)")
    print(f"Wrote {meta_path}")


if __name__ == "__main__":
    main()
