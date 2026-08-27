#!/usr/bin/env python
"""Build curb linework from TGSIM provided lane polygons.

Unions all lane polygons in ``Foggy_Bottom_boundaries.txt`` and keeps only the
roadway boundary rings (outer curb + block holes). Internal lane edges drop out.

Outputs under ``derived_boundaries/``:
  street_boundaries.csv
  street_boundaries_meta.csv
"""

from __future__ import annotations

import argparse
import ast
import re
from pathlib import Path

import numpy as np
import pandas as pd
from shapely.geometry import MultiPolygon, Polygon
from shapely.ops import unary_union

_SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_TXT = _SCRIPT_DIR / "Foggy_Bottom_boundaries.txt"
DEFAULT_TRAJ = _SCRIPT_DIR / "prepared" / "trajectories_calibration.csv"
DEFAULT_OUT = _SCRIPT_DIR / "derived_boundaries"


def parse_pixel_polygons(path: Path) -> dict[int, np.ndarray]:
    text = path.read_text(encoding="utf-8")
    out: dict[int, np.ndarray] = {}
    for line in text.splitlines():
        line = line.strip()
        if not line or "=" not in line or line.lower().startswith("lane"):
            continue
        m = re.match(r"(\d+)\s*=\s*(\[.*\])", line)
        if not m:
            continue
        pts = ast.literal_eval(m.group(2).replace(",(", ", ("))
        out[int(m.group(1))] = np.asarray(pts, dtype=float)
    return out


def estimate_m_per_px(traj: pd.DataFrame, polys: dict[int, np.ndarray]) -> float:
    ratios = []
    for lid, poly in polys.items():
        t = traj[traj["lane_kf"] == lid]
        if len(t) < 30 or len(poly) < 2:
            continue
        tx = float(t["xloc_kf"].max() - t["xloc_kf"].min())
        ty = float(t["yloc_kf"].max() - t["yloc_kf"].min())
        px = float(poly[:, 0].max() - poly[:, 0].min())
        py = float(poly[:, 1].max() - poly[:, 1].min())
        if min(tx, ty, px, py) > 1:
            ratios.extend([tx / px, ty / py])
    return float(np.median(ratios)) if ratios else 0.01861459


def to_polygon(xy: np.ndarray) -> Polygon:
    poly = Polygon(xy)
    if not poly.is_valid:
        poly = poly.buffer(0)
    return poly


def boundary_rings(geom) -> list[tuple[str, np.ndarray]]:
    rings: list[tuple[str, np.ndarray]] = []
    if geom.is_empty:
        return rings

    def _poly_rings(poly: Polygon, prefix: str) -> None:
        rings.append((f"{prefix}_exterior", np.asarray(poly.exterior.coords, dtype=float)))
        for hi, hole in enumerate(poly.interiors):
            rings.append((f"{prefix}_hole_{hi}", np.asarray(hole.coords, dtype=float)))

    if isinstance(geom, Polygon):
        _poly_rings(geom, "roadway")
    elif isinstance(geom, MultiPolygon):
        for i, g in enumerate(geom.geoms):
            _poly_rings(g, f"roadway_{i}")
    else:
        for g in getattr(geom, "geoms", []):
            rings.extend(boundary_rings(g))
    return rings


def build_outer_curb(
    polys_m: dict[int, np.ndarray],
    *,
    seal_m: float = 0.25,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    parts = [to_polygon(xy) for xy in polys_m.values()]
    merged = unary_union(parts)
    merged = merged.buffer(seal_m).buffer(-seal_m)
    if merged.is_empty:
        raise RuntimeError("union produced empty geometry")

    rings = boundary_rings(merged)
    vertex_rows = []
    meta_rows = []
    used = ",".join(str(i) for i in sorted(polys_m))
    for part_i, (name, ring) in enumerate(rings):
        for vi, (x, y) in enumerate(ring):
            vertex_rows.append(
                {
                    "street_id": 1,
                    "street_name": name,
                    "part_index": part_i,
                    "vertex_index": vi,
                    "x": float(x),
                    "y": float(y),
                }
            )
        meta_rows.append(
            {
                "street_id": 1,
                "street_name": name,
                "part_index": part_i,
                "member_lanes": used,
                "n_lanes": len(polys_m),
                "n_vertices": len(ring),
                "area_m2": float(merged.area) if part_i == 0 else np.nan,
                "centroid_x": float(np.mean(ring[:, 0])),
                "centroid_y": float(np.mean(ring[:, 1])),
            }
        )
    return pd.DataFrame(vertex_rows), pd.DataFrame(meta_rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--polygons", type=Path, default=DEFAULT_TXT)
    parser.add_argument("--traj", type=Path, default=DEFAULT_TRAJ)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--m-per-px", type=float, default=None)
    parser.add_argument("--seal-m", type=float, default=0.25)
    args = parser.parse_args()

    polys_px = parse_pixel_polygons(args.polygons)
    traj = pd.read_csv(args.traj) if args.traj.exists() else pd.DataFrame()
    m_per_px = args.m_per_px
    if m_per_px is None:
        m_per_px = estimate_m_per_px(traj, polys_px) if len(traj) else 0.01861459
    print(f"m_per_px = {m_per_px:.8f}")
    polys_m = {lid: pts * m_per_px for lid, pts in polys_px.items()}
    print(f"unioning {len(polys_m)} lane polygons → curb rings")

    street_df, meta = build_outer_curb(polys_m, seal_m=args.seal_m)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    street_path = args.out_dir / "street_boundaries.csv"
    meta_path = args.out_dir / "street_boundaries_meta.csv"
    street_df.to_csv(street_path, index=False)
    meta.to_csv(meta_path, index=False)
    print(f"Wrote {street_path} ({meta.shape[0]} ring(s))")
    print(meta[["street_name", "part_index", "n_vertices", "area_m2"]].to_string(index=False))
    print("Plot via: python data/_plot_site_boundaries.py")


if __name__ == "__main__":
    main()
