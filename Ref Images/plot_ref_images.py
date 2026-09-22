"""Render freeway / TGSIM / roundabout curb maps as trajectory backgrounds.

    python "Ref Images/plot_ref_images.py"
"""
from __future__ import annotations

from functools import lru_cache
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Polygon as MplPolygon
from shapely.geometry import JOIN_STYLE, Polygon

ROOT = Path(__file__).resolve().parents[1]
OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

ROAD = "#C9D6E3"
CURB = "#1F4E79"
ISLAND = "#E8C4C0"
BLOCK = "#F3EFE8"
OFF = "#FAFBFC"
def _closed(xy: np.ndarray) -> np.ndarray:
    xy = np.asarray(xy, float)
    if len(xy) and not np.allclose(xy[0], xy[-1]):
        xy = np.vstack([xy, xy[0]])
    return xy


def load_jounieh():
    from Calibration.calibrate_utility_from_data import load_site_roadway

    path = ROOT / "data" / "Lebanon_Jounieh" / "Jounieh_Road_Boundaries.csv"
    return load_site_roadway(path)


def smooth_roundabout(roadway, radius: float = 1.65, simplify: float = 0.21):
    """Round extracted vertices for the map only; the RL curb CSV is unchanged.

    Closing with a round join keeps the long arms straight, fillets sharp corners,
    and flattens the digitization zigzag on the small island.
    """
    geom = roadway.simplify(simplify, preserve_topology=True)
    geom = geom.buffer(radius, join_style=JOIN_STYLE.round, resolution=32)
    geom = geom.buffer(-radius, join_style=JOIN_STYLE.round, resolution=32)
    if geom.is_empty:
        return roadway
    if geom.geom_type == "MultiPolygon":
        geom = max(geom.geoms, key=lambda part: part.area)
    return geom


LANE = "#F8FCFF"
LANE_EDGE = "#6E8AAB"


def load_tgsim():
    from Calibration.calibrate_utility_from_data import load_site_roadway

    path = ROOT / "data" / "TGSIM FB" / "derived_boundaries" / "street_boundaries.csv"
    return load_site_roadway(path)


@lru_cache(maxsize=1)
def load_tgsim_lanes():
    """Provided Foggy Bottom lane polygons, in the same metre frame as the curb."""
    from importlib.util import module_from_spec, spec_from_file_location

    spec = spec_from_file_location(
        "tgsim_street_boundaries",
        ROOT / "data" / "TGSIM FB" / "build_street_boundaries.py",
    )
    mod = module_from_spec(spec)
    spec.loader.exec_module(mod)
    txt = ROOT / "data" / "TGSIM FB" / "Foggy_Bottom_boundaries.txt"
    traj_path = ROOT / "data" / "TGSIM FB" / "prepared" / "trajectories_calibration.csv"
    polys_px = mod.parse_pixel_polygons(txt)
    traj = pd.read_csv(traj_path, usecols=["lane_kf", "xloc_kf", "yloc_kf"]) if traj_path.is_file() else pd.DataFrame()
    scale = mod.estimate_m_per_px(traj, polys_px) if len(traj) else 0.01861459
    lanes = []
    for xy in polys_px.values():
        poly = Polygon(np.asarray(xy, float) * scale)
        if not poly.is_valid:
            poly = poly.buffer(0)
        if not poly.is_empty:
            lanes.append(poly)
    return lanes


def _plot_linework(ax, geom, **kwargs) -> None:
    if geom is None or geom.is_empty:
        return
    kind = geom.geom_type
    if kind == "LineString":
        xy = np.asarray(geom.coords, float)
        if len(xy) >= 2:
            ax.plot(xy[:, 0], xy[:, 1], **kwargs)
        return
    if kind in ("MultiLineString", "GeometryCollection"):
        for part in geom.geoms:
            _plot_linework(ax, part, **kwargs)


def draw_tgsim_lanes(ax, roadway, lanes) -> None:
    from shapely.ops import unary_union

    curb = roadway.boundary.buffer(0.45)
    internals = []
    for lane in lanes:
        edge = lane.boundary.difference(curb)
        if not edge.is_empty:
            internals.append(edge)
    if not internals:
        return
    _plot_linework(
        ax,
        unary_union(internals),
        color=LANE_EDGE,
        lw=0.7,
        ls=(0, (3.2, 2.4)),
        solid_capstyle="butt",
        zorder=3,
        alpha=0.95,
    )


def load_freeway(run_id: int = 2, lane_kf: int = 1):
    df = pd.read_csv(ROOT / "data" / "Lebanon_Highway" / "derived_highway_boundaries" / "highway_boundaries.csv")
    g = df[(df["run_id"] == run_id) & (df["lane_kf"] == lane_kf)].sort_values("point_index")
    lower = g[["lower_x", "lower_y"]].to_numpy(float)
    upper = g[["upper_x", "upper_y"]].to_numpy(float)
    ring = np.vstack([lower, upper[::-1]])
    poly = Polygon(_closed(ring))
    if not poly.is_valid:
        poly = poly.buffer(0)
    return poly, lower, upper, (run_id, lane_kf)


def draw_site(ax, roadway, *, hole_fill: str) -> None:
    ax.set_facecolor(OFF)
    ext = np.asarray(roadway.exterior.coords, float)
    ax.add_patch(MplPolygon(ext, closed=True, facecolor=ROAD, edgecolor=CURB, lw=1.7, zorder=1, joinstyle="round"))
    for hole in roadway.interiors:
        ax.add_patch(
            MplPolygon(np.asarray(hole.coords, float), closed=True, facecolor=hole_fill, edgecolor=CURB, lw=1.25, zorder=2)
        )


def draw_freeway(ax, poly, lower, upper) -> None:
    ax.set_facecolor(OFF)
    ax.add_patch(
        MplPolygon(np.asarray(poly.exterior.coords, float), closed=True, facecolor=ROAD, edgecolor=CURB, lw=1.5, zorder=1)
    )
    ax.plot(lower[:, 0], lower[:, 1], color=CURB, lw=1.5, zorder=3)
    ax.plot(upper[:, 0], upper[:, 1], color=CURB, lw=1.5, zorder=3)


def scale_bar(ax, length_m: float, corner: str = "lower left", pad_frac: float = 0.045, xy=None) -> None:
    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    span_x, span_y = x1 - x0, y1 - y0
    if xy is None:
        x = x1 - pad_frac * span_x - length_m if "right" in corner else x0 + pad_frac * span_x
        y = y1 - pad_frac * span_y if "upper" in corner else y0 + pad_frac * span_y
        upper = "upper" in corner
    else:
        x, y = xy
        upper = False
    tick = 0.012 * span_y
    ax.plot([x, x + length_m], [y, y], color="#1a1a1a", lw=2.4, zorder=10, solid_capstyle="butt")
    ax.plot([x, x], [y - tick, y + tick], color="#1a1a1a", lw=2.0, zorder=10)
    ax.plot([x + length_m, x + length_m], [y - tick, y + tick], color="#1a1a1a", lw=2.0, zorder=10)
    va = "top" if upper else "bottom"
    dy = -0.028 * span_y if upper else 0.028 * span_y
    ax.text(x + 0.5 * length_m, y + dy, f"{int(length_m)} m", ha="center", va=va, fontsize=9, color="#1a1a1a", zorder=10)


def finish(ax, roadway_or_bounds, *, pad: float, labeled: bool, title: str | None) -> list[float]:
    if hasattr(roadway_or_bounds, "bounds"):
        minx, miny, maxx, maxy = roadway_or_bounds.bounds
    else:
        minx, miny, maxx, maxy = roadway_or_bounds
    ax.set_xlim(minx - pad, maxx + pad)
    ax.set_ylim(miny - pad, maxy + pad)
    ax.set_aspect("equal", adjustable="box")
    if labeled:
        ax.set_xlabel("x (m)")
        ax.set_ylabel("y (m)")
        ax.set_title(title, fontsize=13, pad=8)
        ax.tick_params(labelsize=8)
        for spine in ax.spines.values():
            spine.set_color("#C5CDD4")
    else:
        ax.set_axis_off()
    return [float(minx - pad), float(maxx + pad), float(miny - pad), float(maxy + pad)]


def save(fig, path: Path, *, bleed: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if bleed:
        fig.savefig(path, dpi=220, facecolor="white", pad_inches=0)
    else:
        fig.savefig(path, dpi=220, facecolor="white", bbox_inches="tight", pad_inches=0.12)
    plt.close(fig)
    print(f"Wrote {path}", flush=True)


def _fig_for(extent, *, labeled: bool, long_side: float):
    xmin, xmax, ymin, ymax = extent
    w, h = xmax - xmin, ymax - ymin
    aspect = w / max(h, 1e-6)
    if labeled:
        extra = 0.85
        if aspect >= 1:
            figsize = (long_side + 0.4, long_side / aspect + extra)
        else:
            figsize = (long_side * aspect + 0.7, long_side + extra)
        fig, ax = plt.subplots(figsize=figsize)
        return fig, ax
    if aspect >= 1:
        figsize = (long_side, long_side / aspect)
    else:
        figsize = (long_side * aspect, long_side)
    fig, ax = plt.subplots(figsize=figsize)
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    ax.set_position([0, 0, 1, 1])
    return fig, ax


def panel_freeway(ax, labeled: bool):
    poly, lower, upper, key = load_freeway()
    draw_freeway(ax, poly, lower, upper)
    pad = 18.0
    extent = finish(ax, poly, pad=pad, labeled=labeled, title="Freeway  ·  extracted curb")
    if labeled:
        scale_bar(ax, 50, corner="lower right")
    return extent, key


def panel_tgsim(ax, labeled: bool):
    road = load_tgsim()
    draw_site(ax, road, hole_fill=BLOCK)
    draw_tgsim_lanes(ax, road, load_tgsim_lanes())
    pad = 8.0
    extent = finish(ax, road, pad=pad, labeled=labeled, title="TGSIM Foggy Bottom  ·  extracted curb and lane markings")
    if labeled:
        scale_bar(ax, 50, xy=(70.0, 92.0))
    return extent


def panel_roundabout(ax, labeled: bool):
    road = smooth_roundabout(load_jounieh())
    draw_site(ax, road, hole_fill=ISLAND)
    pad = 2.0
    extent = finish(ax, road, pad=pad, labeled=labeled, title="Jounieh roundabout  ·  extracted curb and islands")
    if labeled:
        scale_bar(ax, 10, corner="lower left")
    return extent


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    meta = {}

    fig, ax = plt.subplots(figsize=(8.8, 6.8))
    extent, key = panel_freeway(ax, True)
    save(fig, OUT / "freeway.png")
    fig, ax = _fig_for(extent, labeled=False, long_side=8.4)
    panel_freeway(ax, False)
    save(fig, OUT / "freeway_bg.png", bleed=True)
    meta["freeway"] = {"file": "freeway_bg.png", "extent": extent, "run_id": key[0], "lane_kf": key[1]}

    fig, ax = plt.subplots(figsize=(6.5, 10.4))
    extent = panel_tgsim(ax, True)
    save(fig, OUT / "tgsim.png")
    fig, ax = _fig_for(extent, labeled=False, long_side=10.0)
    panel_tgsim(ax, False)
    save(fig, OUT / "tgsim_bg.png", bleed=True)
    meta["tgsim"] = {"file": "tgsim_bg.png", "extent": extent}

    fig, ax = plt.subplots(figsize=(9.4, 5.7))
    extent = panel_roundabout(ax, True)
    save(fig, OUT / "roundabout.png")
    fig, ax = _fig_for(extent, labeled=False, long_side=9.2)
    panel_roundabout(ax, False)
    save(fig, OUT / "roundabout_bg.png", bleed=True)
    meta["roundabout"] = {"file": "roundabout_bg.png", "extent": extent}

    fig = plt.figure(figsize=(13.6, 6.8))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.15, 0.72, 1.15], wspace=0.18)
    ax0 = fig.add_subplot(gs[0, 0])
    ax1 = fig.add_subplot(gs[0, 1])
    ax2 = fig.add_subplot(gs[0, 2])
    panel_freeway(ax0, True)
    panel_tgsim(ax1, True)
    panel_roundabout(ax2, True)
    fig.suptitle("Extracted site geometry used as trajectory backgrounds", fontsize=14, y=1.02)
    save(fig, OUT / "sites.png")

    (OUT / "extents.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    print("Wrote", OUT / "extents.json", flush=True)


if __name__ == "__main__":
    main()
