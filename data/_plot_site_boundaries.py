#!/usr/bin/env python
"""QA plots for Jounieh and TGSIM site curbs, trajectories, and per-ID origin/dest.

Writes under ``data/_qa_plots/``:
  {site}_boundaries_only.png
  {site}_boundaries_with_traj.png
  {site}_origin_dest.png
  {site}_example_ids.png
  {site}_setup_summary.csv
  tgsim_boundaries_on_reference.png  (if the Foggy Bottom still is present)
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import numpy as np
import pandas as pd
from shapely.geometry import Polygon
from shapely.ops import unary_union
import shapely

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "_qa_plots"


def _closed_xy(group: pd.DataFrame) -> np.ndarray:
    xy = group.sort_values("vertex_index")[["x", "y"]].to_numpy(float)
    if len(xy) and not np.allclose(xy[0], xy[-1]):
        xy = np.vstack([xy, xy[0]])
    return xy


def load_jounieh_rings() -> tuple[list[tuple[str, np.ndarray]], object]:
    bound = pd.read_csv(ROOT / "Lebanon_Jounieh" / "Jounieh_Road_Boundaries.csv")
    rings = []
    outers, holes = [], []
    for (kind, pid), g in bound.groupby(["kind", "polygon_id"], sort=True):
        xy = _closed_xy(g)
        rings.append((f"{kind} {pid}", xy))
        poly = Polygon(xy)
        if not poly.is_valid:
            poly = poly.buffer(0)
        if str(kind) == "outer":
            outers.append(poly)
        else:
            holes.append(poly)
    road = unary_union(outers) if outers else None
    if road is not None:
        for hole in holes:
            road = road.difference(hole)
    return rings, road


def load_tgsim_rings() -> tuple[list[tuple[str, np.ndarray]], object]:
    bound = pd.read_csv(ROOT / "TGSIM FB" / "derived_boundaries" / "street_boundaries.csv")
    rings = []
    xy_parts = []
    for part_i, g in bound.groupby("part_index", sort=True):
        xy = _closed_xy(g)
        name = str(g["street_name"].iloc[0]) if "street_name" in g.columns else f"part {part_i}"
        rings.append((name, xy))
        xy_parts.append(xy)
    if not xy_parts:
        return rings, None
    poly = Polygon(xy_parts[0], xy_parts[1:]) if len(xy_parts) > 1 else Polygon(xy_parts[0])
    if not poly.is_valid:
        poly = poly.buffer(0)
    return rings, poly if not poly.is_empty else None


def draw_rings(ax, rings: list[tuple[str, np.ndarray]], *, fill_holes: bool = True) -> None:
    for i, (name, xy) in enumerate(rings):
        is_hole = "island" in name.lower() or "hole" in name.lower()
        color = "#C0392B" if is_hole else "#1F4E79"
        lw = 1.8 if is_hole else 2.2
        ax.plot(xy[:, 0], xy[:, 1], color=color, lw=lw, label=name if i < 8 else None)
        if fill_holes and is_hole:
            ax.fill(xy[:, 0], xy[:, 1], color="#C0392B", alpha=0.18)


def _sample(df: pd.DataFrame, n: int) -> pd.DataFrame:
    return df if len(df) <= n else df.sample(n, random_state=0)


def coverage(xy: np.ndarray, roadway) -> dict[str, float]:
    if roadway is None or len(xy) == 0:
        return {"n": float(len(xy)), "frac_inside": float("nan"), "median_edge_m": float("nan")}
    geoms = shapely.points(xy)
    inside = np.asarray(shapely.contains(roadway, geoms), dtype=bool)
    d_edge = np.asarray(shapely.distance(geoms, roadway.boundary), dtype=float)
    return {
        "n": float(len(xy)),
        "frac_inside": float(inside.mean()),
        "median_edge_m": float(np.median(d_edge[inside])) if inside.any() else float("nan"),
        "p95_outside_m": float(np.quantile(d_edge[~inside], 0.95)) if (~inside).any() else 0.0,
    }


def plot_boundaries_only(rings, title: str, out: Path, figsize=(9, 7)) -> None:
    fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)
    draw_rings(ax, rings)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=8)
    fig.savefig(out, dpi=170)
    plt.close(fig)


def plot_with_traj(rings, traj: pd.DataFrame, title: str, out: Path, figsize=(9, 7), n=120_000) -> None:
    fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)
    s = _sample(traj, n)
    ax.scatter(s["xloc_kf"], s["yloc_kf"], s=1.0, alpha=0.10, color="0.35", rasterized=True, label="trajectories")
    draw_rings(ax, rings)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", markerscale=6, fontsize=8)
    fig.savefig(out, dpi=170)
    plt.close(fig)


def plot_origin_dest(
    rings,
    od: pd.DataFrame,
    title: str,
    out: Path,
    figsize=(9, 7),
    max_arrows: int = 80,
) -> None:
    ego = od[od["keep_ego"]].copy() if "keep_ego" in od.columns else od
    fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)
    draw_rings(ax, rings)
    ax.scatter(ego["x0"], ego["y0"], s=18, c="#2E7D32", zorder=3, label="initial (t0)", edgecolors="white", linewidths=0.3)
    ax.scatter(ego["dest_x"], ego["dest_y"], s=28, c="#C62828", marker="^", zorder=3, label="destination (t_end)", edgecolors="white", linewidths=0.3)
    if len(ego) > max_arrows:
        arrows = ego.sample(max_arrows, random_state=0)
    else:
        arrows = ego
    segs = np.stack(
        [arrows[["x0", "y0"]].to_numpy(float), arrows[["dest_x", "dest_y"]].to_numpy(float)],
        axis=1,
    )
    ax.add_collection(LineCollection(segs, colors="#6A1B9A", linewidths=0.6, alpha=0.45, zorder=2, label="origin→dest"))
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=8)
    fig.savefig(out, dpi=170)
    plt.close(fig)


def plot_example_ids(
    rings,
    traj: pd.DataFrame,
    od: pd.DataFrame,
    title: str,
    out: Path,
    n_ids: int = 6,
    figsize=(11, 8),
) -> None:
    ego = od[od["keep_ego"]].copy() if "keep_ego" in od.columns else od
    if ego.empty:
        return
    pick = ego.sort_values("path_m", ascending=False).head(max(n_ids * 3, n_ids))
    if len(pick) > n_ids:
        # spread starts across the map
        xy = pick[["x0", "y0"]].to_numpy(float)
        chosen = [0]
        for _ in range(n_ids - 1):
            dmin = np.min(np.linalg.norm(xy[:, None, :] - xy[chosen][None, :, :], axis=2), axis=1)
            dmin[chosen] = -1
            chosen.append(int(np.argmax(dmin)))
        pick = pick.iloc[chosen]
    else:
        pick = pick.head(n_ids)

    fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)
    draw_rings(ax, rings)
    cmap = plt.get_cmap("tab10")
    for i, (_, row) in enumerate(pick.iterrows()):
        g = traj[(traj["run_id"] == row["run_id"]) & (traj["id"] == row["id"])].sort_values("time")
        color = cmap(i % 10)
        ax.plot(g["xloc_kf"], g["yloc_kf"], color=color, lw=1.8, label=f"id {int(row['id'])}")
        ax.scatter(row["x0"], row["y0"], s=40, color=color, marker="o", zorder=4, edgecolors="k", linewidths=0.4)
        ax.scatter(row["dest_x"], row["dest_y"], s=55, color=color, marker="^", zorder=4, edgecolors="k", linewidths=0.4)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_title(title + "\ncircles = initial, triangles = destination")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=8)
    fig.savefig(out, dpi=170)
    plt.close(fig)


def plot_tgsim_on_reference(rings, traj: pd.DataFrame, od: pd.DataFrame, out: Path) -> Path | None:
    img_path = ROOT / "TGSIM FB" / "Reference_Image_Foggy Bottom.png"
    if not img_path.exists():
        return None
    from PIL import Image

    Image.MAX_IMAGE_PIXELS = None
    im = Image.open(img_path)
    scale = 0.0185869  # m/px used to convert TGSIM polygons
    down = 8
    small = im.resize((im.width // down, im.height // down), Image.BILINEAR)
    arr = np.asarray(small)
    fig, ax = plt.subplots(figsize=(10, 11), constrained_layout=True)
    ax.imshow(arr, extent=[0, im.width * scale, im.height * scale, 0], origin="upper")
    s = _sample(traj, 80_000)
    ax.scatter(s["xloc_kf"], s["yloc_kf"], s=0.6, alpha=0.08, color="yellow", rasterized=True)
    for _, xy in rings:
        ax.plot(xy[:, 0], xy[:, 1], color="cyan", lw=1.2)
    ego = od[od["keep_ego"]] if "keep_ego" in od.columns else od
    if len(ego) > 400:
        ego = ego.sample(400, random_state=0)
    ax.scatter(ego["x0"], ego["y0"], s=8, c="lime", label="initial")
    ax.scatter(ego["dest_x"], ego["dest_y"], s=10, c="red", marker="^", label="dest")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_title("TGSIM — curb + origin/dest on reference still (y down)")
    ax.legend(loc="best", fontsize=8, markerscale=2)
    fig.savefig(out, dpi=160)
    plt.close(fig)
    return out


def summarize(name: str, traj: pd.DataFrame, od: pd.DataFrame, roadway, rings) -> dict:
    ego = od[od["keep_ego"]] if "keep_ego" in od.columns else od
    xy = traj[["xloc_kf", "yloc_kf"]].to_numpy(float)
    if len(xy) > 80_000:
        rng = np.random.default_rng(0)
        xy = xy[rng.choice(len(xy), 80_000, replace=False)]
    cov = coverage(xy, roadway)
    row = {
        "site": name,
        "n_rows": int(len(traj)),
        "n_ids": int(od.shape[0]),
        "n_ego": int(len(ego)),
        "n_scene_only": int(len(od) - len(ego)),
        "frac_points_inside_curb": cov["frac_inside"],
        "median_inside_edge_clearance_m": cov["median_edge_m"],
        "p95_outside_distance_m": cov.get("p95_outside_m", np.nan),
        "n_boundary_rings": len(rings),
        "ego_chord_m_p50": float(ego["chord_m"].median()) if len(ego) else np.nan,
        "ego_path_m_p50": float(ego["path_m"].median()) if len(ego) else np.nan,
        "dt_s_p50": float(np.nanmedian(np.diff(np.sort(traj["time"].unique())))) if traj["time"].nunique() > 1 else np.nan,
    }
    return row


def run_jounieh() -> dict:
    rings, road = load_jounieh_rings()
    traj = pd.read_csv(ROOT / "Lebanon_Jounieh" / "prepared" / "trajectories_calibration.csv")
    od_path = ROOT / "Lebanon_Jounieh" / "prepared" / "per_id_origin_dest.csv"
    od = pd.read_csv(od_path) if od_path.exists() else None
    if od is None:
        from data.site_prep import origin_dest_table

        od = origin_dest_table(traj)
    plot_boundaries_only(rings, "Jounieh — site curb (outer + islands)", OUT / "jounieh_boundaries_only.png")
    plot_with_traj(rings, traj, "Jounieh — curb + all trajectories", OUT / "jounieh_boundaries_with_traj.png")
    plot_origin_dest(rings, od, "Jounieh — per-ID initial (green) and destination (red)", OUT / "jounieh_origin_dest.png")
    plot_example_ids(rings, traj, od, "Jounieh — example ego IDs", OUT / "jounieh_example_ids.png")
    return summarize("jounieh", traj, od, road, rings)


def run_tgsim() -> dict:
    rings, road = load_tgsim_rings()
    traj = pd.read_csv(ROOT / "TGSIM FB" / "prepared" / "trajectories_calibration.csv")
    od_path = ROOT / "TGSIM FB" / "prepared" / "per_id_origin_dest.csv"
    od = pd.read_csv(od_path) if od_path.exists() else None
    if od is None:
        from data.site_prep import origin_dest_table

        od = origin_dest_table(traj)
    cars = traj[traj["class"] == 3.0] if "class" in traj.columns else traj
    plot_boundaries_only(
        rings,
        "TGSIM Foggy Bottom — street curb (outer + block hole)",
        OUT / "tgsim_boundaries_only.png",
        figsize=(8, 11),
    )
    plot_with_traj(
        rings,
        cars,
        "TGSIM Foggy Bottom — curb + class-3 trajectories",
        OUT / "tgsim_boundaries_with_traj.png",
        figsize=(8, 11),
    )
    od_cars = od.copy()
    if "class" in od_cars.columns:
        od_cars = od_cars[od_cars["class"] == 3.0]
    if "keep_ego" in od_cars.columns:
        od_cars = od_cars[od_cars["keep_ego"].astype(bool)]
    plot_origin_dest(
        rings,
        od_cars,
        "TGSIM — per-ID initial (green) and destination (red), class 3 ego",
        OUT / "tgsim_origin_dest.png",
        figsize=(8, 11),
        max_arrows=100,
    )
    plot_example_ids(
        rings,
        cars,
        od_cars,
        "TGSIM — example class-3 ego IDs",
        OUT / "tgsim_example_ids.png",
        figsize=(8, 11),
    )
    overlay = plot_tgsim_on_reference(rings, cars, od, OUT / "tgsim_boundaries_on_reference.png")
    if overlay:
        print("wrote", overlay)
    return summarize("tgsim", traj, od, road, rings)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--site", choices=("jounieh", "tgsim", "both"), default="both")
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    if args.site in ("jounieh", "both"):
        rows.append(run_jounieh())
        print("wrote Jounieh QA plots under", OUT)
    if args.site in ("tgsim", "both"):
        rows.append(run_tgsim())
        print("wrote TGSIM QA plots under", OUT)
    summary = pd.DataFrame(rows)
    summary_path = OUT / "site_setup_summary.csv"
    summary.to_csv(summary_path, index=False)
    print(summary.to_string(index=False))
    print("wrote", summary_path)


if __name__ == "__main__":
    main()
