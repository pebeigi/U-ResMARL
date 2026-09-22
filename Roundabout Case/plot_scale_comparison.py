"""Compare the supplied Jounieh scale with a 3x coordinate scale.

This is visualization only: the source curb and trajectory CSVs are unchanged.
Run from the repository root with ``python "Roundabout Case/plot_scale_comparison.py"``.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as PolygonPatch
import numpy as np
import pandas as pd
from shapely.affinity import scale as scale_polygon

CASE = Path(__file__).resolve().parent
ROOT = CASE.parent
sys.path[:0] = [str(CASE), str(ROOT)]

import activate  # noqa: E402
from config import TRAJECTORIES_CSV  # noqa: E402
from RL.corridor import oriented_box_corners  # noqa: E402


def draw(ax, roadway, poses, tracks, factor, length, width, title):
    polygon = scale_polygon(roadway, xfact=factor, yfact=factor, origin=(0, 0))
    ax.add_patch(PolygonPatch(np.asarray(polygon.exterior.coords), closed=True,
                              facecolor="#dbeafe", edgecolor="#1e3a5f", linewidth=1.3))
    for interior in polygon.interiors:
        ax.add_patch(PolygonPatch(np.asarray(interior.coords), closed=True,
                                  facecolor="white", edgecolor="#1e3a5f", linewidth=1.1))
    ax.scatter(tracks[:, 0] * factor, tracks[:, 1] * factor, s=0.4,
               c="#64748b", alpha=0.15, linewidths=0, label="recorded positions")
    colors = plt.get_cmap("tab10")
    for index, (pos, heading) in enumerate(poses):
        p = pos * factor
        corners = oriented_box_corners(p, heading, length=length, width=width)
        ax.add_patch(PolygonPatch(corners, closed=True, facecolor=colors(index),
                                  edgecolor="#111827", linewidth=0.7, alpha=0.95))
        ax.plot(p[0], p[1], ".", color="#111827", markersize=2)
    minx, miny, maxx, maxy = polygon.bounds
    padding = 0.07 * max(maxx - minx, maxy - miny)
    ax.set_xlim(minx - padding, maxx + padding)
    # Jounieh source coordinates use the image convention: y increases down.
    ax.set_ylim(maxy + padding, miny - padding)
    ax.set_aspect("equal", adjustable="box")
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("metres")
    ax.set_ylabel("metres (image-down)")
    ax.grid(alpha=0.15)


def main():
    activate.apply()
    from Baselines.scenario import build_scenario
    from Calibration.calibrate_utility_from_data import load_site_roadway

    preserved_curb = (
        ROOT / "data" / "Lebanon_Jounieh" / "_source_1x" / "Jounieh_Road_Boundaries_extended_1x.csv"
    )
    roadway = load_site_roadway(preserved_curb)
    scene = build_scenario(seed=810000, num_agents=6, max_steps=2)
    poses = [(agent.pos / 3.0, agent.heading) for agent in scene.agents]
    raw = pd.read_csv(TRAJECTORIES_CSV, usecols=["xloc_kf", "yloc_kf"])
    tracks = raw.sample(n=min(12000, len(raw)), random_state=0)[["xloc_kf", "yloc_kf"]].to_numpy(float) / 3.0

    fig, axes = plt.subplots(1, 3, figsize=(17, 5.2), constrained_layout=True)
    draw(axes[0], roadway, poses, tracks, 1.0, 1.6, 0.64,
         "Former 1× coordinates + 1.6 × 0.64 m cars")
    draw(axes[1], roadway, poses, tracks, 1.0, 4.5, 1.8,
         "Former 1× coordinates + 4.5 × 1.8 m cars")
    draw(axes[2], roadway, poses, tracks, 3.0, 4.5, 1.8,
         "Active 3× coordinates + 4.5 × 1.8 m cars")
    fig.suptitle("Jounieh curb and the same six recorded vehicle poses", fontsize=13)
    out = CASE / "figures" / "roundabout_scale_comparison_3x"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out.with_suffix(".png"), dpi=200)
    fig.savefig(out.with_suffix(".pdf"))
    plt.close(fig)
    zoom, zoom_ax = plt.subplots(figsize=(8.5, 7.0), constrained_layout=True)
    draw(zoom_ax, roadway, poses, tracks, 3.0, 4.5, 1.8,
         "3× Jounieh roundabout with 4.5 × 1.8 m cars")
    zoom_ax.set_xlim(55, 145)
    zoom_ax.set_ylim(105, -2)
    zoom_path = out.with_name(out.name + "_zoom.png")
    zoom.savefig(zoom_path, dpi=220)
    plt.close(zoom)
    print(out.with_suffix(".png"))
    print(out.with_suffix(".pdf"))
    print(zoom_path)


if __name__ == "__main__":
    main()
