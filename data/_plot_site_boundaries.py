#!/usr/bin/env python
"""One boundary + trajectories plot per site (Jounieh, TGSIM).

Uses the approved site boundaries:
  Jounieh — Jounieh_Road_Boundaries.csv (outer + islands)
  TGSIM   — derived_boundaries/street_boundaries.csv (curb union)
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "_qa_plots"
OUT.mkdir(parents=True, exist_ok=True)


def _sample(df: pd.DataFrame, n: int = 100_000) -> pd.DataFrame:
    return df if len(df) <= n else df.sample(n, random_state=0)


def plot_jounieh() -> Path:
    bound = pd.read_csv(ROOT / "Lebanon_Jounieh" / "Jounieh_Road_Boundaries.csv")
    traj = pd.read_csv(ROOT / "Lebanon_Jounieh" / "prepared" / "trajectories_calibration.csv")

    fig, ax = plt.subplots(figsize=(9, 7), constrained_layout=True)
    s = _sample(traj)
    ax.scatter(s["xloc_kf"], s["yloc_kf"], s=1.2, alpha=0.12, color="0.35", label="trajectories")
    for (kind, pid), g in bound.groupby(["kind", "polygon_id"], sort=True):
        g = g.sort_values("vertex_index")
        xy = g[["x", "y"]].to_numpy(float)
        ax.plot(
            np.r_[xy[:, 0], xy[0, 0]],
            np.r_[xy[:, 1], xy[0, 1]],
            lw=2.5 if kind == "outer" else 2.0,
            color="C0" if kind == "outer" else "C3",
            label=f"{kind} {pid}",
        )
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_title("Jounieh — provided boundaries + trajectories")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", markerscale=5)
    out = OUT / "jounieh_boundaries_with_traj.png"
    fig.savefig(out, dpi=170)
    plt.close(fig)
    return out


def plot_tgsim() -> Path:
    bound = pd.read_csv(ROOT / "TGSIM FB" / "derived_boundaries" / "street_boundaries.csv")
    traj = pd.read_csv(ROOT / "TGSIM FB" / "prepared" / "trajectories_calibration.csv")
    cars = traj[traj["class"] == 3.0] if "class" in traj.columns else traj

    fig, ax = plt.subplots(figsize=(9, 11), constrained_layout=True)
    s = _sample(cars, 120_000)
    ax.scatter(s["xloc_kf"], s["yloc_kf"], s=1.0, alpha=0.08, color="0.35", label="class=3 traj")
    for _, part in bound.groupby("part_index"):
        xy = part.sort_values("vertex_index")[["x", "y"]].to_numpy(float)
        ax.plot(xy[:, 0], xy[:, 1], lw=2.0, color="C0")
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_title("TGSIM Foggy Bottom — curb boundaries + trajectories")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", markerscale=6)
    out = OUT / "tgsim_boundaries_with_traj.png"
    fig.savefig(out, dpi=170)
    plt.close(fig)
    return out


if __name__ == "__main__":
    j = plot_jounieh()
    t = plot_tgsim()
    print("wrote", j)
    print("wrote", t)
