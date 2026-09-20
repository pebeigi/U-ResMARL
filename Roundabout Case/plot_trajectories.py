"""One shared Jounieh scene: utility prior vs residual on the roundabout curb.

    python "Roundabout Case/plot_trajectories.py"
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Polygon as MplPolygon

CASE_ROOT = Path(__file__).resolve().parent
REPO_ROOT = CASE_ROOT.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(CASE_ROOT))

import activate  # noqa: E402
from config import CALIBRATION, NUM_AGENTS, TRAJECTORIES_CSV  # noqa: E402
from RL.corridor import oriented_box_corners  # noqa: E402

OUT_DIR = CASE_ROOT / "figures"
COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]


def _draw_curb(ax, roadway, recorded_xy=None) -> None:
    exterior = np.asarray(roadway.exterior.coords, dtype=float)
    ax.add_patch(MplPolygon(exterior, closed=True, facecolor="#d6e4f0", edgecolor="#1f4e79", lw=1.6, zorder=0))
    for hole in roadway.interiors:
        ring = np.asarray(hole.coords, dtype=float)
        ax.add_patch(MplPolygon(ring, closed=True, facecolor="#f4c7c3", edgecolor="#1f4e79", lw=1.2, zorder=1))
    if recorded_xy is not None and len(recorded_xy):
        ax.plot(recorded_xy[:, 0], recorded_xy[:, 1], ",", color="0.35", alpha=0.12, zorder=1.5)


def _draw_box(ax, pos, heading, color, length, width, alpha=0.9, zorder=5):
    corners = oriented_box_corners(pos, heading, length=length, width=width)
    ax.add_patch(MplPolygon(corners, closed=True, facecolor=color, edgecolor="k", lw=0.6, alpha=alpha, zorder=zorder))


def _panel(ax, result, scenario, title: str, recorded_xy=None) -> None:
    roadway = scenario.corridor.roadway
    _draw_curb(ax, roadway, recorded_xy=recorded_xy)
    n = result.num_agents
    length, width = result.vehicle_length, result.vehicle_width
    agents = scenario.spawn_agents()
    for i in range(n):
        color = COLORS[i % len(COLORS)]
        try:
            route = scenario.corridor.for_agent(agents[i])
            ax.plot(route.center[:, 0], route.center[:, 1], color=color, lw=0.9, ls="--", alpha=0.35, zorder=2)
        except Exception:
            pass
        xy = result.positions[:, i]
        ax.plot(xy[:, 0], xy[:, 1], color=color, lw=1.8, alpha=0.9, zorder=3)
        ax.plot(xy[0, 0], xy[0, 1], "o", color=color, ms=5, markeredgecolor="k", zorder=6)
        dest = np.asarray(scenario.agents[i].dest, dtype=float)
        ax.plot(dest[0], dest[1], "*", color=color, ms=12, markeredgecolor="k", zorder=6)
        _draw_box(ax, xy[0], result.headings[0, i], color, length, width, alpha=0.35, zorder=4)
        _draw_box(ax, xy[-1], result.headings[-1, i], color, length, width)
    minx, miny, maxx, maxy = roadway.bounds
    ax.set_xlim(minx - 3, maxx + 3)
    ax.set_ylim(miny - 3, maxy + 3)
    ax.set_aspect("equal", adjustable="box")
    ax.set_title(title, fontsize=11)
    ax.set_xticks([])
    ax.set_yticks([])
    arrived = int(np.sum(result.arrival_step >= 0)) if len(result.arrival_step) else 0
    ax.text(
        0.02,
        0.02,
        f"collisions={result.collision_events}  off-road={result.offroad_steps}  "
        f"arrived={arrived}/{n}  steps={result.steps}",
        transform=ax.transAxes,
        fontsize=8,
        va="bottom",
        bbox=dict(boxstyle="round,pad=0.25", facecolor="white", alpha=0.85),
    )


def main() -> None:
    activate.apply()
    from Baselines.runner import rollout
    from Baselines.scenario import build_scenario
    from Baselines.utility_prior import UtilityPriorController

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    prior = UtilityPriorController(calibration=CALIBRATION, prefer="robust")
    seeds = []
    panels = []
    for seed in range(24):
        if len(panels) >= 4:
            break
        scenario = build_scenario(seed=seed, num_agents=NUM_AGENTS, max_steps=80)
        print(f"Rolling out utility prior seed={seed}...", flush=True)
        try:
            result = rollout(scenario, prior)
        except Exception as exc:
            print(f"  skip seed={seed}: {type(exc).__name__}: {exc}", flush=True)
            continue
        arrived = int(np.sum(result.arrival_step >= 0))
        print(
            f"  collisions={result.collision_events} offroad={result.offroad_steps} "
            f"arrived={arrived}/{result.num_agents} steps={result.steps}",
            flush=True,
        )
        seeds.append(seed)
        panels.append((result, scenario, seed))
    if len(panels) < 4:
        raise RuntimeError(f"Only {len(panels)} closed-loop scenes completed")
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.2), constrained_layout=True)
    recorded_xy = None
    if TRAJECTORIES_CSV.is_file():
        import pandas as pd

        raw = pd.read_csv(TRAJECTORIES_CSV, usecols=["xloc_kf", "yloc_kf"])
        recorded_xy = raw.sample(n=min(12000, len(raw)), random_state=0)[["xloc_kf", "yloc_kf"]].to_numpy(float)
    handles = [
        Line2D([0], [0], color="#1f4e79", lw=1.6, label="Extracted curb / island"),
        Line2D([0], [0], color="0.35", lw=0, marker=",", label="Recorded tracks"),
        Line2D([0], [0], color="#64748b", lw=1.0, ls="--", label="Planned route"),
        Line2D([0], [0], marker="o", color="k", lw=0, markersize=6, label="Start"),
        Line2D([0], [0], marker="*", color="k", lw=0, markersize=10, label="Destination"),
    ]
    for ax, (result, scenario, seed) in zip(axes.ravel(), panels):
        _panel(ax, result, scenario, f"Utility prior  ·  seed {seed}", recorded_xy=recorded_xy)
    fig.legend(handles=handles, loc="upper center", ncol=5, frameon=False, bbox_to_anchor=(0.5, 1.03))
    fig.suptitle("Jounieh roundabout — closed-loop utility prior on extracted curb", fontsize=13, y=1.06)
    out = OUT_DIR / "roundabout_utility_prior.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out}", flush=True)


if __name__ == "__main__":
    main()
