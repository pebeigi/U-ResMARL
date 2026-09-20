"""One shared TGSIM scene: utility prior vs residual, plotted on the curb polygon.

    python "TGSIM Case/plot_trajectories.py"
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
from config import CALIBRATION, CHECKPOINT_DIR, NUM_AGENTS  # noqa: E402
from RL.corridor import oriented_box_corners  # noqa: E402

OUT_DIR = CASE_ROOT / "figures"
COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]


def _draw_curb(ax, roadway) -> None:
    exterior = np.asarray(roadway.exterior.coords, dtype=float)
    ax.add_patch(
        MplPolygon(exterior, closed=True, facecolor="#d6e4f0", edgecolor="#1f4e79", lw=1.6, zorder=0)
    )
    for hole in roadway.interiors:
        ring = np.asarray(hole.coords, dtype=float)
        ax.add_patch(
            MplPolygon(ring, closed=True, facecolor="white", edgecolor="#1f4e79", lw=1.2, zorder=1)
        )


def _draw_box(ax, pos, heading, color, length, width, alpha=0.9, zorder=5):
    corners = oriented_box_corners(pos, heading, length=length, width=width)
    ax.add_patch(
        MplPolygon(corners, closed=True, facecolor=color, edgecolor="k", lw=0.6, alpha=alpha, zorder=zorder)
    )


def _latest_residual_checkpoint() -> Path:
    export = CHECKPOINT_DIR / "residual_policy_seed0.pt"
    resume = CHECKPOINT_DIR / "residual_policy_seed0.resume.pt"
    if not export.is_file():
        raise FileNotFoundError(f"Missing residual checkpoint: {export}")
    if resume.is_file():
        import torch
        from tempfile import NamedTemporaryFile

        blob = torch.load(export, map_location="cpu")
        live = torch.load(resume, map_location="cpu")
        if "state_dict" in live:
            blob["state_dict"] = live["state_dict"]
            blob["selection_status"] = f"live_update_{live.get('update', '?')}"
        tmp = CASE_ROOT / "checkpoints" / "_plot_live_residual.pt"
        torch.save(blob, tmp)
        return tmp
    return export


def _panel(ax, result, scenario, title: str) -> None:
    roadway = scenario.corridor.roadway
    _draw_curb(ax, roadway)
    n = result.num_agents
    length, width = result.vehicle_length, result.vehicle_width
    for i in range(n):
        color = COLORS[i % len(COLORS)]
        xy = result.positions[:, i]
        ax.plot(xy[:, 0], xy[:, 1], color=color, lw=1.8, alpha=0.9, zorder=3)
        ax.plot(xy[0, 0], xy[0, 1], "o", color=color, ms=5, markeredgecolor="k", zorder=6)
        dest = np.asarray(scenario.agents[i].dest, dtype=float)
        ax.plot(dest[0], dest[1], "*", color=color, ms=12, markeredgecolor="k", zorder=6)
        _draw_box(ax, xy[-1], result.headings[-1, i], color, length, width)
    minx, miny, maxx, maxy = roadway.bounds
    ax.set_xlim(minx - 8, maxx + 8)
    ax.set_ylim(miny - 8, maxy + 8)
    ax.set_aspect("equal", adjustable="box")
    ax.set_title(title, fontsize=11)
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
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
    from Baselines.residual_marl import ResidualMARLController
    from Baselines.runner import rollout
    from Baselines.scenario import build_scenario
    from Baselines.utility_prior import UtilityPriorController

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    scenario = build_scenario(seed=0, num_agents=NUM_AGENTS, max_steps=80)
    prior = UtilityPriorController(calibration=CALIBRATION, prefer="robust")
    ckpt = _latest_residual_checkpoint()
    residual = ResidualMARLController(checkpoint=ckpt, calibration=CALIBRATION, prefer="robust")

    print("Rolling out utility prior...", flush=True)
    prior_result = rollout(scenario, prior)
    print(
        f"  prior: collisions={prior_result.collision_events} "
        f"offroad={prior_result.offroad_steps} steps={prior_result.steps}",
        flush=True,
    )
    print("Rolling out residual MARL...", flush=True)
    residual_result = rollout(scenario, residual)
    print(
        f"  residual: collisions={residual_result.collision_events} "
        f"offroad={residual_result.offroad_steps} steps={residual_result.steps}",
        flush=True,
    )

    fig, axes = plt.subplots(1, 2, figsize=(14, 7), constrained_layout=True)
    _panel(axes[0], prior_result, scenario, "Utility prior (TGSIM calibration)")
    _panel(axes[1], residual_result, scenario, "Residual MARL (current TGSIM policy)")
    handles = [
        Line2D([0], [0], color="#1f4e79", lw=1.6, label="Curb / block hole"),
        Line2D([0], [0], marker="o", color="k", lw=0, markersize=6, label="Start"),
        Line2D([0], [0], marker="*", color="k", lw=0, markersize=10, label="Destination"),
    ]
    fig.legend(handles=handles, loc="upper center", ncol=3, frameon=False)
    fig.suptitle("TGSIM Foggy Bottom — same spawn, extracted network curb", fontsize=13, y=1.03)
    out = OUT_DIR / "tgsim_prior_vs_residual.png"
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out}", flush=True)


if __name__ == "__main__":
    main()
