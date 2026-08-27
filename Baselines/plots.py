"""Comparison figures for the benchmark."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import Baselines._paths  # noqa: F401
from Baselines.registry import LABELS
from Baselines.runner import RolloutResult
from Baselines.scenario import Scenario
from RL.corridor import oriented_box_corners

PANEL_METRICS = [
    ("collision_events", "Collision events / episode", False),
    ("min_ttc_s", "Minimum TTC (s)", True),
    ("offroad_rate", "Off-corridor rate", False),
    ("arrival_rate", "Arrival rate", True),
    ("mean_speed_mps", "Mean speed (m/s)", True),
    ("rms_jerk", "RMS jerk (m/s^3)", False),
    # Lower = closer to measured Lebanon distributions (mean 1-Wasserstein).
    ("realism_score", "Realism distance to data", False),
]


def _label(model: str) -> str:
    return LABELS.get(model, model)


def plot_metric_bars(frame: pd.DataFrame, output_path: Path) -> None:
    """One bar panel per headline metric, mean +/- std over scenarios."""
    models = list(dict.fromkeys(frame["model"]))
    metrics = [m for m in PANEL_METRICS if m[0] in frame.columns]
    ncols = 3
    nrows = int(np.ceil(len(metrics) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.6 * ncols, 3.4 * nrows))
    axes = np.atleast_1d(axes).ravel()

    colors = plt.cm.tab10(np.linspace(0, 1, max(len(models), 10)))
    for ax, (col, title, higher_better) in zip(axes, metrics):
        means = [frame.loc[frame["model"] == m, col].mean() for m in models]
        stds = [frame.loc[frame["model"] == m, col].std(ddof=0) for m in models]
        ax.bar(range(len(models)), means, yerr=stds, capsize=4, color=colors[: len(models)])
        ax.set_xticks(range(len(models)))
        ax.set_xticklabels([_label(m) for m in models], rotation=30, ha="right", fontsize=8)
        arrow = "higher is better" if higher_better else "lower is better"
        ax.set_title(f"{title}\n({arrow})", fontsize=10)
        ax.grid(axis="y", alpha=0.3)

    for ax in axes[len(metrics) :]:
        ax.axis("off")

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def _draw_corridor(ax, scenario: Scenario) -> None:
    corridor = scenario.corridor
    ax.plot(corridor.lower[:, 0], corridor.lower[:, 1], color="0.25", lw=1.4)
    ax.plot(corridor.upper[:, 0], corridor.upper[:, 1], color="0.25", lw=1.4)
    ax.plot(corridor.center[:, 0], corridor.center[:, 1], color="0.6", lw=0.8, ls="--")
    poly = np.vstack([corridor.lower, corridor.upper[::-1]])
    ax.fill(poly[:, 0], poly[:, 1], color="0.92", zorder=0)


def plot_trajectory_grid(
    results: list[RolloutResult],
    scenario: Scenario,
    output_path: Path,
) -> None:
    """Side-by-side trajectories of every model on the same scenario."""
    n_models = len(results)
    ncols = min(3, n_models)
    nrows = int(np.ceil(n_models / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(6.0 * ncols, 4.0 * nrows), squeeze=False)
    axes_flat = axes.ravel()

    for ax, result in zip(axes_flat, results):
        _draw_corridor(ax, scenario)
        colors = plt.cm.viridis(np.linspace(0, 0.9, result.num_agents))
        steps = result.steps
        for i in range(result.num_agents):
            xy = result.positions[: steps + 1, i]
            ax.plot(xy[:, 0], xy[:, 1], color=colors[i], lw=1.2, alpha=0.9)
            ax.scatter(xy[0, 0], xy[0, 1], color=colors[i], s=18, marker="o", zorder=4)
            corners = oriented_box_corners(
                result.positions[steps, i],
                result.headings[steps, i],
                result.vehicle_length,
                result.vehicle_width,
            )
            ax.fill(corners[:, 0], corners[:, 1], color=colors[i], alpha=0.8, zorder=5)
        ax.set_title(
            f"{_label(result.model)}  |  collisions={result.collision_events}, "
            f"off-corridor steps={result.offroad_steps}",
            fontsize=10,
        )
        ax.set_aspect("equal", adjustable="datalim")
        ax.set_xlabel("x (m)")
        ax.set_ylabel("y (m)")

    for ax in axes_flat[n_models:]:
        ax.axis("off")

    fig.suptitle(
        f"Same scenario, different models (run_id={scenario.run_id}, "
        f"lane_kf={scenario.lane_kf}, seed={scenario.seed})",
        fontsize=12,
    )
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_trajectory_grid_frenet(
    results: list[RolloutResult],
    scenario: Scenario,
    output_path: Path,
) -> None:
    """Same trajectories in corridor coordinates, which makes the corridor readable.

    The corridor is ~590 m long and ~10 m wide, so the (x, y) view compresses all
    the interesting lateral behaviour into a line.
    """
    corridor = scenario.corridor
    half_width = 0.5 * np.linalg.norm(corridor.upper - corridor.lower, axis=1)
    station_axis = corridor.cumulative_s

    n_models = len(results)
    fig, axes = plt.subplots(n_models, 1, figsize=(11.0, 2.4 * n_models), squeeze=False, sharex=True)
    axes_flat = axes.ravel()

    for ax, result in zip(axes_flat, results):
        ax.fill_between(station_axis, -half_width, half_width, color="0.92", zorder=0)
        ax.plot(station_axis, half_width, color="0.25", lw=1.2)
        ax.plot(station_axis, -half_width, color="0.25", lw=1.2)
        ax.axhline(0.0, color="0.6", lw=0.8, ls="--")

        colors = plt.cm.viridis(np.linspace(0, 0.9, result.num_agents))
        steps = result.steps
        for i in range(result.num_agents):
            s = result.station[: steps + 1, i]
            lat = result.lateral[: steps + 1, i]
            ax.plot(s, lat, color=colors[i], lw=1.1, alpha=0.9)
            ax.scatter(s[0], lat[0], color=colors[i], s=16, marker="o", zorder=4)
            ax.scatter(result.dest_s[i], 0.0, color=colors[i], s=30, marker="*", zorder=4)
        arrived = int((result.arrival_step >= 0).sum())
        ax.set_title(
            f"{_label(result.model)}  |  arrived {arrived}/{result.num_agents}, "
            f"collisions={result.collision_events}",
            fontsize=10,
        )
        ax.set_ylabel("lateral (m)")
        ax.set_xlim(0, float(corridor.length))

    axes_flat[-1].set_xlabel("along-corridor station s (m)")
    fig.suptitle(
        f"Corridor coordinates, same scenario (run_id={scenario.run_id}, "
        f"lane_kf={scenario.lane_kf}, seed={scenario.seed})",
        fontsize=12,
    )
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_distribution_comparison(
    results: list[RolloutResult],
    output_path: Path,
    run_id: int,
    lane_kf: int,
    dt: float,
) -> None:
    """Simulated vs measured speed / acceleration / lateral-offset distributions."""
    from Baselines.realism import FEATURES, observed_features, simulated_features

    observed = observed_features(run_id, lane_kf, dt)
    by_model: dict[str, dict[str, list]] = {}
    for r in results:
        feats = simulated_features(r)
        bucket = by_model.setdefault(r.model, {k: [] for k in FEATURES})
        for k in FEATURES:
            bucket[k].append(feats[k])

    titles = {
        "speed": "Speed (m/s)",
        "accel": "Longitudinal acceleration (m/s^2)",
        "lateral": "Lateral offset from centreline (m)",
    }
    fig, axes = plt.subplots(1, len(FEATURES), figsize=(5.2 * len(FEATURES), 3.8))
    for ax, key in zip(np.atleast_1d(axes), FEATURES):
        ax.hist(
            observed[key],
            bins=40,
            density=True,
            histtype="stepfilled",
            color="0.7",
            alpha=0.7,
            label="Measured data",
        )
        for model, bucket in by_model.items():
            values = np.concatenate([v for v in bucket[key] if v.size]) if bucket[key] else np.array([])
            if values.size == 0:
                continue
            ax.hist(values, bins=40, density=True, histtype="step", lw=1.6, label=_label(model))
        ax.set_title(titles[key], fontsize=10)
        ax.grid(alpha=0.3)
    np.atleast_1d(axes)[0].legend(fontsize=8)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
