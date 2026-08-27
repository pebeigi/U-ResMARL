"""Paper-facing figures from existing rollouts / fresh stress comparisons.

    # Regenerate metric bars (with realism) + realism distributions from benchmark_raw
    python -m Baselines.paper_figures --metrics --realism-panel

    # Qualitative Frenet: prior vs residual vs MAPPO on one dense stress scenario
    python -m Baselines.paper_figures --stress-frenet

    python -m Baselines.paper_figures --all
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import Baselines._paths  # noqa: F401
from Baselines.ablation_stress import STRESS_SPAWN
from Baselines.plots import (
    PANEL_METRICS,
    plot_distribution_comparison,
    plot_metric_bars,
    plot_trajectory_grid_frenet,
)
from Baselines.registry import LABELS, build_controller, controller_kwargs
from Baselines.runner import rollout
from Baselines.scenario import build_scenario
from RL.corridor import DEFAULT_LANE_KF, DEFAULT_RUN_ID

DEFAULT_OUTPUT = Path("Baselines/results/paper")
BENCHMARK_RAW = Path("Baselines/results/benchmark_raw.csv")

# Clean main-table models for the paper figure (no reward-exploit / unstable MARL).
PAPER_MODELS = [
    "orca",
    "social_force",
    "dwa",
    "mppi",
    "mappo",
    "utility_pt",
    "residual_marl",
]

STRESS_COMPARE = ["utility_pt", "residual_marl", "mappo"]


def _label(model: str) -> str:
    return LABELS.get(model, model)


def make_metrics_figure(raw_path: Path, output_dir: Path, models: list[str] | None = None) -> Path:
    frame = pd.read_csv(raw_path)
    if models is not None:
        frame = frame[frame["model"].isin(models)].copy()
        # Preserve paper order.
        frame["model"] = pd.Categorical(frame["model"], categories=models, ordered=True)
        frame = frame.sort_values(["model", "seed"])
    out = output_dir / "paper_metrics.png"
    plot_metric_bars(frame, out)
    print(f"Wrote {out}")
    if "realism_score" in frame.columns:
        summary = (
            frame.groupby("model", sort=False)["realism_score"].agg(["mean", "std"]).round(3)
        )
        print("Realism distance (lower = more like measured data):\n", summary.to_string())
    return out


def make_realism_panel(
    raw_results_needed: bool,
    output_dir: Path,
    models: list[str],
    scenarios: int,
    seed: int,
    num_agents: int,
    max_steps: int,
    run_id: int,
    lane_kf: int,
) -> Path:
    """Focused speed / accel / lateral histograms for the paper's key models."""
    from Baselines.realism import FEATURES, observed_features, simulated_features

    # Re-roll a few scenarios so the panel matches current checkpoints.
    seeds = list(range(seed, seed + scenarios))
    scenario_list = [
        build_scenario(
            seed=s,
            num_agents=num_agents,
            max_steps=max_steps,
            run_id=run_id,
            lane_kf=lane_kf,
        )
        for s in seeds
    ]
    results = []
    for model in models:
        controller = build_controller(model, **controller_kwargs(model))
        for scenario in scenario_list:
            results.append(rollout(scenario, controller))
        print(f"  realism rollouts: {_label(model)} done")

    observed = observed_features(run_id, lane_kf, scenario_list[0].dt)
    by_model: dict[str, dict[str, list]] = {}
    for r in results:
        feats = simulated_features(r)
        bucket = by_model.setdefault(r.model, {k: [] for k in FEATURES})
        for k in FEATURES:
            bucket[k].append(feats[k])

    titles = {
        "speed": "Speed (m/s)",
        "accel": "Longitudinal acceleration (m/s^2)",
        "lateral": "Lateral offset (m)",
    }
    # Distinct colors for the three paper models of interest.
    color_map = {
        "utility_pt": "#4C78A8",
        "residual_marl": "#F58518",
        "mappo": "#E45756",
        "social_force": "#54A24B",
    }

    fig, axes = plt.subplots(1, len(FEATURES), figsize=(5.0 * len(FEATURES), 3.6))
    for ax, key in zip(np.atleast_1d(axes), FEATURES):
        ax.hist(
            observed[key],
            bins=40,
            density=True,
            histtype="stepfilled",
            color="0.75",
            alpha=0.75,
            label="Measured data",
        )
        for model in models:
            bucket = by_model.get(model, {})
            values = (
                np.concatenate([v for v in bucket.get(key, []) if v.size])
                if bucket.get(key)
                else np.array([])
            )
            if values.size == 0:
                continue
            ax.hist(
                values,
                bins=40,
                density=True,
                histtype="step",
                lw=2.0,
                color=color_map.get(model),
                label=_label(model),
            )
        ax.set_title(titles[key], fontsize=11)
        ax.grid(alpha=0.3)
        ax.set_ylabel("density")
    np.atleast_1d(axes)[0].legend(fontsize=8, loc="best")
    fig.suptitle(
        "Behavioural realism: simulated vs measured distributions "
        f"(run_id={run_id}, lane_kf={lane_kf})",
        fontsize=12,
    )
    fig.tight_layout()
    out = output_dir / "paper_realism_panel.png"
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=220)
    plt.close(fig)
    print(f"Wrote {out}")
    # Also keep the generic multi-model distribution helper for completeness.
    plot_distribution_comparison(
        results,
        output_dir / "paper_distributions.png",
        run_id,
        lane_kf,
        scenario_list[0].dt,
    )
    return out


def make_stress_frenet(
    output_dir: Path,
    seed: int,
    stress_agents: int,
    max_steps: int,
    run_id: int,
    lane_kf: int,
    models: list[str],
) -> Path:
    """Side-by-side Frenet trajectories under dense traffic for the paper narrative."""
    scenario = build_scenario(
        seed=seed,
        num_agents=stress_agents,
        max_steps=max_steps,
        run_id=run_id,
        lane_kf=lane_kf,
        **STRESS_SPAWN,
    )
    results = []
    for model in models:
        controller = build_controller(model, **controller_kwargs(model))
        result = rollout(scenario, controller)
        results.append(result)
        arrived = int((result.arrival_step >= 0).sum())
        print(
            f"  stress seed={seed} {_label(model)}: "
            f"arrived {arrived}/{result.num_agents}, collisions={result.collision_events}"
        )

    out = output_dir / "paper_stress_frenet.png"
    plot_trajectory_grid_frenet(results, scenario, out)
    print(f"Wrote {out}")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Build paper figures")
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--metrics", action="store_true", help="metric bars with realism panel")
    parser.add_argument(
        "--realism-panel",
        action="store_true",
        help="histogram panel: measured vs utility / residual / MAPPO",
    )
    parser.add_argument(
        "--stress-frenet",
        action="store_true",
        help="qualitative Frenet under dense stress for prior / residual / MAPPO",
    )
    parser.add_argument("--raw", type=Path, default=BENCHMARK_RAW)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--scenarios", type=int, default=5, help="scenarios for realism rollouts")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-agents", type=int, default=10)
    parser.add_argument("--stress-agents", type=int, default=18)
    parser.add_argument("--max-steps", type=int, default=300)
    parser.add_argument("--run-id", type=int, default=DEFAULT_RUN_ID)
    parser.add_argument("--lane-kf", type=int, default=DEFAULT_LANE_KF)
    parser.add_argument(
        "--stress-seed",
        type=int,
        default=0,
        help="scenario seed for the qualitative stress Frenet figure",
    )
    args = parser.parse_args()
    if args.all:
        args.metrics = args.realism_panel = args.stress_frenet = True
    if not (args.metrics or args.realism_panel or args.stress_frenet):
        parser.error("Pick at least one of --metrics / --realism-panel / --stress-frenet / --all")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.metrics:
        if not args.raw.exists():
            raise SystemExit(f"Missing {args.raw}; run Baselines.benchmark first.")
        make_metrics_figure(args.raw, args.output_dir, PAPER_MODELS)

    if args.realism_panel:
        make_realism_panel(
            True,
            args.output_dir,
            models=["utility_pt", "residual_marl", "mappo"],
            scenarios=args.scenarios,
            seed=args.seed,
            num_agents=args.num_agents,
            max_steps=args.max_steps,
            run_id=args.run_id,
            lane_kf=args.lane_kf,
        )

    if args.stress_frenet:
        make_stress_frenet(
            args.output_dir,
            seed=args.stress_seed,
            stress_agents=args.stress_agents,
            max_steps=args.max_steps,
            run_id=args.run_id,
            lane_kf=args.lane_kf,
            models=STRESS_COMPARE,
        )


if __name__ == "__main__":
    main()
