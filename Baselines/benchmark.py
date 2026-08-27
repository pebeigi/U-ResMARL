"""Run every model on identical scenarios and write the comparison table/figures.

    python -m Baselines.benchmark --scenarios 20 --num-agents 12
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

import Baselines._paths  # noqa: F401
from Baselines.metrics import aggregate, metrics_frame, to_latex
from Baselines.registry import (
    DEFAULT_MODELS,
    LABELS,
    LEARNED_CHECKPOINTS,
    build_controller,
    controller_kwargs,
    is_learned,
    resolve_train_seeds,
)
from Baselines.runner import RolloutResult, rollout
from Baselines.scenario import build_scenario
from RL.corridor import DEFAULT_LANE_KF, DEFAULT_RUN_ID

DEFAULT_OUTPUT = Path("Baselines/results")


def run_benchmark(args: argparse.Namespace) -> tuple[pd.DataFrame, dict[str, list[RolloutResult]]]:
    scenario_seeds = list(range(args.seed, args.seed + args.scenarios))
    scenarios = [
        build_scenario(
            seed=s,
            num_agents=args.num_agents,
            max_steps=args.max_steps,
            dt=args.dt,
            run_id=args.run_id,
            lane_kf=args.lane_kf,
            obb_safety_filter=False if args.no_obb_safety_filter else None,
        )
        for s in scenario_seeds
    ]
    print(
        f"Corridor run_id={args.run_id}, lane_kf={args.lane_kf}, "
        f"length={scenarios[0].corridor.length:.1f} m | "
        f"{len(scenarios)} scenarios x {args.num_agents} agents x {args.max_steps} steps"
    )

    results: dict[str, list[RolloutResult]] = {}
    train_seeds: dict[str, list[int]] = {}
    for model in args.models:
        seed_pairs = resolve_train_seeds(model, args.train_seeds, base=_base_checkpoint(model, args))
        if args.train_seeds and is_learned(model) and seed_pairs[0][0] == -1:
            print(f"  [{model}] no per-seed checkpoints found; falling back to a single checkpoint")
        model_results: list[RolloutResult] = []
        model_train_seeds: list[int] = []
        for train_seed, checkpoint in seed_pairs:
            kwargs = controller_kwargs(
                model,
                residual_checkpoint=args.residual_checkpoint,
                pure_rl_checkpoint=args.pure_rl_checkpoint,
                calibration=args.calibration,
                checkpoint_dir=args.checkpoint_dir,
                checkpoint_override=checkpoint if train_seed >= 0 else None,
            )
            controller = build_controller(model, **kwargs)
            for scenario in scenarios:
                model_results.append(rollout(scenario, controller))
                model_train_seeds.append(train_seed)
        results[model] = model_results
        train_seeds[model] = model_train_seeds
        collisions = np.mean([r.collision_events for r in model_results])
        arrivals = np.mean([(r.arrival_step >= 0).mean() for r in model_results])
        seconds = np.sum([r.wall_time for r in model_results])
        n_seeds = len({s for s in model_train_seeds if s >= 0})
        seed_note = f" | {n_seeds} train seeds" if n_seeds > 1 else ""
        print(
            f"  {LABELS.get(model, model):<28} collisions={collisions:6.2f} | "
            f"arrival={arrivals:5.2f} | {seconds:6.1f}s{seed_note}"
        )

    flat = [r for model_results in results.values() for r in model_results]
    frame = metrics_frame(flat)
    frame["train_seed"] = [s for model in results for s in train_seeds[model]]
    return frame, results


def _base_checkpoint(model: str, args: argparse.Namespace) -> Path | None:
    """Checkpoint stem that per-seed filenames are derived from."""
    from Baselines.registry import RESIDUAL_INFERENCE_VARIANTS

    if model in RESIDUAL_INFERENCE_VARIANTS and args.residual_checkpoint is not None:
        return args.residual_checkpoint
    if model == "pure_rl" and args.pure_rl_checkpoint is not None:
        return args.pure_rl_checkpoint
    if model in {"mappo", "happo", "hatrpo"} and args.checkpoint_dir is not None:
        return args.checkpoint_dir / f"{model}_policy.pt"
    return LEARNED_CHECKPOINTS.get(model)


def write_statistics(
    frame: pd.DataFrame,
    args: argparse.Namespace,
    metrics: list[str],
) -> list[Path]:
    """Per-model intervals and paired comparisons against the reference model."""
    from Baselines.stats import comparison_frame, summary_frame, to_latex_ci

    models = list(dict.fromkeys(frame["model"]))
    written: list[Path] = []

    stats = summary_frame(frame, metrics, models=models, n_boot=args.n_boot)
    stats_path = args.output_dir / "benchmark_stats.csv"
    stats.to_csv(stats_path, index=False)
    written.append(stats_path)

    ci_path = args.output_dir / "benchmark_table_ci.tex"
    ci_path.write_text(
        to_latex_ci(frame, metrics, labels=LABELS, models=models, n_boot=args.n_boot),
        encoding="utf-8",
    )
    written.append(ci_path)

    if args.reference_model in models and len(models) > 1:
        comparisons = comparison_frame(
            frame,
            metrics,
            reference=args.reference_model,
            models=models,
            n_boot=args.n_boot,
        )
        cmp_path = args.output_dir / "benchmark_comparisons.csv"
        comparisons.to_csv(cmp_path, index=False)
        written.append(cmp_path)

    n_seeds = frame.loc[frame["train_seed"] >= 0, "train_seed"].nunique()
    if n_seeds <= 1:
        print(
            "[stats] single training seed: intervals reflect scenario variation only. "
            "Train several seeds and pass --train-seeds for training-seed intervals."
        )
    return written


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark baselines against residual MARL")
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    parser.add_argument("--scenarios", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--train-seeds",
        type=int,
        nargs="+",
        default=None,
        help="Training seeds to evaluate for learned models; expects <stem>_seed<k>.pt checkpoints",
    )
    parser.add_argument(
        "--reference-model",
        default="residual_marl",
        help="Model that paired comparisons are computed against",
    )
    parser.add_argument(
        "--n-boot",
        type=int,
        default=10000,
        help="Bootstrap resamples for confidence intervals",
    )
    parser.add_argument("--num-agents", type=int, default=10)
    parser.add_argument("--max-steps", type=int, default=240)
    parser.add_argument("--dt", type=float, default=0.5)
    parser.add_argument("--run-id", type=int, default=DEFAULT_RUN_ID)
    parser.add_argument("--lane-kf", type=int, default=DEFAULT_LANE_KF)
    parser.add_argument("--calibration", type=Path, default=None)
    parser.add_argument("--residual-checkpoint", type=Path, default=None)
    parser.add_argument("--pure-rl-checkpoint", type=Path, default=None)
    parser.add_argument("--checkpoint-dir", type=Path, default=None)
    parser.add_argument(
        "--no-obb-safety-filter",
        action="store_true",
        help="Disable the shared 1.5 s / 4-substep OBB closed-loop filter on all controllers",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--no-realism", action="store_true", help="skip data-distribution metrics")
    parser.add_argument("--no-figures", action="store_true")
    args = parser.parse_args()

    frame, results = run_benchmark(args)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if not args.no_realism:
        try:
            from Baselines.realism import realism_frame

            flat = [r for model_results in results.values() for r in model_results]
            realism = realism_frame(flat)
            frame = frame.merge(realism, on=["model", "seed"], how="left")
        except Exception as exc:  # data file or scipy missing
            print(f"[realism] skipped: {exc}")

    raw_path = args.output_dir / "benchmark_raw.csv"
    summary_path = args.output_dir / "benchmark_summary.csv"
    latex_path = args.output_dir / "benchmark_table.tex"
    frame.to_csv(raw_path, index=False)

    summary_columns = None
    if "realism_score" in frame.columns:
        from Baselines.metrics import AGGREGATE_COLUMNS

        summary_columns = [c for c in AGGREGATE_COLUMNS if c in frame.columns] + ["realism_score"]
    summary = aggregate(frame, summary_columns)
    summary.to_csv(summary_path, index=False)

    latex_columns = [
        "collision_events",
        "offroad_rate",
        "min_ttc_s",
        "arrival_rate",
        "mean_travel_time_s",
        "rms_jerk",
    ]
    if "realism_score" in frame.columns:
        latex_columns.append("realism_score")
    latex_path.write_text(to_latex(frame, latex_columns), encoding="utf-8")

    stats_paths = write_statistics(frame, args, latex_columns)

    if not args.no_figures:
        from Baselines.plots import (
            plot_distribution_comparison,
            plot_metric_bars,
            plot_trajectory_grid,
            plot_trajectory_grid_frenet,
        )

        plot_metric_bars(frame, args.output_dir / "benchmark_metrics.png")
        first_scenario = build_scenario(
            seed=args.seed,
            num_agents=args.num_agents,
            max_steps=args.max_steps,
            dt=args.dt,
            run_id=args.run_id,
            lane_kf=args.lane_kf,
        )
        first_results = [results[m][0] for m in args.models]
        plot_trajectory_grid(
            first_results,
            first_scenario,
            args.output_dir / "benchmark_trajectories.png",
        )
        plot_trajectory_grid_frenet(
            first_results,
            first_scenario,
            args.output_dir / "benchmark_trajectories_frenet.png",
        )
        try:
            flat = [r for model_results in results.values() for r in model_results]
            plot_distribution_comparison(
                flat,
                args.output_dir / "benchmark_distributions.png",
                args.run_id,
                args.lane_kf,
                args.dt,
            )
        except Exception as exc:
            print(f"[distributions] skipped: {exc}")

    with pd.option_context("display.width", 200, "display.max_columns", 50):
        headline = [
            c
            for c in [
                "collision_events",
                "offroad_rate",
                "min_ttc_s",
                "arrival_rate",
                "mean_travel_time_s",
                "rms_jerk",
                "realism_score",
            ]
            if c in frame.columns
        ]
        print("\n" + frame.groupby("model", sort=False)[headline].mean().to_string())
    print(f"\nWrote {raw_path}, {summary_path}, {latex_path}")
    for path in stats_paths:
        print(f"Wrote {path}")


if __name__ == "__main__":
    main()
