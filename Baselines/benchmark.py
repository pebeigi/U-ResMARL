"""Run every model on identical scenarios and write the comparison table/figures.

    python -m Baselines.benchmark --scenarios 20 --num-agents 12
"""

from __future__ import annotations

import argparse
import json
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

DEFAULT_OUTPUT = Path("Baselines/results/revision5")


def preflight_models(models, args, scenario):
    """Load every requested checkpoint before spending time on rollouts."""
    for model in models:
        for train_seed, checkpoint in resolve_train_seeds(
            model, args.train_seeds, base=_base_checkpoint(model, args)
        ):
            controller = build_controller(model, **controller_kwargs(
                model, residual_checkpoint=args.residual_checkpoint,
                pure_rl_checkpoint=getattr(args, "pure_rl_checkpoint", None),
                calibration=args.calibration, checkpoint_dir=args.checkpoint_dir,
                checkpoint_override=checkpoint if train_seed >= 0 else None,
                train_seed=train_seed,
            ))
            controller.reset(scenario)


def _attach_realism_metrics(frame: pd.DataFrame, realism: pd.DataFrame) -> pd.DataFrame:
    """Attach rollout-level realism metrics without a many-to-many key merge.

    ``model`` and scenario ``seed`` are not unique when several training seeds
    are evaluated.  ``realism_frame`` receives the same flattened rollout list
    as ``metrics_frame``, so rows must be joined by that shared order.
    """
    left = frame.reset_index(drop=True)
    right = realism.reset_index(drop=True)
    if len(left) != len(right):
        raise ValueError(
            f"Realism row count ({len(right)}) does not match benchmark row count ({len(left)})"
        )
    keys = ["model", "seed"]
    if not left[keys].equals(right[keys]):
        raise ValueError("Realism rows are not aligned with benchmark rollout order")
    overlap = (set(left.columns) & set(right.columns)) - set(keys)
    if overlap:
        raise ValueError(f"Duplicate realism metric columns: {sorted(overlap)}")
    return pd.concat([left, right.drop(columns=keys)], axis=1)


def run_benchmark(args: argparse.Namespace) -> tuple[pd.DataFrame, dict[str, list[RolloutResult]]]:
    data_mode = getattr(args, "data_evaluation", False)
    recorded = {}
    if data_mode:
        from Baselines.data_evaluation import build_recorded_scenes, file_sha256
        from Baselines.realism import DEFAULT_DATA_CSV, paired_realism_metrics
        from Baselines.data_evaluation import trajectory_metrics
        from RL.calibration_io import DEFAULT_CALIBRATION_PATH

        data_scenes, manifest = build_recorded_scenes(
            getattr(args, "data_csv", None) or DEFAULT_DATA_CSV,
            count=args.scenarios, seed=args.seed, run_id=args.run_id, lane_kf=args.lane_kf,
            dt=args.dt, horizons=args.data_horizons, split=args.data_split,
            obb_safety_filter=not args.no_obb_safety_filter)
        recorded = {item.scenario.seed: item for item in data_scenes}
        scenarios = [item.scenario for item in data_scenes]
        calibration = args.calibration or DEFAULT_CALIBRATION_PATH
        manifest["calibration"] = {"path": str(Path(calibration).resolve()),
                                   "sha256": file_sha256(calibration)}
        manifest["checkpoints"] = []
        for model in args.models:
            for train_seed, checkpoint in resolve_train_seeds(
                    model, args.train_seeds, base=_base_checkpoint(model, args)):
                if is_learned(model) and checkpoint is not None:
                    manifest["checkpoints"].append({
                        "model": model, "train_seed": train_seed, "path": str(checkpoint.resolve()),
                        "sha256": file_sha256(checkpoint)})
        manifest["metric_processing"] = {
            "kinematics": "backward position differences at dt for both traces; "
                          "heading held below 0.05 m/s; wrapped angular differences",
            "coverage": "ADE/FDE require full observed horizon; realism uses common "
                        "observed agent/time coverage, including model arrivals and contacts",
            "realism": "per-scene W1 and natural-log JS; reference-defined shared bins with tail bins",
            "interaction_geometry": "two covering discs per shared vehicle footprint",
            "gap_cap_m": 60., "ttc_cap_s": 10.,
            "realism_score": "mean normalized W1 across six features, lower is better; diagnostic",
            "normalization_std_floors": dict(speed=.1, accel=.1, lateral=.1, yaw_rate=.01, gap=1., ttc=1.),
        }
        args._data_scenes, args._data_manifest = data_scenes, manifest
        args.output_dir.mkdir(parents=True, exist_ok=True)
        (args.output_dir / "data_manifest.json").write_text(
            json.dumps(manifest, indent=2, allow_nan=False), encoding="utf-8")
        print(f"[data] {manifest['holdout_status']}: {manifest['holdout_note']}", flush=True)
    else:
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
            for s in range(args.seed, args.seed + args.scenarios)
        ]
    print(
        f"Corridor run_id={args.run_id}, lane_kf={args.lane_kf}, "
        f"length={scenarios[0].corridor.length:.1f} m | "
        f"{len(scenarios)} scenarios | agents={min(s.num_agents for s in scenarios)}"
        f"..{max(s.num_agents for s in scenarios)} | {scenarios[0].max_steps} steps",
        flush=True,
    )

    preflight_models(args.models, args, scenarios[0])
    results: dict[str, list[RolloutResult]] = {}
    train_seeds: dict[str, list[int]] = {}
    for model in args.models:
        seed_pairs = resolve_train_seeds(model, args.train_seeds, base=_base_checkpoint(model, args))
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
                train_seed=train_seed,
            )
            controller = build_controller(model, **kwargs)
            for scenario in scenarios:
                result = rollout(scenario, controller, stop_when_all_arrived=not data_mode)
                if data_mode:
                    reference = recorded[scenario.seed]
                    scores = trajectory_metrics(result, reference, args.data_horizons)
                    if not args.no_realism:
                        scores.update(paired_realism_metrics(result, reference))
                    scores.update(
                        data_split=args.data_split,
                        data_time_block=reference.metadata["time_block"],
                        data_scene_start_s=reference.metadata["start_time_s"],
                        initial_overlap_pairs=reference.metadata["initial_overlap_pairs"],
                        initial_excluded_agents=len(reference.metadata["excluded_initial_agents"]),
                        holdout_status=args._data_manifest["holdout_status"],
                    )
                    result.extra["data_metrics"] = scores
                    print(f"    {model}: recorded scene {scenario.seed} "
                          f"({scenario.num_agents} agents) evaluated", flush=True)
                model_results.append(result)
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
    if data_mode:
        data_frame = pd.DataFrame([dict(model=r.model, seed=r.seed, **r.extra["data_metrics"])
                                   for r in flat])
        frame = _attach_realism_metrics(frame, data_frame)
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

    if "data_time_block" in frame:
        # Nearby windows share traffic. Bootstrap block means, not frames/cars
        # or overlapping traffic windows as independent samples.
        frame = frame.groupby(["model", "train_seed", "data_time_block"], as_index=False)[metrics].mean()
        frame = frame.rename(columns={"data_time_block": "seed"})
        print("[stats] recorded-scene intervals resample time blocks (and training seeds when available)")
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
    parser.add_argument("--data-evaluation", action="store_true",
                        help="Evaluate recorded scenes with matched trajectory/realism metrics; "
                             "--num-agents and --max-steps apply only to random scenarios")
    parser.add_argument("--data-csv", type=Path, default=None)
    parser.add_argument("--data-split", choices=["train", "validation", "test"], default="test")
    parser.add_argument("--data-horizons", nargs="+", type=float, default=[1., 3., 5.])
    parser.add_argument("--no-figures", action="store_true")
    args = parser.parse_args()
    if args.scenarios < 1 or args.dt <= 0:
        parser.error("--scenarios and --dt must be positive")
    if args.data_csv is not None and not args.data_evaluation:
        parser.error("--data-csv requires --data-evaluation")

    frame, results = run_benchmark(args)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if not args.no_realism and not args.data_evaluation:
        try:
            from Baselines.realism import realism_frame

            flat = [r for model_results in results.values() for r in model_results]
            realism = realism_frame(flat)
            frame = _attach_realism_metrics(frame, realism)
        except Exception as exc:  # data file or scipy missing
            print(f"[realism] skipped: {exc}")

    raw_path = args.output_dir / "benchmark_raw.csv"
    summary_path = args.output_dir / "benchmark_summary.csv"
    latex_path = args.output_dir / "benchmark_table.tex"
    frame.to_csv(raw_path, index=False)

    from Baselines.metrics import AGGREGATE_COLUMNS
    data_columns = [c for c in frame if c.startswith((
        "ade_", "fde_", "speed_rmse_", "heading_mae_", "trajectory_", "heading_samples_",
        "w1_", "js_", "realism_samples_")) or c == "realism_score"]
    summary_columns = [c for c in AGGREGATE_COLUMNS if c in frame] + data_columns
    summary = aggregate(frame, summary_columns)
    summary.to_csv(summary_path, index=False)

    latex_columns = [
        "closed_loop_score",
        "collision_events",
        "offroad_rate",
        "min_ttc_s",
        "arrival_rate",
        "mean_travel_time_s",
        "rms_jerk",
    ]
    if "realism_score" in frame.columns:
        latex_columns.append("realism_score")
    if args.data_evaluation:
        latex_columns += [c for c in data_columns if c.startswith(("ade_", "fde_"))]
    latex_path.write_text(to_latex(frame, latex_columns), encoding="utf-8")

    statistical_metrics = list(dict.fromkeys(latex_columns + [
        c for c in data_columns if c.startswith(("speed_rmse_", "heading_mae_", "w1_", "js_"))]))
    stats_paths = write_statistics(frame, args, statistical_metrics)

    if not args.no_figures:
        from Baselines.plots import (
            plot_distribution_comparison,
            plot_metric_bars,
            plot_trajectory_grid,
            plot_trajectory_grid_frenet,
        )

        plot_metric_bars(frame, args.output_dir / "benchmark_metrics.png")
        first_scenario = args._data_scenes[0].scenario if args.data_evaluation else build_scenario(
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
        if args.data_evaluation:
            from Baselines.plots import plot_recorded_comparison
            plot_recorded_comparison(results, args._data_scenes, args.output_dir,
                                     realism=not args.no_realism)
        elif not args.no_realism:
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
    if args.data_evaluation:
        accuracy = [c for c in frame if c.startswith(("ade_", "fde_", "speed_rmse_",
                                                      "heading_mae_", "trajectory_coverage_"))]
        print("\nRecorded trajectory metrics:\n" + frame.groupby("model")[accuracy].mean().to_string())
        print(f"Wrote {args.output_dir / 'data_manifest.json'}")
    for path in stats_paths:
        print(f"Wrote {path}")


if __name__ == "__main__":
    main()
