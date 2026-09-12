"""Paper ablations and denser stress tests on matched seeds.

Minimum ICLR/CoRL ablation package (see ``Baselines.ablation_models``):

  * utility prior (calibrated + nominal)
  * full residual MARL
  * weights-only residual (Δσ frozen)
  * σ-only residual (weight residuals frozen)
  * direct discrete RL + MAPPO (matched OBB safety layer)
  * residual on nominal prior (calibration ablation)
  * full vs. no-lookahead OBB conflict horizon (1.5 s / 4 substeps vs. 1 step)

Default evaluation uses 30 matched scenarios and 3 RL training seeds with paired
bootstrap confidence intervals.

    python -m Baselines.ablation_stress
    python -m Baselines.ablation_stress --mode ablation --lookahead both
    python -m Baselines.ablation_stress --mode stress
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

import Baselines._paths  # noqa: F401
from Baselines.ablation_models import (
    ABLATION_REFERENCE,
    DEFAULT_ABLATION_SCENARIOS,
    DEFAULT_STRESS_SCENARIOS,
    DEFAULT_TRAIN_SEEDS,
    KEY_PAIRED_COMPARISONS,
    PAPER_ABLATION_MODELS,
    STRESS_ABLATION_MODELS,
)
from Baselines.benchmark import _base_checkpoint, preflight_models
from Baselines.metrics import aggregate, metrics_frame
from Baselines.plots import plot_metric_bars, plot_trajectory_grid_frenet
from Baselines.registry import LABELS, build_controller, controller_kwargs, resolve_train_seeds
from Baselines.runner import RolloutResult, rollout
from Baselines.scenario import Scenario, build_scenario
from Baselines.stats import comparison_frame, summary_frame
from RL.corridor import DEFAULT_LANE_KF, DEFAULT_RUN_ID

DEFAULT_OUTPUT = Path("Baselines/results/v2")

STRESS_SPAWN = {
    "spawn_s_range": (20.0, 80.0),
    "spawn_lateral_frac": 0.55,
    "min_initial_spacing": 5.0,
}

_COLLPEN_CHECKPOINTS = {
    "residual_collpen": Path("RL/checkpoints/v3/residual_collpen_policy.pt"),
    "residual_collpen_dense": Path("RL/checkpoints/v3/residual_collpen_dense_policy.pt"),
}

STATS_METRICS = (
    "collision_events",
    "offroad_rate",
    "min_ttc_s",
    "arrival_rate",
    "rms_jerk",
)


def _available_models(models: list[str]) -> list[str]:
    """Skip collpen variants whose checkpoints have not been trained yet."""
    out = []
    for name in models:
        ckpt = _COLLPEN_CHECKPOINTS.get(name)
        if ckpt is not None and not ckpt.exists() and not list(ckpt.parent.glob(ckpt.stem + "_seed*.pt")):
            print(f"[skip] {name}: {ckpt} not found (train with --collision-penalty first)")
            continue
        out.append(name)
    return out


def _build_scenarios(
    args: argparse.Namespace,
    dense: bool,
    conflict_lookahead: str,
) -> list[Scenario]:
    seeds = list(range(args.seed, args.seed + args.scenarios))
    num_agents = args.stress_agents if dense else args.num_agents
    kwargs: dict = {
        "num_agents": num_agents,
        "max_steps": args.max_steps,
        "dt": args.dt,
        "run_id": args.run_id,
        "lane_kf": args.lane_kf,
        "conflict_lookahead": conflict_lookahead,
    }
    if getattr(args, "no_obb_safety_filter", False):
        kwargs["obb_safety_filter"] = False
    if dense:
        kwargs.update(STRESS_SPAWN)
    return [build_scenario(seed=s, **kwargs) for s in seeds]


def _run_suite(
    label: str,
    scenarios: list[Scenario],
    models: list[str],
    args: argparse.Namespace,
    output_dir: Path,
) -> pd.DataFrame:
    output_dir.mkdir(parents=True, exist_ok=True)
    la_note = ""
    if scenarios:
        horizon = scenarios[0].sim_config.get("conflict_horizon")
        substeps = scenarios[0].sim_config.get("conflict_substeps")
        la_note = f" | lookahead={horizon}s/{substeps}sub"
    print(
        f"\n=== {label} ===\n"
        f"Corridor run_id={args.run_id}, lane_kf={args.lane_kf}{la_note} | "
        f"{len(scenarios)} scenarios x {scenarios[0].num_agents} agents x "
        f"{args.max_steps} steps"
    )

    preflight_models(models, args, scenarios[0])
    results: dict[str, list[RolloutResult]] = {}
    train_seeds: dict[str, list[int]] = {}
    for model in models:
        seed_pairs = resolve_train_seeds(
            model,
            getattr(args, "train_seeds", None),
            base=_base_checkpoint(model, args),
        )
        model_results: list[RolloutResult] = []
        model_train_seeds: list[int] = []
        for train_seed, checkpoint in seed_pairs:
            kwargs = controller_kwargs(
                model,
                residual_checkpoint=args.residual_checkpoint,
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
        n_seeds = len({s for s in model_train_seeds if s >= 0})
        seed_note = f" | {n_seeds} train seeds" if n_seeds > 1 else ""
        print(
            f"  {LABELS.get(model, model):<32} collisions={collisions:6.2f} | "
            f"arrival={arrivals:5.2f}{seed_note}"
        )

    flat = [r for model_results in results.values() for r in model_results]
    frame = metrics_frame(flat)
    frame["train_seed"] = [s for model in results for s in train_seeds[model]]
    frame.to_csv(output_dir / f"{label}_raw.csv", index=False)

    stats_metrics = [c for c in STATS_METRICS if c in frame.columns]
    summary_frame(
        frame, stats_metrics, models=models, n_boot=args.n_boot
    ).to_csv(output_dir / f"{label}_stats.csv", index=False)

    if ABLATION_REFERENCE in models:
        comparison_frame(
            frame,
            stats_metrics,
            reference=ABLATION_REFERENCE,
            models=models,
            n_boot=args.n_boot,
        ).to_csv(output_dir / f"{label}_comparisons.csv", index=False)

    _write_paired_comparisons(frame, stats_metrics, models, args, output_dir / f"{label}_paired.csv")

    summary = aggregate(frame)
    summary.to_csv(output_dir / f"{label}_summary.csv", index=False)

    show = summary.set_index("model")
    mean_cols = [
        c
        for c in (
            "mean_collision_events",
            "mean_offroad_rate",
            "mean_arrival_rate",
            "mean_mean_travel_time_s",
            "mean_mean_speed_mps",
            "mean_rms_jerk",
        )
        if c in show.columns
    ]
    print(show[mean_cols].round(3).to_string())

    if not args.no_figures:
        try:
            plot_metric_bars(frame, output_dir / f"{label}_metrics.png")
            first_results = [results[m][0] for m in models if results[m]]
            plot_trajectory_grid_frenet(
                first_results,
                scenarios[0],
                output_dir / f"{label}_trajectories_frenet.png",
            )
        except Exception as exc:
            print(f"[figures] skipped: {exc}")

    return summary


def _write_paired_comparisons(
    frame: pd.DataFrame,
    metrics: list[str],
    models: list[str],
    args: argparse.Namespace,
    path: Path,
) -> None:
    rows = []
    for model_a, model_b in KEY_PAIRED_COMPARISONS:
        if model_a not in models or model_b not in models:
            continue
        for metric in metrics:
            cmp_row = comparison_frame(
                frame,
                [metric],
                reference=model_b,
                models=[model_a],
                n_boot=args.n_boot,
            )
            if not cmp_row.empty:
                rows.append(cmp_row.iloc[0].to_dict())
    if rows:
        pd.DataFrame(rows).to_csv(path, index=False)


def _resolve_model_list(args: argparse.Namespace, dense: bool) -> list[str]:
    if list(args.models) != PAPER_ABLATION_MODELS:
        return _available_models(list(args.models))
    return _available_models(STRESS_ABLATION_MODELS if dense else PAPER_ABLATION_MODELS)


def _lookahead_suites(args: argparse.Namespace) -> list[tuple[str, str]]:
    if args.lookahead == "both":
        return [("ablation", "full"), ("ablation_no_lookahead", "none")]
    if args.lookahead == "none":
        return [("ablation_no_lookahead", "none")]
    return [("ablation", "full")]


def main() -> None:
    parser = argparse.ArgumentParser(description="Paper ablations and dense stress tests")
    parser.add_argument("--mode", choices=("ablation", "stress", "both"), default="both")
    parser.add_argument("--models", nargs="+", default=PAPER_ABLATION_MODELS)
    parser.add_argument("--scenarios", type=int, default=DEFAULT_ABLATION_SCENARIOS)
    parser.add_argument(
        "--stress-scenarios",
        type=int,
        default=DEFAULT_STRESS_SCENARIOS,
        help="Scenario count for dense stress (defaults to same as sparse ablation)",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--train-seeds",
        type=int,
        nargs="+",
        default=DEFAULT_TRAIN_SEEDS,
        help="Training seeds for learned models; expects <stem>_seed<k>.pt checkpoints",
    )
    parser.add_argument(
        "--lookahead",
        choices=("full", "none", "both"),
        default="both",
        help="OBB conflict horizon: full=1.5s/4sub, none=1step; both runs two sparse suites",
    )
    parser.add_argument("--n-boot", type=int, default=10000, help="Bootstrap resamples for CIs")
    parser.add_argument("--num-agents", type=int, default=10, help="agents for standard ablation")
    parser.add_argument("--stress-agents", type=int, default=16, help="agents for dense stress")
    parser.add_argument("--max-steps", type=int, default=240)
    parser.add_argument("--dt", type=float, default=0.5)
    parser.add_argument("--run-id", type=int, default=DEFAULT_RUN_ID)
    parser.add_argument("--lane-kf", type=int, default=DEFAULT_LANE_KF)
    parser.add_argument("--calibration", type=Path, default=None)
    parser.add_argument("--residual-checkpoint", type=Path, default=None)
    parser.add_argument("--checkpoint-dir", type=Path, default=None)
    parser.add_argument("--pure-rl-checkpoint", type=Path, default=None)
    parser.add_argument(
        "--no-obb-safety-filter",
        action="store_true",
        help="Disable the shared closed-loop OBB filter on all controllers",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--no-figures", action="store_true")
    args = parser.parse_args()

    summaries: dict[str, pd.DataFrame] = {}

    if args.mode in {"ablation", "both"}:
        models = _resolve_model_list(args, dense=False)
        if not models:
            raise SystemExit("No ablation models available to evaluate.")
        for label, lookahead in _lookahead_suites(args):
            scenarios = _build_scenarios(args, dense=False, conflict_lookahead=lookahead)
            summaries[label] = _run_suite(
                label, scenarios, models, args, args.output_dir / label
            )

    if args.mode in {"stress", "both"}:
        models = _resolve_model_list(args, dense=True)
        if not models:
            print("[stress] no models available; skipping")
        else:
            saved = args.scenarios
            args.scenarios = args.stress_scenarios
            scenarios = _build_scenarios(args, dense=True, conflict_lookahead="full")
            args.scenarios = saved
            summaries["stress"] = _run_suite(
                "stress", scenarios, models, args, args.output_dir / "stress"
            )

    rows = []
    for regime, summary in summaries.items():
        indexed = summary.set_index("model") if "model" in summary.columns else summary
        if "utility_pt" not in indexed.index:
            continue
        residual_name = ABLATION_REFERENCE
        if residual_name not in indexed.index:
            continue
        for metric in STATS_METRICS:
            col = f"mean_{metric}"
            if col not in indexed.columns:
                continue
            prior = float(indexed.loc["utility_pt", col])
            residual = float(indexed.loc[residual_name, col])
            rows.append(
                {
                    "regime": regime,
                    "residual_model": residual_name,
                    "metric": metric,
                    "utility_pt": prior,
                    "residual": residual,
                    "delta_residual_minus_prior": residual - prior,
                }
            )
    if rows:
        delta = pd.DataFrame(rows)
        path = args.output_dir / "ablation_stress_delta.csv"
        delta.to_csv(path, index=False)
        print(f"\nWrote {path}")
        print(delta.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
