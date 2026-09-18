"""Paper revision protocol: multi-seed training + fair benchmark reruns.

Addresses reviewer concerns (comments 1–4):
  * Planner fairness — shared closed-loop OBB safety filter on all controllers.
  * RL variance — 3 independent training seeds with bootstrap CIs.
  * Matched direct discrete RL vs. residual MARL.
  * Expanded ablation package (weights-only, σ-only, nominal prior, no-lookahead).

Typical workflow (from repo root):

    python -m Baselines.paper_rerun status
    python -m Baselines.paper_rerun train --jobs 2
    python -m Baselines.paper_rerun eval
    python -m Baselines.paper_figures --all
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import Baselines._paths  # noqa: F401
from Baselines._paths import REPO_ROOT
from Baselines.ablation_models import (
    DEFAULT_ABLATION_SCENARIOS,
    DEFAULT_STRESS_SCENARIOS,
    DEFAULT_TRAIN_SEEDS,
    PAPER_ABLATION_MODELS,
)
from Baselines.registry import LEARNED_CHECKPOINTS, seed_checkpoint

PAPER_OUTPUT = Path("Baselines/results/revision5/paper_rerun")
ABLATION_OUTPUT = Path("Baselines/results/revision5/paper_ablation")
STRESS_OUTPUT = Path("Baselines/results/revision5/paper_stress")

TRAIN_SPECS: dict[str, list[str]] = {
    "residual_param": [
        "--updates", "100", "--max-steps", "240",
        "--collision-penalty", "0", "--collision-event-penalty", "1", "--gamma", "0.95",
        "--residual-mode", "param_delta",
    ],
    "residual_nominal": [
        "--updates", "100", "--max-steps", "240",
        "--collision-penalty", "0", "--collision-event-penalty", "1", "--gamma", "0.95",
        "--prefer-params", "nominal",
    ],
    "residual_marl": [
        "--updates", "100", "--max-steps", "240",
        "--collision-penalty", "0", "--collision-event-penalty", "1", "--gamma", "0.95",
        "--residual-mode", "candidate_logits",
    ],
    "mappo": [
        "--updates", "100", "--max-steps", "240",
        "--collision-penalty", "0", "--collision-event-penalty", "1", "--gamma", "0.95",
    ],
    "direct_discrete_rl": [
        "--updates", "100", "--max-steps", "240",
        "--collision-penalty", "0", "--collision-event-penalty", "1", "--gamma", "0.95",
        "--minibatch-size", "512",
    ],
    "residual_collpen_dense": [
        "--updates", "100", "--num-agents", "16", "--dense-spawn", "--max-steps", "240",
        "--collision-penalty", "0", "--collision-event-penalty", "1", "--gamma", "0.95",
        "--residual-mode", "candidate_logits",
    ],
}


def _missing_seed_checkpoints(model: str, seeds: list[int]) -> list[int]:
    base = LEARNED_CHECKPOINTS.get(model)
    if base is None:
        return list(seeds)
    missing = []
    for s in seeds:
        path = seed_checkpoint(model, s, base)
        if path is None or not path.exists():
            missing.append(s)
    return missing


def cmd_train(args: argparse.Namespace) -> None:
    for model in args.models:
        missing = _missing_seed_checkpoints(model, args.seeds)
        if not missing and not args.overwrite:
            print(f"[train] {model}: all {len(args.seeds)} seed checkpoints exist — skip")
            continue
        passthrough = TRAIN_SPECS.get(model, [])
        cmd = [
            sys.executable,
            "-m",
            "Baselines.train_seeds",
            "--model",
            model,
            "--seeds",
            *[str(s) for s in args.seeds],
            "--jobs",
            str(args.jobs),
            "--log-dir",
            str(args.log_dir),
        ]
        if args.overwrite:
            cmd.append("--overwrite")
        cmd.extend(["--", *passthrough])
        print(f"[train] {model}: seeds {args.seeds}")
        subprocess.run(cmd, cwd=REPO_ROOT, check=True)


def cmd_eval(args: argparse.Namespace) -> None:
    train_flag = ["--train-seeds", *[str(s) for s in args.seeds]]
    n_boot = ["--n-boot", str(args.n_boot)]

    ablation_cmd = [
        sys.executable,
        "-m",
        "Baselines.ablation_stress",
        "--mode",
        "both",
        "--lookahead",
        "both",
        "--models",
        *PAPER_ABLATION_MODELS,
        "--scenarios",
        str(args.scenarios),
        "--stress-scenarios",
        str(args.scenarios),
        "--output-dir",
        str(args.ablation_dir),
        *train_flag,
        *n_boot,
    ]
    if args.no_figures:
        ablation_cmd.append("--no-figures")
    print("[eval] full ablation package (sparse + no-lookahead + stress)")
    subprocess.run(ablation_cmd, cwd=REPO_ROOT, check=True)


def cmd_status(args: argparse.Namespace) -> None:
    print("Checkpoint status (per-seed files expected for paper revision):")
    for model in args.models:
        missing = _missing_seed_checkpoints(model, args.seeds)
        base = LEARNED_CHECKPOINTS.get(model)
        print(f"  {model}: base={base}")
        if missing:
            print(f"    missing seeds: {missing}")
        else:
            print(f"    OK — {len(args.seeds)} seed checkpoints present")


def main() -> None:
    parser = argparse.ArgumentParser(description="Paper revision training and evaluation protocol")
    sub = parser.add_subparsers(dest="command", required=True)

    p_train = sub.add_parser("train", help="Multi-seed RL training for paper models")
    p_train.add_argument(
        "--models",
        nargs="+",
        default=["residual_marl", "residual_param", "residual_nominal", "direct_discrete_rl", "mappo"],
        choices=sorted(TRAIN_SPECS),
    )
    p_train.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_TRAIN_SEEDS)
    p_train.add_argument("--jobs", type=int, default=1)
    p_train.add_argument("--log-dir", type=Path, default=Path("RL/logs/revision5/paper_rerun"))
    p_train.add_argument("--overwrite", action="store_true")
    p_train.set_defaults(func=cmd_train)

    p_eval = sub.add_parser("eval", help="Full ablation + stress with train-seed CIs")
    p_eval.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_TRAIN_SEEDS)
    p_eval.add_argument("--scenarios", type=int, default=DEFAULT_ABLATION_SCENARIOS)
    p_eval.add_argument("--ablation-dir", type=Path, default=ABLATION_OUTPUT)
    p_eval.add_argument("--n-boot", type=int, default=10000)
    p_eval.add_argument("--no-figures", action="store_true")
    p_eval.set_defaults(func=cmd_eval)

    p_status = sub.add_parser("status", help="List missing per-seed checkpoints")
    p_status.add_argument(
        "--models",
        nargs="+",
        default=["residual_marl", "residual_param", "residual_nominal", "direct_discrete_rl", "mappo"],
    )
    p_status.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_TRAIN_SEEDS)
    p_status.set_defaults(func=cmd_status)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
