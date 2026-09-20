"""Train residual / ΔΘ / matched RL, then sparse + ablation + stress benchmarks.

24 PPO updates, seeds 0/1/2, 20 scenes, 10 agents, 240 steps, shared
decision/OBB protocol (16 val episodes every 10 updates).

    python -u -m Baselines._run_2day_pipeline
    python -u -m Baselines._run_2day_pipeline --skip-train
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import Baselines._paths  # noqa: F401
from Baselines._paths import REPO_ROOT

SEEDS = [0, 1, 2]
LOG_DIR = Path("RL/logs/revision6/paper_2day")
PIPELINE_LOG = LOG_DIR / "pipeline.log"
BENCH_OUT = Path("Baselines/results/revision6/paper_2day")

# Matched paper recipe under the shared decision/OBB/selection protocol.
TRAIN_FWD = [
    "--updates", "24",
    "--max-steps", "240",
    "--episodes-per-update", "4",
    "--num-agents", "10",
    "--collision-penalty", "0",
    "--collision-event-penalty", "1",
    "--gamma", "0.95",
    "--val-every", "10",
    "--val-episodes", "16",
    "--test-episodes", "16",
    "--log-every", "1",
    "--skip-test",
]

TRAIN_MODELS = ["residual_marl", "residual_param", "direct_discrete_rl", "mappo"]

BENCH_MODELS = [
    "orca",
    "social_force",
    "dwa",
    "mppi",
    "frenet",
    "utility_pt",
    "direct_discrete_rl",
    "mappo",
    "residual_marl",
    "ctrl_sim",
    "ctg_plus_plus",
]

PARAM_EVAL_MODELS = [
    "utility_pt",
    "residual_marl",
    "residual_param",
    "residual_weights_only",
    "residual_sigma_only",
    "direct_discrete_rl",
    "mappo",
]
PARAM_OUT = Path("Baselines/results/revision6/param_lookahead_stress")
GATE_OUT = Path("Baselines/results/revision6/gate_ablation")


def log(msg: str) -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    line = f"{datetime.now(timezone.utc).isoformat()}  {msg}"
    print(line, flush=True)
    with PIPELINE_LOG.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")


def run(cmd: list[str], step: str) -> None:
    log(f"START {step}: {' '.join(cmd)}")
    t0 = time.time()
    proc = subprocess.run(cmd, cwd=REPO_ROOT)
    hours = (time.time() - t0) / 3600.0
    if proc.returncode != 0:
        log(f"FAIL  {step} (exit {proc.returncode}) after {hours:.2f} h")
        raise SystemExit(proc.returncode)
    log(f"OK    {step} in {hours:.2f} h")


def main() -> None:
    parser = argparse.ArgumentParser(description="2-day residual + baseline sprint")
    parser.add_argument("--skip-train", action="store_true")
    parser.add_argument("--skip-benchmark", action="store_true")
    parser.add_argument("--jobs", type=int, default=2)
    parser.add_argument("--scenarios", type=int, default=20)
    parser.add_argument("--overwrite", action="store_true", default=True)
    parser.add_argument(
        "--models",
        nargs="+",
        default=None,
        help="Train only these models (default: residual_marl direct_discrete_rl mappo)",
    )
    args = parser.parse_args()
    py = sys.executable
    seed_args = [str(s) for s in SEEDS]
    train_models = args.models if args.models else TRAIN_MODELS

    if not args.skip_train:
        for model in train_models:
            cmd = [
                py, "-u", "-m", "Baselines.train_seeds",
                "--model", model,
                "--seeds", *seed_args,
                "--jobs", str(args.jobs),
                "--log-dir", str(LOG_DIR),
            ]
            if args.overwrite:
                cmd.append("--overwrite")
            cmd.extend(["--", *TRAIN_FWD])
            if model == "residual_marl":
                cmd.extend(["--residual-mode", "candidate_logits"])
            elif model == "residual_param":
                cmd.extend(["--residual-mode", "param_delta"])
            elif model == "direct_discrete_rl":
                cmd.extend(["--minibatch-size", "512"])
            run(cmd, f"train {model}")

    if not args.skip_benchmark:
        run(
            [
                py, "-u", "-m", "Baselines.benchmark",
                "--scenarios", str(args.scenarios),
                "--num-agents", "10",
                "--max-steps", "240",
                "--models", *BENCH_MODELS,
                "--train-seeds", *seed_args,
                "--output-dir", str(BENCH_OUT),
                "--n-boot", "5000",
                "--conflict-lookahead", "full",
            ],
            "sparse benchmark",
        )
        run(
            [
                py, "-u", "-m", "Baselines.ablation_stress",
                "--mode", "both",
                "--lookahead", "both",
                "--models", *PARAM_EVAL_MODELS,
                "--scenarios", str(args.scenarios),
                "--stress-scenarios", str(args.scenarios),
                "--num-agents", "10",
                "--stress-agents", "16",
                "--max-steps", "240",
                "--train-seeds", *seed_args,
                "--n-boot", "5000",
                "--output-dir", str(PARAM_OUT),
            ],
            "param / no-lookahead / dense stress",
        )
        run(
            [
                py, "-u", "-m", "Baselines.ablation_stress",
                "--mode", "gate",
                "--lookahead", "full",
                "--scenarios", str(args.scenarios),
                "--num-agents", "10",
                "--max-steps", "240",
                "--train-seeds", *seed_args,
                "--n-boot", "5000",
                "--output-dir", str(GATE_OUT),
            ],
            "gate ablation",
        )
    log("2-day pipeline complete")


if __name__ == "__main__":
    main()
