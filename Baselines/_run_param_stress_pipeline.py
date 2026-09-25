"""Train Delta-Theta residual, then param/lookahead ablations and dense stress.

Matched to the 2-day sparse protocol (20 scenes, 10 agents, 240 steps, seeds 0/1/2).
Weights-only and sigma-only reuse the param residual checkpoints at inference.

    python -u -m Baselines._run_param_stress_pipeline
    python -u -m Baselines._run_param_stress_pipeline --skip-train
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
LOG_DIR = Path("RL/logs/param_stress")
PIPELINE_LOG = LOG_DIR / "pipeline.log"
BENCH_OUT = Path("Baselines/results/param_lookahead_stress")

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

EVAL_MODELS = [
    "utility_pt",
    "residual_marl",
    "residual_param",
    "residual_weights_only",
    "residual_sigma_only",
    "direct_discrete_rl",
    "mappo",
]


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
    parser = argparse.ArgumentParser(description="Delta-Theta + lookahead + dense stress")
    parser.add_argument("--skip-train", action="store_true")
    parser.add_argument("--skip-eval", action="store_true")
    parser.add_argument("--jobs", type=int, default=2)
    parser.add_argument("--scenarios", type=int, default=20)
    parser.add_argument("--overwrite", action="store_true", default=True)
    args = parser.parse_args()
    py = sys.executable
    seed_args = [str(s) for s in SEEDS]

    if not args.skip_train:
        cmd = [
            py, "-u", "-m", "Baselines.train_seeds",
            "--model", "residual_param",
            "--seeds", *seed_args,
            "--jobs", str(args.jobs),
            "--log-dir", str(LOG_DIR),
        ]
        if args.overwrite:
            cmd.append("--overwrite")
        cmd.extend(["--", *TRAIN_FWD])
        run(cmd, "train residual_param")

    if not args.skip_eval:
        run(
            [
                py, "-u", "-m", "Baselines.ablation_stress",
                "--mode", "both",
                "--lookahead", "both",
                "--models", *EVAL_MODELS,
                "--scenarios", str(args.scenarios),
                "--stress-scenarios", str(args.scenarios),
                "--num-agents", "10",
                "--stress-agents", "16",
                "--max-steps", "240",
                "--train-seeds", *seed_args,
                "--n-boot", "5000",
                "--output-dir", str(BENCH_OUT),
            ],
            "param / no-lookahead / dense stress",
        )
    log("param-stress pipeline complete")


if __name__ == "__main__":
    main()
