"""Train CtRL-Sim and CTG++ and evaluate them on the paper sparse protocol.

Uses the already-prepared New Baselines train/validation caches, the documented
10,000-step starting budget, and the same closed-loop harness as the other
paper baselines (20 scenes, 10 agents, 240 steps). Seed 0 is the starting
checkpoint; pass --seeds 0 1 2 later if you want matched multi-seed CIs.

    python -u -m Baselines._run_new_baselines
    python -u -m Baselines._run_new_baselines --skip-train
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import Baselines._paths  # noqa: F401
from Baselines._paths import REPO_ROOT

LOG_DIR = Path("RL/logs/revision6/new_baselines")
PIPELINE_LOG = LOG_DIR / "pipeline.log"
CKPT_DIR = Path("New Baselines/checkpoints")
BENCH_OUT = Path("Baselines/results/revision6/new_baselines")
TRAIN_SCRIPT = Path("New Baselines/run.py")
MODELS = ("ctrl_sim", "ctg_plus_plus")


def log(msg: str) -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    line = f"{datetime.now(timezone.utc).isoformat()}  {msg}"
    print(line, flush=True)
    with PIPELINE_LOG.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")


def run(cmd: list[str], step: str, env: dict[str, str] | None = None) -> None:
    log(f"START {step}: {' '.join(cmd)}")
    t0 = time.time()
    proc = subprocess.run(cmd, cwd=REPO_ROOT, env=env)
    hours = (time.time() - t0) / 3600.0
    if proc.returncode != 0:
        log(f"FAIL  {step} (exit {proc.returncode}) after {hours:.2f} h")
        raise SystemExit(proc.returncode)
    log(f"OK    {step} in {hours:.2f} h")


def train_one(name: str, gpu: int, args: argparse.Namespace, py: str) -> subprocess.Popen:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    log_path = LOG_DIR / f"{name}_train.log"
    handle = log_path.open("w", encoding="utf-8")
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    output = CKPT_DIR / f"{name}_policy.pt"
    cmd = [
        py, "-u", str(TRAIN_SCRIPT), "train", name,
        "--steps", str(args.steps),
        "--batch-size", str(args.batch_size),
        "--eval-every", str(args.eval_every),
        "--seed", str(args.seed),
        "--device", "cuda",
        "--output", str(output),
    ]
    log(f"START train {name} on GPU {gpu}: {' '.join(cmd)}")
    proc = subprocess.Popen(
        cmd, cwd=REPO_ROOT, stdout=handle, stderr=subprocess.STDOUT, env=env
    )
    proc._log_handle = handle  # type: ignore[attr-defined]
    proc._started = time.time()  # type: ignore[attr-defined]
    proc._output = output  # type: ignore[attr-defined]
    return proc


def main() -> None:
    parser = argparse.ArgumentParser(description="Train/eval New Baselines on the paper protocol")
    parser.add_argument("--skip-train", action="store_true")
    parser.add_argument("--skip-eval", action="store_true")
    parser.add_argument("--steps", type=int, default=10000)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--eval-every", type=int, default=250)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--scenarios", type=int, default=20)
    parser.add_argument("--num-agents", type=int, default=10)
    parser.add_argument("--max-steps", type=int, default=240)
    args = parser.parse_args()
    py = sys.executable

    if not args.skip_train:
        n_gpu = 2
        try:
            import torch
            n_gpu = max(1, int(torch.cuda.device_count()))
        except Exception:
            pass
        procs = {
            name: train_one(name, gpu % n_gpu, args, py)
            for gpu, name in enumerate(MODELS)
        }
        failed = []
        for name, proc in procs.items():
            code = proc.wait()
            proc._log_handle.close()  # type: ignore[attr-defined]
            hours = (time.time() - proc._started) / 3600.0  # type: ignore[attr-defined]
            if code != 0:
                failed.append(name)
                log(f"FAIL  train {name} (exit {code}) after {hours:.2f} h")
            else:
                output = Path(proc._output)  # type: ignore[attr-defined]
                seeded = output.with_name(f"{output.stem}_seed{args.seed}{output.suffix}")
                shutil.copy2(output, seeded)
                log(f"OK    train {name} in {hours:.2f} h -> {output}")
        if failed:
            raise SystemExit(1)

    if not args.skip_eval:
        run(
            [
                py, "-u", "-m", "Baselines.benchmark",
                "--models", "utility_pt", "ctrl_sim", "ctg_plus_plus",
                "--scenarios", str(args.scenarios),
                "--num-agents", str(args.num_agents),
                "--max-steps", str(args.max_steps),
                "--n-boot", "5000",
                "--reference-model", "utility_pt",
                "--output-dir", str(BENCH_OUT),
            ],
            "paper sparse closed-loop eval",
        )
    log("new-baselines pipeline complete")


if __name__ == "__main__":
    main()
