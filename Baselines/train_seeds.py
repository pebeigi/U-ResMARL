"""Train one model under several independent training seeds.

The benchmark reports intervals over training seeds, which requires one
checkpoint per seed named exactly as ``Baselines.registry.seed_checkpoint``
expects. Example:

    python -m Baselines.train_seeds --model residual_marl --seeds 0 1 2 3 4 \
        --jobs 3 -- --updates 400 --behavior-coef 0.1

Everything after ``--`` is forwarded verbatim to the underlying trainer, so the
seeded runs use the same hyperparameters as the single-seed run.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import Baselines._paths  # noqa: F401
from Baselines._paths import REPO_ROOT
from Baselines.registry import LEARNED_CHECKPOINTS, seed_checkpoint

TRAINERS: dict[str, list[str]] = {
    "residual_marl": ["-m", "RL.train_ppo"],
    "residual_param": ["-m", "RL.train_ppo", "--residual-mode", "param_delta"],
    "residual_nominal": ["-m", "RL.train_ppo", "--prefer-params", "nominal"],
    "residual_collpen": ["-m", "RL.train_ppo"],
    "residual_collpen_dense": ["-m", "RL.train_ppo"],
    "direct_discrete_rl": ["-m", "Baselines.train_direct_discrete_rl"],
    "pure_rl": ["-m", "Baselines.train_pure_rl"],
    "pure_rl_safe": ["-m", "Baselines.train_pure_rl"],
    "mappo": ["-m", "Baselines.train_marl", "--algo", "mappo"],
    "happo": ["-m", "Baselines.train_marl", "--algo", "happo"],
    "hatrpo": ["-m", "Baselines.train_marl", "--algo", "hatrpo"],
}


def run_one(
    model: str,
    train_seed: int,
    passthrough: list[str],
    log_dir: Path,
    single_thread: bool,
) -> tuple[int, int, Path, float]:
    checkpoint = seed_checkpoint(model, train_seed)
    assert checkpoint is not None
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{model}_seed{train_seed}.log"

    cmd = [sys.executable, "-u", *TRAINERS[model], "--seed", str(train_seed), "--save", str(checkpoint)]
    cmd.extend(passthrough)

    env = dict(os.environ)
    env["PYTHONIOENCODING"] = "utf-8"
    if single_thread:
        # Parallel workers would otherwise each spawn a full thread pool and thrash.
        env["OMP_NUM_THREADS"] = "1"
        env["MKL_NUM_THREADS"] = "1"

    start = time.time()
    with log_path.open("w", encoding="utf-8") as log_file:
        proc = subprocess.run(cmd, cwd=REPO_ROOT, env=env, stdout=log_file, stderr=subprocess.STDOUT)
    return train_seed, proc.returncode, log_path, time.time() - start


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a model under several seeds")
    parser.add_argument("--model", choices=sorted(TRAINERS), default="residual_marl")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--jobs", type=int, default=1, help="Concurrent training runs")
    parser.add_argument("--log-dir", type=Path, default=Path("RL/logs/revision5/seeds"))
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Retrain seeds whose checkpoint already exists",
    )
    parser.add_argument(
        "trainer_args",
        nargs=argparse.REMAINDER,
        help="Arguments after -- are forwarded to the trainer",
    )
    args = parser.parse_args()

    passthrough = [a for a in args.trainer_args if a != "--"]
    pending = []
    for s in args.seeds:
        ckpt = seed_checkpoint(args.model, s)
        if ckpt is not None and ckpt.exists() and not args.overwrite:
            print(f"[skip] seed {s}: {ckpt} exists (use --overwrite to retrain)")
            continue
        pending.append(s)

    if not pending:
        print("Nothing to train.")
        return

    base = LEARNED_CHECKPOINTS.get(args.model)
    print(
        f"Training {args.model} for seeds {pending} "
        f"({args.jobs} concurrent), checkpoints next to {base}"
    )
    if passthrough:
        print(f"Forwarded trainer args: {' '.join(passthrough)}")

    failures = []
    single_thread = args.jobs > 1
    with ThreadPoolExecutor(max_workers=max(1, args.jobs)) as pool:
        futures = [
            pool.submit(run_one, args.model, s, passthrough, args.log_dir, single_thread)
            for s in pending
        ]
        for future in futures:
            train_seed, code, log_path, elapsed = future.result()
            status = "ok" if code == 0 else f"FAILED (exit {code})"
            print(f"  seed {train_seed}: {status} in {elapsed / 60:.1f} min -> {log_path}")
            if code != 0:
                failures.append(train_seed)

    trained = [s for s in args.seeds if (seed_checkpoint(args.model, s) or Path()).exists()]
    print(f"\nCheckpoints available for seeds: {trained}")
    print(
        "Evaluate with: python -m Baselines.benchmark --train-seeds "
        + " ".join(str(s) for s in trained)
    )
    if failures:
        raise SystemExit(f"Training failed for seeds {failures}")


if __name__ == "__main__":
    main()
