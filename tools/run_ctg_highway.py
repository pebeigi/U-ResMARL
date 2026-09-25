"""Refresh and evaluate only the highway CTG++ checkpoints before a shared deadline."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import threading
import time

from run_ctg_sites import terminate_tree, utc_now, write_json


ROOT = Path(__file__).resolve().parents[1]
CONTROL = ROOT / "runs" / "ctg_plus_plus_sites_20260923"
STATUS_PATH = CONTROL / "highway_status.json"
LOGS = CONTROL / "logs"
CHECKPOINTS = ROOT / "New Baselines" / "checkpoints"
DATA = ROOT / "New Baselines" / "artifacts" / "data"
CALIBRATION = ROOT / "Calibration" / "utility_calibration.json"


def main() -> None:
    shared = json.loads((CONTROL / "status.json").read_text(encoding="utf-8"))
    deadline_wall = datetime.fromisoformat(shared["deadline_utc"]).timestamp()
    state = {
        "status": "waiting_for_site_gpus",
        "started_utc": utc_now(),
        "deadline_utc": shared["deadline_utc"],
        "scope": "Highway CTG++ only; reuse valid training candidates where available",
        "jobs": {},
    }
    lock = threading.Lock()
    LOGS.mkdir(parents=True, exist_ok=True)
    write_json(STATUS_PATH, state)

    def remaining() -> float:
        return deadline_wall - time.time()

    # Preserve the original global 5 h 45 min cap and wait for the two site
    # workers to release the GPUs instead of oversubscribing them.
    while remaining() > 0:
        site_state = json.loads((CONTROL / "status.json").read_text(encoding="utf-8"))
        if site_state.get("status") != "running":
            break
        time.sleep(min(15.0, max(1.0, remaining())))
    if remaining() <= 0:
        state.update(status="timed_out", finished_utc=utc_now())
        write_json(STATUS_PATH, state)
        return

    state["status"] = "running"
    write_json(STATUS_PATH, state)

    def run(key: str, arguments: list[str], gpu: int) -> None:
        timeout = remaining()
        if timeout <= 0:
            raise TimeoutError("shared CTG++ deadline reached")
        command = [sys.executable, "-u", str(ROOT / "New Baselines" / "run.py"), *map(str, arguments)]
        log = LOGS / f"highway_{key}.log"
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(gpu)
        env["OMP_NUM_THREADS"] = env["MKL_NUM_THREADS"] = "1"
        started = time.monotonic()
        with log.open("w", encoding="utf-8") as handle:
            process = subprocess.Popen(command, cwd=str(ROOT), env=env, stdout=handle,
                                       stderr=subprocess.STDOUT)
            with lock:
                state["jobs"][key] = dict(status="running", pid=process.pid, gpu=gpu,
                                           command=command, log=str(log), started_utc=utc_now())
                write_json(STATUS_PATH, state)
            try:
                code = process.wait(timeout=max(1.0, timeout))
            except subprocess.TimeoutExpired:
                terminate_tree(process)
                with lock:
                    state["jobs"][key].update(status="timed_out", finished_utc=utc_now(),
                                               wall_time_s=time.monotonic() - started)
                    write_json(STATUS_PATH, state)
                raise TimeoutError("shared CTG++ deadline reached")
        with lock:
            state["jobs"][key].update(status="complete" if code == 0 else "failed", exit_code=code,
                                       finished_utc=utc_now(), wall_time_s=time.monotonic() - started)
            write_json(STATUS_PATH, state)
        if code:
            raise RuntimeError(f"{key} failed; see {log}")

    try:
        # Seed 0 predates retained matched-protocol candidates, so only this
        # seed needs fresh offline training. Seeds 1/2 already have 10k-step
        # candidates; calibration affects closed-loop selection, not training.
        raw0 = CHECKPOINTS / "ctg_plus_plus_raw_seed0.pt"
        run("seed0_train", [
            "--threads", "1", "train", "ctg_plus_plus", "--data", DATA,
            "--steps", "10000", "--batch-size", "4", "--eval-every", "2000",
            "--seed", "0", "--device", "cuda", "--output", raw0,
        ], 0)

        candidate_dirs = {0: raw0.with_suffix(".candidates")}
        for seed in (1, 2):
            source = CHECKPOINTS / f"ctg_plus_plus_policy_seed{seed}.candidates"
            shortlist = CHECKPOINTS / f"ctg_plus_plus_current_shortlist_seed{seed}.candidates"
            shortlist.mkdir(parents=True, exist_ok=True)
            for old in shortlist.glob("step_*.pt"):
                old.unlink()
            for step in (2000, 4000, 6000, 8000, 10000):
                src = source / f"step_{step:08d}.pt"
                if not src.exists():
                    raise FileNotFoundError(src)
                shutil.copy2(src, shortlist / src.name)
            candidate_dirs[seed] = shortlist

        def select(seed: int, gpu: int) -> None:
            run(f"seed{seed}_select", [
                "--threads", "1", "select", "ctg_plus_plus", "--candidates", candidate_dirs[seed],
                "--device", "cuda", "--output", CHECKPOINTS / f"ctg_plus_plus_policy_seed{seed}.pt",
                "--num-agents", "10", "--max-steps", "240", "--val-episodes", "16",
                "--validation-seed-start", "910000", "--test-episodes", "16",
                "--test-seed-start", "810000", "--run-id", "2", "--lane-kf", "1",
                "--calibration", CALIBRATION,
            ], gpu)

        # Two selections at a time, one per GPU.
        for batch in ((0, 1), (2,)):
            with ThreadPoolExecutor(max_workers=len(batch)) as pool:
                futures = [pool.submit(select, seed, gpu) for gpu, seed in enumerate(batch)]
                for future in as_completed(futures):
                    future.result()

        benchmark_output = ROOT / "Baselines" / "results" / "paper_2day" / "ctg_plus_plus_only_current_calibration"
        # Benchmark is a module rather than a New Baselines CLI command.
        timeout = remaining()
        if timeout <= 0:
            raise TimeoutError("shared CTG++ deadline reached")
        key = "benchmark"
        command = [sys.executable, "-u", "-m", "Baselines.benchmark",
                   "--models", "ctg_plus_plus", "--seed", "810000", "--scenarios", "20",
                   "--num-agents", "10", "--max-steps", "240", "--calibration", str(CALIBRATION),
                   "--checkpoint-dir", str(CHECKPOINTS), "--train-seeds", "0", "1", "2",
                   "--require-matched-protocol", "--n-boot", "5000", "--output-dir", str(benchmark_output),
                   "--conflict-lookahead", "full", "--no-figures"]
        log = LOGS / "highway_benchmark.log"
        env = os.environ.copy(); env["CUDA_VISIBLE_DEVICES"] = "0"; env["OMP_NUM_THREADS"] = "1"
        started = time.monotonic()
        with log.open("w", encoding="utf-8") as handle:
            process = subprocess.Popen(command, cwd=str(ROOT), env=env, stdout=handle, stderr=subprocess.STDOUT)
            state["jobs"][key] = dict(status="running", pid=process.pid, gpu=0, command=command,
                                       log=str(log), started_utc=utc_now())
            write_json(STATUS_PATH, state)
            try:
                code = process.wait(timeout=max(1.0, timeout))
            except subprocess.TimeoutExpired:
                terminate_tree(process)
                state["jobs"][key].update(status="timed_out", finished_utc=utc_now(),
                                           wall_time_s=time.monotonic() - started)
                write_json(STATUS_PATH, state)
                raise TimeoutError("shared CTG++ deadline reached")
        state["jobs"][key].update(status="complete" if code == 0 else "failed", exit_code=code,
                                   finished_utc=utc_now(), wall_time_s=time.monotonic() - started)
        write_json(STATUS_PATH, state)
        if code:
            raise RuntimeError(f"benchmark failed; see {log}")
        state["status"] = "complete"
    except TimeoutError as exc:
        state.update(status="timed_out", error=str(exc))
    except BaseException as exc:
        state.update(status="failed", error=repr(exc))
        raise
    finally:
        state["finished_utc"] = utc_now()
        write_json(STATUS_PATH, state)


if __name__ == "__main__":
    main()
