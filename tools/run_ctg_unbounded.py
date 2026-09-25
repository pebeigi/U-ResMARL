"""Run only CTG++ on highway, TGSIM, and roundabout without a time limit.

This launcher keeps the paper protocols (three training seeds, safety-first then
PDMS checkpoint selection on 16 validation scenes, and 20 matched test scenes)
while reducing CTG++ inference cost in two documented ways:

* 20 reverse-diffusion steps instead of 100;
* two retained 10k-step training candidates instead of five.

The job is resumable.  Completed compatible training, selection, and benchmark
outputs are reused; no other controller is trained or evaluated.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import json
import os
from pathlib import Path
from queue import Queue
import subprocess
import sys
import threading
import time

import torch


ROOT = Path(__file__).resolve().parents[1]
CONTROL = ROOT / "runs" / "ctg_plus_plus_unbounded_20260923"

SITES = {
    "highway": {
        "case": None,
        "run": ROOT / "New Baselines",
        "data": ROOT / "New Baselines" / "artifacts" / "data",
        "checkpoints": ROOT / "New Baselines" / "checkpoints",
        "calibration": ROOT / "Calibration" / "utility_calibration.json",
        "run_env": None,
        "num_agents": 10,
        "max_steps": 240,
        "run_id": 2,
        "lane_kf": 1,
        "result": ROOT / "Baselines" / "results" / "paper_2day"
                  / "ctg_plus_plus_only_fast20",
    },
    "tgsim": {
        "case": ROOT / "TGSIM Case",
        "run": ROOT / "TGSIM Case" / "runs" / "tgsim_recalibrated_fixed_20260922",
        "data": ROOT / "TGSIM Case" / "runs" / "tgsim_recalibrated_fixed_20260922" / "data",
        "checkpoints": ROOT / "TGSIM Case" / "runs" / "tgsim_recalibrated_fixed_20260922" / "checkpoints",
        "calibration": ROOT / "Calibration" / "utility_calibration_tgsim.json",
        "run_env": "TGSIM_RUN_DIR",
        "num_agents": 6,
        "max_steps": 80,
        "run_id": 0,
        "lane_kf": 0,
        "result": ROOT / "TGSIM Case" / "runs" / "tgsim_recalibrated_fixed_20260922"
                  / "results" / "ctg_plus_plus_only_fast20",
    },
    "roundabout": {
        "case": ROOT / "Roundabout Case",
        "run": ROOT / "Roundabout Case" / "runs" / "roundabout_3x_recalibrated_20260922",
        "data": ROOT / "Roundabout Case" / "runs" / "roundabout_3x_recalibrated_20260922" / "data",
        "checkpoints": ROOT / "Roundabout Case" / "runs" / "roundabout_3x_recalibrated_20260922" / "checkpoints",
        "calibration": ROOT / "Calibration" / "utility_calibration_jounieh.json",
        "run_env": "ROUNDABOUT_RUN_DIR",
        "num_agents": 6,
        "max_steps": 80,
        "run_id": 0,
        "lane_kf": 0,
        "result": ROOT / "Roundabout Case" / "runs" / "roundabout_3x_recalibrated_20260922"
                  / "results" / "ctg_plus_plus_only_fast20",
    },
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def checkpoint_matches(path: Path, diffusion_steps: int, *, selected: bool = False) -> bool:
    if not path.exists():
        return False
    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
        if payload.get("model") != "ctg_plus_plus":
            return False
        if int(payload.get("config", {}).get("diffusion_steps", -1)) != diffusion_steps:
            return False
        if selected:
            return payload.get("selection_status") in {"safety_feasible", "safety_failed"}
        return payload.get("selection_status") == "offline_candidate"
    except Exception:
        return False


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=10_000)
    parser.add_argument("--diffusion-steps", type=int, default=20)
    parser.add_argument("--candidate-count", type=int, default=2)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--gpus", type=int, nargs="+", default=[0, 1])
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.steps < 1 or args.diffusion_steps < 1 or args.candidate_count < 1:
        parser.error("steps, diffusion-steps, and candidate-count must be positive")
    if args.steps % args.candidate_count:
        parser.error("--steps must be divisible by --candidate-count")
    if not args.gpus:
        parser.error("at least one GPU is required")

    for name, site in SITES.items():
        required = [
            site["data"] / "train.npz",
            site["data"] / "train.json",
            site["data"] / "validation.npz",
            site["calibration"],
        ]
        if site["case"] is not None:
            required.append(site["case"] / "run_module.py")
        missing = [str(path) for path in required if not path.exists()]
        if missing:
            raise FileNotFoundError(name + " missing required inputs: " + ", ".join(missing))

    status_path = CONTROL / "status.json"
    logs = CONTROL / "logs"
    configs = CONTROL / "configs"
    logs.mkdir(parents=True, exist_ok=True)
    configs.mkdir(parents=True, exist_ok=True)
    lock = threading.Lock()
    interval = args.steps // args.candidate_count
    candidate_steps = list(range(interval, args.steps + 1, interval))

    state = {
        "status": "dry_run" if args.dry_run else "running",
        "started_utc": utc_now(),
        "finished_utc": None,
        "model": "ctg_plus_plus",
        "scope": "CTG++ only on highway, TGSIM, and roundabout; no wall-clock deadline",
        "optimizer_steps": args.steps,
        "diffusion_steps": args.diffusion_steps,
        "candidate_steps": candidate_steps,
        "selection": "safety non-regression, then maximum PDMS on 16 validation scenes",
        "benchmark_scenarios": 20,
        "seeds": args.seeds,
        "gpus": args.gpus,
        "jobs": {},
    }
    write_json(status_path, state)

    site_configs: dict[str, Path] = {}
    for name, site in SITES.items():
        manifest = json.loads((site["data"] / "train.json").read_text(encoding="utf-8"))
        config = dict(manifest["config"])
        config["diffusion_steps"] = args.diffusion_steps
        config_path = configs / f"{name}.json"
        write_json(config_path, config)
        site_configs[name] = config_path
        state.setdefault("sites", {})[name] = {
            "config": str(config_path),
            "vehicle_length_m": config["length"],
            "vehicle_width_m": config["width"],
            "max_speed_mps": config["max_speed"],
            "result_dir": str(site["result"]),
        }
    write_json(status_path, state)

    def update(key: str, **values) -> None:
        with lock:
            state["jobs"].setdefault(key, {}).update(values)
            write_json(status_path, state)

    def command_prefix(site_name: str, benchmark: bool) -> list[str]:
        site = SITES[site_name]
        if site["case"] is not None:
            base = [sys.executable, "-u", str(site["case"] / "run_module.py")]
            return base if benchmark else [*base, "new_baselines_cli", "--threads", "1"]
        if benchmark:
            return [sys.executable, "-u", "-m"]
        return [sys.executable, "-u", str(ROOT / "New Baselines" / "run.py"), "--threads", "1"]

    def run_command(site_name: str, key: str, arguments: list[str], gpu: int,
                    *, benchmark: bool = False) -> None:
        site = SITES[site_name]
        command = [*command_prefix(site_name, benchmark), *map(str, arguments)]
        logfile = logs / f"{key}.log"
        update(key, site=site_name, status="planned" if args.dry_run else "running",
               gpu=gpu, command=command, log=str(logfile), started_utc=utc_now())
        if args.dry_run:
            return
        env = os.environ.copy()
        if site["run_env"]:
            env[site["run_env"]] = str(site["run"])
        env["CUDA_VISIBLE_DEVICES"] = str(gpu)
        env["OMP_NUM_THREADS"] = "1"
        env["MKL_NUM_THREADS"] = "1"
        started = time.monotonic()
        with logfile.open("w", encoding="utf-8") as handle:
            process = subprocess.Popen(
                command, cwd=str(ROOT), env=env, stdout=handle, stderr=subprocess.STDOUT
            )
            update(key, pid=process.pid)
            code = process.wait()  # Deliberately unbounded: allow CTG++ to finish.
        update(key, status="complete" if code == 0 else "failed", exit_code=code,
               wall_time_s=time.monotonic() - started, finished_utc=utc_now())
        if code:
            raise RuntimeError(f"{key} failed; see {logfile}")

    gpu_queue: Queue[int] = Queue()
    for gpu in args.gpus:
        gpu_queue.put(gpu)

    def run_parallel(jobs: list[tuple]) -> None:
        if not jobs:
            return

        def worker(job: tuple) -> None:
            gpu = gpu_queue.get()
            try:
                site_name, key, arguments, benchmark = job
                run_command(site_name, key, arguments, gpu, benchmark=benchmark)
            finally:
                gpu_queue.put(gpu)

        errors = []
        with ThreadPoolExecutor(max_workers=len(args.gpus)) as pool:
            futures = [pool.submit(worker, job) for job in jobs]
            for future in as_completed(futures):
                try:
                    future.result()
                except BaseException as exc:
                    errors.append(exc)
        if errors:
            raise errors[0]

    started = time.monotonic()
    try:
        train_jobs = []
        for site_name, site in SITES.items():
            for seed in args.seeds:
                raw = site["checkpoints"] / f"ctg_plus_plus_fast20_raw_seed{seed}.pt"
                candidates = raw.with_suffix(".candidates")
                expected = [candidates / f"step_{step:08d}.pt" for step in candidate_steps]
                key = f"{site_name}_seed{seed}_train"
                if all(checkpoint_matches(path, args.diffusion_steps) for path in expected):
                    update(key, site=site_name, status="reused", outputs=[str(p) for p in expected])
                    continue
                train_jobs.append((site_name, key, [
                    "train", "ctg_plus_plus", "--data", site["data"],
                    "--steps", str(args.steps), "--batch-size", "4",
                    "--eval-every", str(interval), "--seed", str(seed),
                    "--device", "cuda", "--config", site_configs[site_name],
                    "--output", raw,
                ], False))
        run_parallel(train_jobs)

        select_jobs = []
        for site_name, site in SITES.items():
            for seed in args.seeds:
                raw = site["checkpoints"] / f"ctg_plus_plus_fast20_raw_seed{seed}.pt"
                selected = site["checkpoints"] / f"ctg_plus_plus_policy_seed{seed}.pt"
                key = f"{site_name}_seed{seed}_select"
                if checkpoint_matches(selected, args.diffusion_steps, selected=True):
                    update(key, site=site_name, status="reused", output=str(selected))
                    continue
                select_jobs.append((site_name, key, [
                    "select", "ctg_plus_plus", "--candidates", raw.with_suffix(".candidates"),
                    "--device", "cuda", "--output", selected,
                    "--num-agents", str(site["num_agents"]),
                    "--max-steps", str(site["max_steps"]),
                    "--val-episodes", "16", "--validation-seed-start", "910000",
                    "--test-episodes", "16", "--test-seed-start", "810000",
                    "--run-id", str(site["run_id"]), "--lane-kf", str(site["lane_kf"]),
                    "--calibration", site["calibration"],
                ], False))
        run_parallel(select_jobs)

        benchmark_jobs = []
        for site_name, site in SITES.items():
            summary = site["result"] / "benchmark_summary.csv"
            key = f"{site_name}_benchmark"
            if summary.exists() and not args.dry_run:
                update(key, site=site_name, status="reused", output=str(summary))
                continue
            benchmark_jobs.append((site_name, key, [
                "Baselines.benchmark", "--models", "ctg_plus_plus",
                "--seed", "810000", "--scenarios", "20",
                "--num-agents", str(site["num_agents"]),
                "--max-steps", str(site["max_steps"]),
                "--calibration", site["calibration"],
                "--checkpoint-dir", site["checkpoints"],
                "--train-seeds", *map(str, args.seeds), "--require-matched-protocol",
                "--n-boot", "5000", "--output-dir", site["result"],
                "--conflict-lookahead", "full", "--no-figures",
            ], True))
        run_parallel(benchmark_jobs)
        state["status"] = "dry_run" if args.dry_run else "complete"
    except BaseException as exc:
        state["status"] = "failed"
        state["error"] = repr(exc)
        raise
    finally:
        state["finished_utc"] = utc_now()
        state["wall_time_s"] = time.monotonic() - started
        write_json(status_path, state)


if __name__ == "__main__":
    main()
