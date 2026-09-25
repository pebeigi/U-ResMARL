"""Train, select, and benchmark only CTG++ on TGSIM and Roundabout.

The two sites run concurrently on separate GPUs.  A global wall-clock deadline
terminates the current command trees so this add-on cannot run indefinitely.
Existing offline caches and every non-CTG checkpoint are reused unchanged.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time


ROOT = Path(__file__).resolve().parents[1]
SITES = {
    "tgsim": {
        "case": ROOT / "TGSIM Case",
        "run": ROOT / "TGSIM Case" / "runs" / "tgsim_recalibrated_fixed_20260922",
        "run_env": "TGSIM_RUN_DIR",
        "calibration": ROOT / "Calibration" / "utility_calibration_tgsim.json",
    },
    "roundabout": {
        "case": ROOT / "Roundabout Case",
        "run": ROOT / "Roundabout Case" / "runs" / "roundabout_3x_recalibrated_20260922",
        "run_env": "ROUNDABOUT_RUN_DIR",
        "calibration": ROOT / "Calibration" / "utility_calibration_jounieh.json",
    },
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def terminate_tree(process: subprocess.Popen) -> None:
    if process.poll() is not None:
        return
    if os.name == "nt":
        subprocess.run(
            ["taskkill", "/PID", str(process.pid), "/T", "/F"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )
    else:
        process.kill()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hours", type=float, default=5.75)
    parser.add_argument("--steps", type=int, default=10_000)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--gpu", type=int, nargs=2, default=[0, 1], metavar=("TGSIM", "ROUNDABOUT"))
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if not 0 < args.hours <= 6:
        parser.error("--hours must be in (0, 6]")
    if args.steps < 1:
        parser.error("--steps must be positive")

    for name, site in SITES.items():
        required = [
            site["case"] / "run_module.py",
            site["run"] / "data" / "train.npz",
            site["run"] / "data" / "validation.npz",
            site["calibration"],
        ]
        missing = [str(path) for path in required if not path.exists()]
        if missing:
            raise FileNotFoundError(name + " missing required inputs: " + ", ".join(missing))

    control = ROOT / "runs" / "ctg_plus_plus_sites_20260923"
    status_path = control / "status.json"
    logs = control / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    lock = threading.Lock()
    deadline = time.monotonic() + args.hours * 3600
    state = {
        "status": "dry_run" if args.dry_run else "running",
        "started_utc": utc_now(),
        "deadline_hours": args.hours,
        "deadline_utc": datetime.fromtimestamp(time.time() + args.hours * 3600, timezone.utc).isoformat(),
        "model": "ctg_plus_plus",
        "optimizer_steps": args.steps,
        "seeds": args.seeds,
        "scope": "CTG++ only; cached offline data reused; no other model is trained or evaluated",
        "sites": {},
    }
    write_json(status_path, state)

    def update(site_name: str, key: str, **values) -> None:
        with lock:
            site_state = state["sites"].setdefault(site_name, {"jobs": {}})
            job = site_state["jobs"].setdefault(key, {})
            job.update(values)
            write_json(status_path, state)

    def run_command(site_name: str, key: str, arguments: list[str], gpu: int) -> None:
        site = SITES[site_name]
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError("global CTG++ wall-clock limit reached")
        command = [sys.executable, "-u", str(site["case"] / "run_module.py"), *map(str, arguments)]
        logfile = logs / f"{site_name}_{key}.log"
        update(site_name, key, status="planned" if args.dry_run else "running", command=command,
               gpu=gpu, log=str(logfile), started_utc=utc_now())
        if args.dry_run:
            return
        env = os.environ.copy()
        env[site["run_env"]] = str(site["run"])
        env["CUDA_VISIBLE_DEVICES"] = str(gpu)
        env["OMP_NUM_THREADS"] = "1"
        env["MKL_NUM_THREADS"] = "1"
        started = time.monotonic()
        with logfile.open("w", encoding="utf-8") as handle:
            process = subprocess.Popen(command, cwd=str(ROOT), env=env, stdout=handle,
                                       stderr=subprocess.STDOUT)
            update(site_name, key, pid=process.pid)
            try:
                code = process.wait(timeout=max(1.0, remaining))
            except subprocess.TimeoutExpired:
                terminate_tree(process)
                update(site_name, key, status="timed_out", wall_time_s=time.monotonic() - started,
                       finished_utc=utc_now())
                raise TimeoutError("global CTG++ wall-clock limit reached")
        update(site_name, key, status="complete" if code == 0 else "failed", exit_code=code,
               wall_time_s=time.monotonic() - started, finished_utc=utc_now())
        if code:
            raise RuntimeError(f"{site_name} {key} failed; see {logfile}")

    def run_site(site_name: str, gpu: int) -> None:
        site = SITES[site_name]
        run = site["run"]
        checkpoints = run / "checkpoints"
        for seed in args.seeds:
            raw = checkpoints / f"ctg_plus_plus_raw_seed{seed}.pt"
            selected = checkpoints / f"ctg_plus_plus_policy_seed{seed}.pt"
            run_command(site_name, f"seed{seed}_train", [
                "new_baselines_cli", "--threads", "1", "train", "ctg_plus_plus",
                "--data", run / "data", "--steps", str(args.steps), "--batch-size", "4",
                "--eval-every", str(max(1, args.steps // 5)), "--seed", str(seed),
                "--device", "cuda", "--output", raw,
            ], gpu)
            run_command(site_name, f"seed{seed}_select", [
                "new_baselines_cli", "--threads", "1", "select", "ctg_plus_plus",
                "--candidates", raw.with_suffix(".candidates"), "--device", "cuda",
                "--output", selected, "--num-agents", "6", "--max-steps", "80",
                "--val-episodes", "16", "--validation-seed-start", "910000",
                "--test-episodes", "16", "--test-seed-start", "810000",
                "--calibration", site["calibration"],
            ], gpu)

        run_command(site_name, "benchmark", [
            "Baselines.benchmark", "--models", "ctg_plus_plus", "--seed", "810000",
            "--scenarios", "20", "--num-agents", "6", "--max-steps", "80",
            "--calibration", site["calibration"], "--checkpoint-dir", checkpoints,
            "--train-seeds", *map(str, args.seeds), "--require-matched-protocol",
            "--n-boot", "5000", "--output-dir", run / "results" / "ctg_plus_plus_only",
            "--conflict-lookahead", "full", "--no-figures",
        ], gpu)
        with lock:
            state["sites"][site_name]["status"] = "complete"
            write_json(status_path, state)

    if args.dry_run:
        for (site_name, _), gpu in zip(SITES.items(), args.gpu):
            run_site(site_name, gpu)
        return

    try:
        with ThreadPoolExecutor(max_workers=2) as pool:
            futures = {
                pool.submit(run_site, site_name, gpu): site_name
                for site_name, gpu in zip(SITES, args.gpu)
            }
            for future in as_completed(futures):
                site_name = futures[future]
                try:
                    future.result()
                except BaseException as exc:
                    with lock:
                        state["sites"].setdefault(site_name, {"jobs": {}})["status"] = "failed"
                        state["sites"][site_name]["error"] = repr(exc)
                        write_json(status_path, state)
                    raise
        state["status"] = "complete"
    except TimeoutError as exc:
        state["status"] = "timed_out"
        state["error"] = str(exc)
        raise
    except BaseException as exc:
        state["status"] = "failed"
        state["error"] = repr(exc)
        raise
    finally:
        state["finished_utc"] = utc_now()
        state["wall_time_s"] = args.hours * 3600 - max(0.0, deadline - time.monotonic())
        write_json(status_path, state)


if __name__ == "__main__":
    main()
