"""Fit and audit fresh Jounieh, TGSIM and freeway utility priors.

Results remain in an isolated run directory until all three pass the protocol
audit. Existing calibration JSONs are not overwritten by this launcher.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
CASES = (
    ("jounieh", "data/Lebanon_Jounieh/prepared/trajectories_calibration.csv",
     ["--site-polygon-csv", "data/Lebanon_Jounieh/Jounieh_Road_Boundaries.csv",
      "--vehicle-length", "4.5", "--vehicle-width", "1.8", "--wheelbase", "2.8"]),
    ("tgsim", "data/TGSIM FB/prepared/trajectories_calibration.csv", []),
    ("highway", "data/Lebanon_Highway/Final_Lebanon_Data.csv", []),
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def audit(result_path: Path, site: str) -> dict:
    result = json.loads(result_path.read_text())
    bounds = result["search_bounds"]
    if result["calibration_site"] != site or bounds["xi_i"] != [1.1, 5.0]:
        raise ValueError(f"{site}: wrong site or speed-shape range")
    if result.get("choice_samples_partition") != "train_vehicle_keys_only":
        raise ValueError(f"{site}: one-step fitting used held-out vehicles")
    grid = result["candidate_grid"]
    if (len(grid["acceleration"]), len(grid["steering"])) != (7, 9):
        raise ValueError(f"{site}: calibration control grid differs from execution")
    split = result["window_split"]
    groups = [set(map(tuple, split[k])) for k in
              ("train_vehicle_keys", "validation_vehicle_keys", "test_vehicle_keys")]
    if groups[0] & groups[1] or groups[0] & groups[2] or groups[1] & groups[2]:
        raise ValueError(f"{site}: train/validation/test vehicles overlap")
    near_edges = []
    for name, value in result["working_params"].items():
        low, high = bounds[name]
        if not math.isfinite(value) or not low <= value <= high:
            raise ValueError(f"{site}: {name}={value} outside search range")
        if min(value - low, high - value) < 0.025 * (high - low):
            near_edges.append(name)
    if site == "jounieh":
        vehicle = result["calibration_vehicle"]
        if (vehicle["vehicle_length"], vehicle["vehicle_width"], vehicle["wheelbase"]) != (4.5, 1.8, 2.8):
            raise ValueError("Jounieh calibration used the wrong vehicle dimensions")
    if not math.isfinite(float(result["test_closed_loop_loss"])):
        raise ValueError(f"{site}: held-out test loss is not finite")
    return {
        "output": str(result_path), "sha256": sha256(result_path),
        "working_params_source": result["working_params_source"],
        "test_closed_loop_loss": result["test_closed_loop_loss"],
        "parameters_near_search_edge": near_edges,
        "n_choice_samples": result["n_samples"],
        "n_scored_trials": result["identifiability"]["n_scored_trials"],
    }


def write_status(path: Path, state: dict) -> None:
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(state, indent=2) + "\n")
    temp.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=ROOT / "Calibration/runs/recalibration_20260922_p99")
    args = parser.parse_args()
    run_dir = args.run_dir.resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    status_path = run_dir / "status.json"
    if status_path.exists():
        raise FileExistsError(f"Use a fresh calibration run directory: {status_path}")
    inputs = [ROOT / "Calibration/calibrate_utility_from_data.py",
              ROOT / "data/Lebanon_Jounieh/Jounieh_Road_Boundaries.csv"]
    inputs += [ROOT / row[1] for row in CASES]
    state = {"status": "running", "started_at": time.time(),
             "input_hashes": {str(p.relative_to(ROOT)): sha256(p) for p in inputs},
             "sites": {}}
    write_status(status_path, state)
    for site, csv, extra in CASES:
        if any(sha256(ROOT / name) != digest for name, digest in state["input_hashes"].items()):
            raise RuntimeError("Calibration input changed during the three-site fit")
        output = run_dir / f"utility_calibration_{site}.json"
        command = [sys.executable, "-u", "-m", "Calibration.calibrate_utility_from_data",
                   "--csv", str(ROOT / csv), "--output", str(output),
                   "--diagnostics-dir", str(run_dir / f"diagnostics_{site}"),
                   "--n-samples", "2000", "--n-trials", "400", "--n-restarts", "3",
                   "--closed-loop-candidates", "80", "--closed-loop-windows", "100",
                   "--closed-loop-val-windows", "40", "--closed-loop-test-windows", "40",
                   "--val-top-k", "10", "--no-plots", "--verbose", *extra]
        record = {"status": "running", "command": command, "started_at": time.time(),
                  "log": str(run_dir / f"{site}.log")}
        state["sites"][site] = record
        write_status(status_path, state)
        with Path(record["log"]).open("w", encoding="utf-8") as log:
            completed = subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
        record["exit_code"] = completed.returncode
        record["elapsed_seconds"] = time.time() - record["started_at"]
        if completed.returncode:
            record["status"] = "failed"
            state["status"] = "failed"
            write_status(status_path, state)
            raise RuntimeError(f"{site} calibration failed: {record['log']}")
        record["audit"] = audit(output, site)
        record["status"] = "complete"
        write_status(status_path, state)
        print(f"{site} complete: {output}", flush=True)
    state["status"] = "complete"
    state["elapsed_seconds"] = time.time() - state["started_at"]
    write_status(status_path, state)
    print(status_path, flush=True)


if __name__ == "__main__":
    main()
