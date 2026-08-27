#!/usr/bin/env python
"""Global Sobol GSA of the NR-MAP utility prior on the calibrated percentile box.

This is not the pre-recalibration 10-D ad-hoc scan. It:
  - varies all twelve utility coordinates, including (sigma_long, sigma_lat);
  - draws independent Sobol samples from the top-cloud p05--p95 box in
    calibration/utility_calibration.json (the box contains theta_rob);
  - scores candidates with utility_model (OBB surface-gap collision kernel);
  - counts oriented-box overlaps in the simulation-level output Y.

Sobol still treats coordinates as independent; the identifiability cloud's
S_v--w_c correlation is not encoded in the design.
"""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

import Sensitivity._paths  # noqa: F401
from Sensitivity._paths import REPO_ROOT
from RL.corridor import boxes_overlap
from utility_model import (
    DEFAULT_SIM_CONFIG,
    UTILITY_PARAM_KEYS,
    TrafficAgent,
    select_best_candidate,
)

CALIBRATION_JSON = REPO_ROOT / "calibration" / "utility_calibration.json"
OUTPUT_DIR = REPO_ROOT / "Sensitivity" / "results"

# Closed-loop-style collision lookahead (matches RL/traffic_env.py, not calibration).
N_STEPS = 80
N_AGENTS = 3
COLLISION_PENALTY_Y = 10.0


def load_top_cloud_bounds() -> dict:
    payload = json.loads(CALIBRATION_JSON.read_text(encoding="utf-8"))
    ranges = payload["recommended_ranges_from_top_trials"]
    bounds = []
    for key in UTILITY_PARAM_KEYS:
        lo = float(ranges[key]["p05"])
        hi = float(ranges[key]["p95"])
        if hi <= lo:
            raise ValueError(f"{key}: p95 ({hi}) is not greater than p05 ({lo})")
        bounds.append([lo, hi])
    return {
        "num_vars": len(UTILITY_PARAM_KEYS),
        "names": list(UTILITY_PARAM_KEYS),
        "bounds": bounds,
    }


def _sim_config() -> dict:
    cfg = dict(DEFAULT_SIM_CONFIG)
    cfg.update(
        {
            "dt": 0.5,
            "total_time_steps": N_STEPS,
            "destination_threshold": 2.0,
            "kappa_perception_horizon": 2.0,
            "min_perception_horizon": 5.0,
            "vehicle_length": 4.5,
            "vehicle_width": 1.8,
            "max_agent_speed": 12.0,
            "path_mode": "boundary",
            "road_y_min": -4.0,
            "road_y_max": 6.0,
            "boundary_buffer": 1.5,
            "utility_frame": "destination",
            "use_obb_collisions": True,
            "conflict_horizon": 1.5,
            "conflict_substeps": 4,
            "steering_penalty_weight": 0.5,
        }
    )
    return cfg


def _spawn_agents() -> list[TrafficAgent]:
    """Leader--follower stream plus a merging agent so OBB contacts can occur."""
    return [
        TrafficAgent(
            agent_id=0,
            pos=np.array([0.0, 0.0]),
            vel=np.array([7.0, 0.0]),
            dest=np.array([36.0, 0.0]),
            desired_speed=8.0,
            nominal_y=0.0,
        ),
        TrafficAgent(
            agent_id=1,
            pos=np.array([9.0, 0.2]),
            vel=np.array([5.0, 0.0]),
            dest=np.array([36.0, 0.0]),
            desired_speed=8.0,
            nominal_y=0.0,
        ),
        TrafficAgent(
            agent_id=2,
            pos=np.array([6.0, 5.5]),
            vel=np.array([4.0, -2.5]),
            dest=np.array([32.0, 0.5]),
            desired_speed=7.0,
            nominal_y=0.0,
        ),
    ]


def _count_obb_overlaps(agents: list[TrafficAgent], length: float, width: float) -> int:
    hits = 0
    for i in range(len(agents)):
        if agents[i].reached_destination:
            continue
        for j in range(i + 1, len(agents)):
            if agents[j].reached_destination:
                continue
            if boxes_overlap(
                agents[i].pos,
                float(agents[i].heading_angle),
                agents[j].pos,
                float(agents[j].heading_angle),
                length,
                width,
            ):
                hits += 1
    return hits


def evaluate_y(theta: np.ndarray) -> float:
    """Y = residual destination distance + (OBB overlap-steps) * C_penalty."""
    params = {key: float(value) for key, value in zip(UTILITY_PARAM_KEYS, np.asarray(theta, dtype=float))}
    sim_config = _sim_config()
    agents = _spawn_agents()
    length = float(sim_config["vehicle_length"])
    width = float(sim_config["vehicle_width"])
    dt = float(sim_config["dt"])
    dest_eps = float(sim_config["destination_threshold"])

    overlap_steps = 0
    for _ in range(int(sim_config["total_time_steps"])):
        if all(agent.reached_destination for agent in agents):
            break
        chosen = []
        for idx, agent in enumerate(agents):
            if agent.reached_destination:
                chosen.append(None)
                continue
            chosen.append(select_best_candidate(idx, agent, agents, params, sim_config))
        for agent, candidate in zip(agents, chosen):
            if candidate is None:
                continue
            agent.update_state_from_candidate(candidate, dt, dest_eps)
        overlap_steps += _count_obb_overlaps(agents, length, width)

    leftover = 0.0
    for agent in agents:
        if not agent.reached_destination:
            leftover += float(np.linalg.norm(agent.pos - agent.dest))
    return leftover + overlap_steps * COLLISION_PENALTY_Y


def _theta_rob() -> np.ndarray:
    payload = json.loads(CALIBRATION_JSON.read_text(encoding="utf-8"))
    rob = payload["robust_params"]
    return np.array([float(rob[key]) for key in UTILITY_PARAM_KEYS], dtype=float)


def run_smoke() -> None:
    theta = _theta_rob()
    print("theta_rob:")
    for key, value in zip(UTILITY_PARAM_KEYS, theta):
        print(f"  {key:12s} {value:10.4f}")
    y = evaluate_y(theta)
    print(f"Y(theta_rob) = {y:.4f}")


def run_sobol(n_samples: int, workers: int) -> None:
    from SALib.analyze import sobol
    from SALib.sample import sobol as sobol_sample
    import pandas as pd

    # Re-import under the package name so Windows spawn can pickle the worker.
    from Sensitivity.Dis_Sensitivity_Analysis import evaluate_y as eval_fn

    problem = load_top_cloud_bounds()
    samples = sobol_sample.sample(problem, n_samples, calc_second_order=False)
    n_eval = int(samples.shape[0])
    print(
        f"Sobol N={n_samples}, D={problem['num_vars']}, "
        f"N*(D+2)={n_eval}, workers={workers}",
        flush=True,
    )
    for key, (lo, hi) in zip(problem["names"], problem["bounds"]):
        print(f"  {key:12s} [{lo:.4g}, {hi:.4g}]", flush=True)

    y_list = []
    if workers <= 1:
        for i, row in enumerate(samples):
            y_list.append(eval_fn(row))
            if (i + 1) % 50 == 0 or i == 0 or i + 1 == n_eval:
                print(f"  {i + 1}/{n_eval}", flush=True)
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            for i, y in enumerate(pool.map(eval_fn, samples, chunksize=8), start=1):
                y_list.append(y)
                if i % 200 == 0 or i == n_eval:
                    print(f"  {i}/{n_eval}", flush=True)
    y_values = np.asarray(y_list, dtype=float)

    analysis = sobol.analyze(problem, y_values, calc_second_order=False, print_to_console=False)
    frame = pd.DataFrame(
        {
            "Parameter": problem["names"],
            "S1": analysis["S1"],
            "S1_conf": analysis["S1_conf"],
            "ST": analysis["ST"],
            "ST_conf": analysis["ST_conf"],
        }
    )
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = OUTPUT_DIR / "sobol_calibrated.csv"
    tex_path = OUTPUT_DIR / "sobol_calibrated.tex"
    frame.to_csv(csv_path, index=False)
    lines = [
        r"\begin{tabular}{lcccc}",
        r"\hline",
        r"Parameter & \(S_1\) & \(S_{1,\mathrm{conf}}\) & \(S_T\) & \(S_{T,\mathrm{conf}}\)\\",
        r"\hline",
    ]
    latex_names = {
        "S_theta": r"\(S_{\theta}\)",
        "S_v": r"\(S_v\)",
        "xi_i": r"\(\xi_i\)",
        "S_d": r"\(S_d\)",
        "gamma": r"\(\gamma\)",
        "w_x": r"\(w_x\)",
        "w_y": r"\(w_y\)",
        "w_c": r"\(w_c\)",
        "w_ell": r"\(w_{\ell}\)",
        "beta": r"\(\beta\)",
        "sigma_long": r"\(\sigma_{\parallel}\)",
        "sigma_lat": r"\(\sigma_{\perp}\)",
    }
    for row in frame.itertuples(index=False):
        label = latex_names.get(row.Parameter, row.Parameter)
        lines.append(
            f"{label} & {row.S1:.3f} & {row.S1_conf:.3f} & {row.ST:.3f} & {row.ST_conf:.3f}\\\\"
        )
    lines.extend([r"\hline", r"\end{tabular}", ""])
    tex_path.write_text("\n".join(lines), encoding="utf-8")
    print(frame.to_string(index=False, float_format="%.4f"))
    print(f"Y mean={y_values.mean():.3f} std={y_values.std():.3f} "
          f"min={y_values.min():.3f} max={y_values.max():.3f}")
    print(f"Wrote {csv_path}")
    print(f"Wrote {tex_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Calibrated-box Sobol GSA for NR-MAP utility")
    parser.add_argument("--mode", choices=("smoke", "sobol"), default="sobol")
    parser.add_argument("--n-samples", type=int, default=1024, help="Sobol base sample size N")
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()
    if args.mode == "smoke":
        run_smoke()
        return
    run_sobol(n_samples=args.n_samples, workers=args.workers)


if __name__ == "__main__":
    main()
