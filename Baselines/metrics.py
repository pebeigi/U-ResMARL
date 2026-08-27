"""Metrics computed identically for every benchmarked model."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

import Baselines._paths  # noqa: F401
from Baselines.runner import RolloutResult

UNSAFE_TTC = 1.5  # seconds


def _disc_centres(
    positions: np.ndarray,
    headings: np.ndarray,
    length: float,
) -> np.ndarray:
    """Two-disc approximation of the vehicle box: (n, 2, 2) front/rear centres."""
    offset = 0.25 * length
    direction = np.stack([np.cos(headings), np.sin(headings)], axis=-1)
    return np.stack([positions + offset * direction, positions - offset * direction], axis=1)


def _pairwise_ttc(
    centres: np.ndarray,
    velocities: np.ndarray,
    active: np.ndarray,
    disc_radius: float,
) -> tuple[float, float, int, int]:
    """Return (min surface gap, min TTC, unsafe pair count, evaluated pair count).

    Vehicles are approximated by two discs so that a car passing another
    laterally is not scored as an imminent collision, which a single
    circumscribed disc would do.
    """
    idx = np.flatnonzero(active)
    combined = 2.0 * disc_radius
    min_gap = float("inf")
    min_ttc = float("inf")
    unsafe = 0
    pairs = 0
    for a in range(len(idx)):
        for b in range(a + 1, len(idx)):
            i, j = idx[a], idx[b]
            rel_v = velocities[j] - velocities[i]
            a_q = float(rel_v @ rel_v)
            pair_ttc = float("inf")
            pairs += 1
            for ci in range(2):
                for cj in range(2):
                    rel_p = centres[j, cj] - centres[i, ci]
                    min_gap = min(min_gap, float(np.linalg.norm(rel_p)) - combined)
                    if a_q < 1e-9:
                        continue
                    b_q = 2.0 * float(rel_p @ rel_v)
                    c_q = float(rel_p @ rel_p) - combined * combined
                    disc = b_q * b_q - 4.0 * a_q * c_q
                    if disc <= 0.0:
                        continue
                    sqrt_disc = float(np.sqrt(disc))
                    roots = (
                        (-b_q - sqrt_disc) / (2.0 * a_q),
                        (-b_q + sqrt_disc) / (2.0 * a_q),
                    )
                    positive = [t for t in roots if t > 0.0]
                    if positive:
                        pair_ttc = min(pair_ttc, min(positive))
            if np.isfinite(pair_ttc):
                min_ttc = min(min_ttc, pair_ttc)
                if pair_ttc < UNSAFE_TTC:
                    unsafe += 1
    return min_gap, min_ttc, unsafe, pairs


def rollout_metrics(result: RolloutResult) -> dict[str, Any]:
    """Safety, efficiency, comfort and realism-relevant statistics for one rollout."""
    dt = result.dt
    n = result.num_agents
    active = result.active  # (T+1, n)
    steps = result.steps
    # Radius of each of the two discs covering the vehicle box.
    disc_radius = 0.5 * float(np.hypot(0.5 * result.vehicle_length, result.vehicle_width))

    # Velocities from finite differences of position (matches what the metrics see).
    velocities = np.zeros_like(result.positions)
    if result.positions.shape[0] > 1:
        velocities[1:] = (result.positions[1:] - result.positions[:-1]) / dt
        velocities[0] = velocities[1]

    active_steps = int(active[1 : steps + 1].sum()) if steps > 0 else 0
    active_steps = max(active_steps, 1)

    speeds = result.speeds[1 : steps + 1]
    mask = active[1 : steps + 1]
    speed_vals = speeds[mask] if mask.any() else np.array([0.0])

    accels = result.accels
    accel_vals = accels[mask[: accels.shape[0]]] if accels.size else np.array([0.0])
    if accels.shape[0] > 1:
        jerk = np.diff(accels, axis=0) / dt
        jerk_vals = jerk[mask[: jerk.shape[0]]] if jerk.size else np.array([0.0])
    else:
        jerk_vals = np.array([0.0])
    steer_vals = result.steerings[mask[: result.steerings.shape[0]]] if result.steerings.size else np.array([0.0])

    lateral_vals = result.lateral[1 : steps + 1][mask] if mask.any() else np.array([0.0])
    clearance_vals = result.clearance[1 : steps + 1][mask] if mask.any() else np.array([0.0])

    min_gaps, min_ttcs, unsafe_total, pair_total = [], [], 0, 0
    for t in range(1, steps + 1):
        centres = _disc_centres(result.positions[t], result.headings[t], result.vehicle_length)
        gap, ttc, unsafe, pairs = _pairwise_ttc(centres, velocities[t], active[t], disc_radius)
        if np.isfinite(gap):
            min_gaps.append(gap)
        if np.isfinite(ttc):
            min_ttcs.append(ttc)
        unsafe_total += unsafe
        pair_total += pairs

    arrived = result.arrival_step >= 0
    travel_times = result.arrival_step[arrived] * dt
    progress = result.station[min(steps, result.station.shape[0] - 1)] - result.start_s
    goal_progress = np.clip(progress / np.maximum(result.dest_s - result.start_s, 1e-6), 0.0, 1.0)

    return {
        "model": result.model,
        "seed": result.seed,
        "num_agents": n,
        "steps": steps,
        "episode_time_s": steps * dt,
        # Safety
        "collision_events": result.collision_events,
        "collision_steps": result.collision_steps,
        "collision_rate_per_agent": len(result.colliding_agents) / n,
        "collision_free": float(result.collision_events == 0),
        "offroad_rate": result.offroad_steps / active_steps,
        "offroad_agent_frac": len(result.offroad_agents) / n,
        "min_gap_m": float(np.min(min_gaps)) if min_gaps else float("nan"),
        "p5_gap_m": float(np.percentile(min_gaps, 5)) if min_gaps else float("nan"),
        "min_ttc_s": float(np.min(min_ttcs)) if min_ttcs else float("nan"),
        "unsafe_ttc_rate": unsafe_total / max(pair_total, 1),
        # Efficiency
        "arrival_rate": float(arrived.mean()),
        "goal_progress": float(np.mean(goal_progress)),
        "mean_travel_time_s": float(np.mean(travel_times)) if travel_times.size else float("nan"),
        "mean_speed_mps": float(np.mean(speed_vals)),
        "speed_std_mps": float(np.std(speed_vals)),
        # Comfort / plausibility
        "mean_abs_accel": float(np.mean(np.abs(accel_vals))),
        "rms_jerk": float(np.sqrt(np.mean(np.square(jerk_vals)))),
        "mean_abs_steering": float(np.mean(np.abs(steer_vals))),
        "mean_abs_lateral_m": float(np.mean(np.abs(lateral_vals))),
        "min_clearance_m": float(np.min(clearance_vals)) if clearance_vals.size else float("nan"),
        # Cost
        "wall_time_s": result.wall_time,
        "wall_time_per_agent_step_ms": 1000.0 * result.wall_time / max(steps * n, 1),
    }


def metrics_frame(results: list[RolloutResult]) -> pd.DataFrame:
    return pd.DataFrame([rollout_metrics(r) for r in results])


AGGREGATE_COLUMNS = [
    "collision_events",
    "collision_rate_per_agent",
    "collision_free",
    "offroad_rate",
    "min_gap_m",
    "min_ttc_s",
    "unsafe_ttc_rate",
    "arrival_rate",
    "goal_progress",
    "mean_travel_time_s",
    "mean_speed_mps",
    "mean_abs_accel",
    "rms_jerk",
    "mean_abs_steering",
    "mean_abs_lateral_m",
    "wall_time_per_agent_step_ms",
]


def aggregate(frame: pd.DataFrame, columns: list[str] | None = None) -> pd.DataFrame:
    """Mean and standard deviation across scenarios, one row per model."""
    cols = columns or [c for c in AGGREGATE_COLUMNS if c in frame.columns]
    grouped = frame.groupby("model", sort=False)[cols]
    mean = grouped.mean()
    std = grouped.std().fillna(0.0)
    out = pd.concat({"mean": mean, "std": std}, axis=1)
    out.columns = [f"{stat}_{col}" for stat, col in out.columns]
    ordered = []
    for col in cols:
        ordered.extend([f"mean_{col}", f"std_{col}"])
    return out[ordered].reset_index()


def to_latex(frame: pd.DataFrame, columns: list[str] | None = None, precision: int = 3) -> str:
    """Compact mean +/- std LaTeX table for the paper."""
    cols = columns or [
        "collision_events",
        "offroad_rate",
        "min_ttc_s",
        "arrival_rate",
        "mean_travel_time_s",
        "mean_speed_mps",
        "rms_jerk",
    ]
    grouped = frame.groupby("model", sort=False)
    rows = []
    for model, g in grouped:
        cells = [model.replace("_", r"\_")]
        for col in cols:
            if col not in g:
                cells.append("--")
                continue
            cells.append(f"{g[col].mean():.{precision}f} $\\pm$ {g[col].std(ddof=0):.{precision}f}")
        rows.append(" & ".join(cells) + r" \\")
    header = " & ".join(["Model"] + [c.replace("_", r"\_") for c in cols]) + r" \\"
    return "\n".join(
        [
            r"\begin{tabular}{l" + "c" * len(cols) + "}",
            r"\toprule",
            header,
            r"\midrule",
            *rows,
            r"\bottomrule",
            r"\end{tabular}",
        ]
    )
