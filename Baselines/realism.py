"""Behavioural realism: distance between simulated and observed distributions.

Compares the speed, longitudinal-acceleration and lateral-offset distributions
produced by each model against the measured trajectories on the same corridor
(same run_id / lane_kf), using the 1-Wasserstein distance and the
Jensen-Shannon divergence of matched histograms.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import Baselines._paths  # noqa: F401
from Baselines._paths import REPO_ROOT
from Baselines.runner import RolloutResult
from RL.corridor import load_corridor

DEFAULT_DATA_CSV = REPO_ROOT / "data" / "Lebanon_Highway" / "Final_Lebanon_Data.csv"
FEATURES = ("speed", "accel", "lateral")


@lru_cache(maxsize=8)
def observed_features(
    run_id: int,
    lane_kf: int,
    dt: float = 0.5,
    csv_path: str | None = None,
) -> dict[str, np.ndarray]:
    """Speed / acceleration / lateral-offset samples from the measured data."""
    path = Path(csv_path) if csv_path else DEFAULT_DATA_CSV
    df = pd.read_csv(path)
    mask = (df["run_id"] == int(run_id)) & (df["lane_kf"] == int(lane_kf))
    g = df.loc[mask].copy()
    if g.empty:
        raise ValueError(f"No observed trajectories for run_id={run_id}, lane_kf={lane_kf}")

    # Subsample to roughly the simulation timestep so the distributions are comparable.
    sample_dt = float(np.median(np.diff(np.sort(g["time"].unique()))))
    stride = max(1, int(round(dt / max(sample_dt, 1e-6))))
    g = g.sort_values(["id", "time"])
    g = g[g.groupby("id").cumcount() % stride == 0]

    corridor = load_corridor(int(run_id), int(lane_kf))
    lateral = np.array(
        [corridor.project(np.array([x, y]))[1] for x, y in zip(g["xloc_kf"], g["yloc_kf"])]
    )
    return {
        "speed": g["speed_kf"].to_numpy(float),
        "accel": g["acceleration_kf"].to_numpy(float),
        "lateral": lateral,
    }


def simulated_features(result: RolloutResult) -> dict[str, np.ndarray]:
    steps = result.steps
    if steps == 0:
        return {k: np.array([]) for k in FEATURES}
    mask = result.active[1 : steps + 1]
    speeds = result.speeds[1 : steps + 1]
    # Realised longitudinal acceleration, comparable to the data's acceleration_kf.
    accel = np.diff(result.speeds[: steps + 1], axis=0) / result.dt
    lateral = result.lateral[1 : steps + 1]
    return {
        "speed": speeds[mask],
        "accel": accel[mask],
        "lateral": lateral[mask],
    }


def _js_divergence(p: np.ndarray, q: np.ndarray, bins: int = 40) -> float:
    lo = float(min(p.min(), q.min()))
    hi = float(max(p.max(), q.max()))
    if hi - lo < 1e-9:
        return 0.0
    edges = np.linspace(lo, hi, bins + 1)
    hp, _ = np.histogram(p, bins=edges, density=False)
    hq, _ = np.histogram(q, bins=edges, density=False)
    hp = hp / max(hp.sum(), 1)
    hq = hq / max(hq.sum(), 1)
    m = 0.5 * (hp + hq)

    def _kl(a: np.ndarray, b: np.ndarray) -> float:
        nz = a > 0
        return float(np.sum(a[nz] * np.log(a[nz] / np.maximum(b[nz], 1e-12))))

    return float(0.5 * _kl(hp, m) + 0.5 * _kl(hq, m))


def realism_metrics(
    result: RolloutResult,
    csv_path: str | None = None,
) -> dict[str, Any]:
    """Wasserstein / JS distance per feature, plus a single averaged score."""
    from scipy.stats import wasserstein_distance

    observed = observed_features(result.run_id, result.lane_kf, result.dt, csv_path)
    simulated = simulated_features(result)

    out: dict[str, Any] = {"model": result.model, "seed": result.seed}
    wass = []
    for key in FEATURES:
        sim = simulated[key]
        obs = observed[key]
        if sim.size == 0:
            out[f"w1_{key}"] = float("nan")
            out[f"js_{key}"] = float("nan")
            continue
        w = float(wasserstein_distance(sim, obs))
        # Normalise by the observed spread so the features are commensurable.
        scale = float(np.std(obs)) or 1.0
        out[f"w1_{key}"] = w
        out[f"w1_{key}_norm"] = w / scale
        out[f"js_{key}"] = _js_divergence(sim, obs)
        wass.append(w / scale)
    out["realism_score"] = float(np.mean(wass)) if wass else float("nan")
    return out


def realism_frame(results: list[RolloutResult], csv_path: str | None = None) -> pd.DataFrame:
    return pd.DataFrame([realism_metrics(r, csv_path) for r in results])
