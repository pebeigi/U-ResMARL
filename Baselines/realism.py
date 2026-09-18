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
PAIRED_FEATURES = FEATURES + ("yaw_rate", "gap", "ttc")
# Normalization floors prevent nearly constant reference scenes dominating.
SCALE_FLOORS = dict(speed=.1, accel=.1, lateral=.1, yaw_rate=.01, gap=1., ttc=1.)
GAP_CAP_M = 60.
TTC_CAP_S = 10.


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
    # q is the reference. Identical observed scenes must give every model the
    # same bin edges; explicit tail bins retain out-of-reference predictions.
    lo = float(q.min())
    hi = float(q.max())
    if hi - lo < 1e-9:
        lo -= 1e-6
        hi += 1e-6
    edges = np.r_[-np.inf, np.linspace(np.nextafter(lo, -np.inf),
                                      np.nextafter(hi, np.inf), bins + 1), np.inf]
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


def interaction_features(positions, velocities, headings, valid, length, width):
    """Per-agent nearest two-disc gap and earliest constant-velocity TTC.

    Both features are capped, including when there is no neighbor/contact.
    Keeping safe TTC values avoids comparing only dangerous encounters.
    This approximation is shared with benchmark TTC; it is not an OBB contact.
    """
    from Baselines.metrics import _disc_centres

    n = len(positions)
    gap = np.full(n, np.nan)
    ttc = np.full(n, np.nan)
    ids = np.flatnonzero(valid & np.isfinite(velocities).all(axis=-1))
    if ids.size == 0:
        return gap, ttc
    p, v, h = positions[ids], velocities[ids], headings[ids]
    centres = _disc_centres(p, h, length)
    rel_p = centres[None, :, None, :, :] - centres[:, None, :, None, :]
    rel_v = v[None, :, :] - v[:, None, :]
    combined_radius = float(np.hypot(.5 * length, width))
    squared_distance = np.sum(rel_p ** 2, axis=-1)
    pairs = ~np.eye(len(ids), dtype=bool)
    pair_gap = np.sqrt(squared_distance).min(axis=(2, 3)) - combined_radius
    gap[ids] = np.minimum(np.where(pairs, pair_gap, np.inf).min(axis=1), GAP_CAP_M)
    a = np.sum(rel_v ** 2, axis=-1)[:, :, None, None]
    b = 2 * np.sum(rel_p * rel_v[:, :, None, None, :], axis=-1)
    c = squared_distance - combined_radius ** 2
    disc = b ** 2 - 4 * a * c
    root = (-b - np.sqrt(np.maximum(disc, 0))) / np.maximum(2 * a, 1e-12)
    hits = (a > 1e-9) & (disc >= 0) & (root >= 0)
    values = np.where(c <= 0, 0., np.where(hits, root, TTC_CAP_S))
    pair_ttc = values.min(axis=(2, 3))
    ttc[ids] = np.minimum(np.where(pairs, pair_ttc, np.inf).min(axis=1), TTC_CAP_S)
    return gap, ttc


def paired_feature_samples(result: RolloutResult, scene) -> tuple[dict, dict]:
    """Match time/agent coverage, using identical derivatives on both traces."""
    from Baselines.data_evaluation import aligned_motion

    simulated, observed = aligned_motion(result, scene)
    first = scene.history_steps + 1
    samples = []
    # Both sides use reference coverage, never the policy's active/arrival mask.
    valid = observed["valid"] & np.isfinite(observed["velocity"]).all(axis=-1)
    for motion in (simulated, observed):
        values = {key: [] for key in PAIRED_FEATURES}
        for t in range(first, len(motion["positions"])):
            mask = valid[t]
            for key in ("speed", "accel", "yaw_rate"):
                common = mask & np.isfinite(observed[key][t])
                values[key].extend(motion[key][t, common])
            values["lateral"].extend(scene.scenario.corridor.project(p)[1]
                                     for p in motion["positions"][t, mask])
            gap, ttc = interaction_features(
                motion["positions"][t], motion["velocity"][t], motion["heading"][t],
                mask, result.vehicle_length, result.vehicle_width)
            values["gap"].extend(gap[mask])
            values["ttc"].extend(ttc[mask])
        samples.append({key: np.asarray(value, dtype=float) for key, value in values.items()})
    return samples[0], samples[1]


def paired_realism_metrics(result: RolloutResult, scene) -> dict[str, Any]:
    """Scene-conditioned W1/JS; lower is better, including realism_score."""
    from scipy.stats import wasserstein_distance

    simulated, observed = paired_feature_samples(result, scene)
    out, normalized = {}, []
    for key in PAIRED_FEATURES:
        sim, obs = simulated[key], observed[key]
        if sim.size != obs.size or not np.isfinite(sim).all() or not np.isfinite(obs).all():
            raise ValueError(f"Invalid matched {key} samples")
        out[f"realism_samples_{key}"] = int(obs.size)
        if not obs.size:
            out.update({f"w1_{key}": float("nan"), f"w1_{key}_norm": float("nan"),
                        f"js_{key}": float("nan")})
            continue
        distance = float(wasserstein_distance(sim, obs))
        scale = max(float(np.std(obs)), SCALE_FLOORS[key])
        out.update({f"w1_{key}": distance, f"w1_{key}_norm": distance / scale,
                    f"js_{key}": _js_divergence(sim, obs)})
        normalized.append(distance / scale)
    out["realism_score"] = float(np.mean(normalized)) if normalized else float("nan")
    return out
