"""Measured behavioral marginals used as a data term in the residual RL reward.

The realism metric compares simulated and observed distributions of speed,
longitudinal acceleration and lateral offset. Training only on progress /
safety / smoothness leaves that objective unoptimized, so this module exposes
the same three marginals to the environment:

* ``step_deviation`` — dense per-step penalty for leaving the observed central
  band of each feature (zero inside the band, so it does not flatten the
  within-band spread).
* ``distribution_distance`` — normalized 1-Wasserstein distance between an
  episode's simulated marginals and the observed ones, i.e. the quantity the
  reported realism score measures.

The Wasserstein distance is computed from quantile functions so training does
not depend on SciPy.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

import RL._paths  # noqa: F401
from RL._paths import REPO_ROOT
from RL.corridor import load_corridor

DEFAULT_DATA_CSV = REPO_ROOT / "data" / "Lebanon_Highway" / "Final_Lebanon_Data.csv"
FEATURES = ("speed", "accel", "lateral")

# Quantile band treated as "observed normal behavior" by the dense penalty. Kept
# wide so legitimate braking (which the data does contain, in the tails) is not
# penalized; distribution *shape* is handled by the shaping term instead.
DEFAULT_BAND = (2.5, 97.5)
# Deviations are measured in observed standard deviations and capped so a single
# badly behaved agent cannot dominate the return.
DEFAULT_DEVIATION_CAP = 3.0
_QUANTILE_GRID = np.linspace(0.005, 0.995, 199)


@dataclass(frozen=True)
class FeatureReference:
    low: float
    high: float
    scale: float
    quantiles: np.ndarray


@dataclass(frozen=True)
class BehaviorReference:
    """Observed marginals for one corridor at the simulation timestep."""

    run_id: int
    lane_kf: int
    dt: float
    features: dict[str, FeatureReference]
    deviation_cap: float = DEFAULT_DEVIATION_CAP

    def _deviation(self, name: str, value: float) -> float:
        ref = self.features[name]
        if value < ref.low:
            excess = ref.low - value
        elif value > ref.high:
            excess = value - ref.high
        else:
            return 0.0
        return min(excess / ref.scale, self.deviation_cap)

    def step_deviation(self, speed: float, accel: float, lateral: float) -> float:
        """Mean capped out-of-band deviation across the three marginals."""
        values = {"speed": float(speed), "accel": float(accel), "lateral": float(lateral)}
        return float(np.mean([self._deviation(k, v) for k, v in values.items()]))

    def distribution_distance(self, samples: dict[str, np.ndarray]) -> float:
        """Normalized W1 between simulated and observed marginals (lower is better)."""
        distances = []
        for name, ref in self.features.items():
            sim = np.asarray(samples.get(name, ()), dtype=float)
            sim = sim[np.isfinite(sim)]
            if sim.size == 0:
                continue
            sim_q = np.quantile(sim, _QUANTILE_GRID)
            distances.append(float(np.mean(np.abs(sim_q - ref.quantiles)) / ref.scale))
        return float(np.mean(distances)) if distances else 0.0


@lru_cache(maxsize=8)
def load_behavior_reference(
    run_id: int,
    lane_kf: int,
    dt: float = 0.5,
    csv_path: str | None = None,
    band: tuple[float, float] = DEFAULT_BAND,
    deviation_cap: float = DEFAULT_DEVIATION_CAP,
) -> BehaviorReference:
    path = Path(csv_path) if csv_path else DEFAULT_DATA_CSV
    df = pd.read_csv(path)
    mask = (df["run_id"] == int(run_id)) & (df["lane_kf"] == int(lane_kf))
    g = df.loc[mask].copy()
    if g.empty:
        raise ValueError(f"No observed trajectories for run_id={run_id}, lane_kf={lane_kf}")

    # Subsample to the control interval so the marginals are comparable to the sim.
    sample_dt = float(np.median(np.diff(np.sort(g["time"].unique()))))
    stride = max(1, int(round(dt / max(sample_dt, 1e-6))))
    g = g.sort_values(["id", "time"])
    g = g[g.groupby("id").cumcount() % stride == 0]

    corridor = load_corridor(int(run_id), int(lane_kf))
    lateral = np.array(
        [corridor.project(np.array([x, y]))[1] for x, y in zip(g["xloc_kf"], g["yloc_kf"])],
        dtype=float,
    )
    observed = {
        "speed": g["speed_kf"].to_numpy(float),
        "accel": g["acceleration_kf"].to_numpy(float),
        "lateral": lateral,
    }

    features: dict[str, FeatureReference] = {}
    for name, values in observed.items():
        values = values[np.isfinite(values)]
        low, high = np.percentile(values, band)
        scale = float(np.std(values))
        features[name] = FeatureReference(
            low=float(low),
            high=float(high),
            scale=scale if scale > 1e-6 else 1.0,
            quantiles=np.quantile(values, _QUANTILE_GRID),
        )
    return BehaviorReference(
        run_id=int(run_id),
        lane_kf=int(lane_kf),
        dt=float(dt),
        features=features,
        deviation_cap=float(deviation_cap),
    )
