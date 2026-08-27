"""Uncertainty quantification for the closed-loop benchmark.

Two independent sources of randomness affect a learned controller's score:

* **scenario sampling** — spawn stations, lateral offsets and desired speeds, and
* **policy training** — the PPO seed, which for residual learning also fixes the
  exploration path through parameter-residual space.

Averaging over scenarios only (the usual practice) reports the second source as
zero, so a gap between two learned models can be indistinguishable from training
noise. Learned models are therefore trained with several seeds, and their
summaries use the *training seed* as the resampling unit: scenario means are
computed within each seed first, so a model is not credited for having been
evaluated on more scenarios.

Model-vs-model claims use scenarios as matched pairs, since every controller sees
byte-identical initial conditions for a given scenario seed. Resampling is
hierarchical where applicable: training seeds and scenarios are drawn with
replacement, and the scenario draw is shared between the two models being
compared so the pairing is preserved.
"""

from __future__ import annotations

import zlib
from dataclasses import dataclass, asdict
from typing import Any, Iterable

import numpy as np
import pandas as pd

import Baselines._paths  # noqa: F401

# train_seed value used for controllers with no trained weights (ORCA, DWA, ...)
# and for learned models evaluated from a single checkpoint.
NO_TRAIN_SEED = -1
DEFAULT_N_BOOT = 10000
DEFAULT_ALPHA = 0.05


@dataclass
class MetricSummary:
    """Point estimate and interval for one model / one metric."""

    model: str
    metric: str
    mean: float
    std: float
    ci_low: float
    ci_high: float
    n_units: int
    n_scenarios: int
    resampling_unit: str

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)

    def format(self, precision: int = 3) -> str:
        if not np.isfinite(self.mean):
            return "--"
        half = 0.5 * (self.ci_high - self.ci_low)
        return f"{self.mean:.{precision}f} $\\pm$ {half:.{precision}f}"


def _percentile_ci(draws: np.ndarray, alpha: float) -> tuple[float, float]:
    draws = draws[np.isfinite(draws)]
    if draws.size == 0:
        return float("nan"), float("nan")
    lo, hi = np.percentile(draws, [100.0 * alpha / 2.0, 100.0 * (1.0 - alpha / 2.0)])
    return float(lo), float(hi)


def bootstrap_mean_ci(
    values: Iterable[float],
    n_boot: int = DEFAULT_N_BOOT,
    alpha: float = DEFAULT_ALPHA,
    seed: int = 0,
) -> tuple[float, float, float]:
    """Percentile bootstrap CI for the mean of ``values``."""
    arr = np.asarray(list(values), dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float("nan"), float("nan"), float("nan")
    if arr.size == 1:
        return float(arr[0]), float(arr[0]), float(arr[0])
    rng = np.random.default_rng(seed)
    draws = rng.choice(arr, size=(n_boot, arr.size), replace=True).mean(axis=1)
    lo, hi = _percentile_ci(draws, alpha)
    return float(arr.mean()), lo, hi


def _seed_column(frame: pd.DataFrame, seed_col: str) -> pd.Series:
    if seed_col not in frame.columns:
        return pd.Series(NO_TRAIN_SEED, index=frame.index, dtype=float)
    return frame[seed_col].fillna(NO_TRAIN_SEED)


def summarize_metric(
    frame: pd.DataFrame,
    metric: str,
    model: str,
    model_col: str = "model",
    seed_col: str = "train_seed",
    scenario_col: str = "seed",
    n_boot: int = DEFAULT_N_BOOT,
    alpha: float = DEFAULT_ALPHA,
) -> MetricSummary:
    """Summarize one metric for one model, resampling training seeds when available."""
    sub = frame[frame[model_col] == model]
    if metric not in sub.columns or sub.empty:
        return MetricSummary(model, metric, *([float("nan")] * 4), 0, 0, "none")

    train_seeds = _seed_column(sub, seed_col)
    distinct = sorted({s for s in train_seeds.unique() if s != NO_TRAIN_SEED})
    n_scenarios = int(sub[scenario_col].nunique()) if scenario_col in sub.columns else len(sub)

    if len(distinct) > 1:
        # Scenario means within each training seed, then resample seeds.
        per_seed = sub.groupby(train_seeds)[metric].mean()
        unit_values = per_seed.to_numpy(dtype=float)
        unit = "train_seed"
    else:
        unit_values = sub[metric].to_numpy(dtype=float)
        unit = "scenario"

    # Stable across processes, unlike hash() on str.
    boot_seed = int(zlib.crc32(f"{model}:{metric}".encode()))
    mean, lo, hi = bootstrap_mean_ci(unit_values, n_boot=n_boot, alpha=alpha, seed=boot_seed)
    finite = unit_values[np.isfinite(unit_values)]
    std = float(np.std(finite, ddof=1)) if finite.size > 1 else 0.0
    return MetricSummary(
        model=model,
        metric=metric,
        mean=float(mean),
        std=std,
        ci_low=lo,
        ci_high=hi,
        n_units=int(finite.size),
        n_scenarios=n_scenarios,
        resampling_unit=unit,
    )


def summary_frame(
    frame: pd.DataFrame,
    metrics: list[str],
    models: list[str] | None = None,
    **kwargs: Any,
) -> pd.DataFrame:
    """One row per (model, metric) with mean, std and bootstrap CI."""
    model_list = models if models is not None else list(dict.fromkeys(frame["model"]))
    rows = [
        summarize_metric(frame, metric, model, **kwargs).as_dict()
        for model in model_list
        for metric in metrics
    ]
    return pd.DataFrame(rows)


def _cell_matrix(
    frame: pd.DataFrame,
    metric: str,
    model: str,
    model_col: str,
    seed_col: str,
    scenario_col: str,
) -> tuple[np.ndarray, list[Any], list[Any]]:
    """(n_train_seeds, n_scenarios) metric values for one model."""
    sub = frame[frame[model_col] == model].copy()
    sub["_train_seed"] = _seed_column(sub, seed_col)
    pivot = sub.pivot_table(index="_train_seed", columns=scenario_col, values=metric, aggfunc="mean")
    return pivot.to_numpy(dtype=float), list(pivot.index), list(pivot.columns)


def compare_models(
    frame: pd.DataFrame,
    metric: str,
    model_a: str,
    model_b: str,
    model_col: str = "model",
    seed_col: str = "train_seed",
    scenario_col: str = "seed",
    n_boot: int = DEFAULT_N_BOOT,
    alpha: float = DEFAULT_ALPHA,
    seed: int = 0,
) -> dict[str, Any]:
    """Paired comparison of ``model_a`` minus ``model_b`` on shared scenarios.

    Scenarios are resampled jointly (preserving the matched design) and, for
    learned models, training seeds are resampled within each model.
    """
    mat_a, seeds_a, scen_a = _cell_matrix(frame, metric, model_a, model_col, seed_col, scenario_col)
    mat_b, seeds_b, scen_b = _cell_matrix(frame, metric, model_b, model_col, seed_col, scenario_col)
    shared = [s for s in scen_a if s in set(scen_b)]
    out: dict[str, Any] = {
        "metric": metric,
        "model_a": model_a,
        "model_b": model_b,
        "n_shared_scenarios": len(shared),
        "n_train_seeds_a": len(seeds_a),
        "n_train_seeds_b": len(seeds_b),
    }
    if not shared:
        out.update({k: float("nan") for k in ("diff_mean", "diff_ci_low", "diff_ci_high", "p_bootstrap")})
        return out

    cols_a = [scen_a.index(s) for s in shared]
    cols_b = [scen_b.index(s) for s in shared]
    a = mat_a[:, cols_a]
    b = mat_b[:, cols_b]

    diff_mean = float(np.nanmean(np.nanmean(a, axis=0) - np.nanmean(b, axis=0)))

    rng = np.random.default_rng(seed)
    n_scen = len(shared)
    draws = np.empty(n_boot, dtype=float)
    for k in range(n_boot):
        scen_idx = rng.integers(0, n_scen, n_scen)
        rows_a = rng.integers(0, a.shape[0], a.shape[0]) if a.shape[0] > 1 else np.arange(a.shape[0])
        rows_b = rng.integers(0, b.shape[0], b.shape[0]) if b.shape[0] > 1 else np.arange(b.shape[0])
        sample_a = np.nanmean(a[np.ix_(rows_a, scen_idx)], axis=0)
        sample_b = np.nanmean(b[np.ix_(rows_b, scen_idx)], axis=0)
        draws[k] = np.nanmean(sample_a - sample_b)

    lo, hi = _percentile_ci(draws, alpha)
    finite = draws[np.isfinite(draws)]
    if finite.size:
        # Both tails count mass exactly at zero, so two identical models give p = 1
        # rather than p = 0.
        p_left = float(np.mean(finite <= 0.0))
        p_right = float(np.mean(finite >= 0.0))
        p_two = float(min(1.0, 2.0 * min(p_left, p_right)))
    else:
        p_two = float("nan")
    out.update(
        {
            "diff_mean": diff_mean,
            "diff_ci_low": lo,
            "diff_ci_high": hi,
            "p_bootstrap": p_two,
            "significant_at_alpha": bool(np.isfinite(lo) and np.isfinite(hi) and (lo > 0.0 or hi < 0.0)),
        }
    )
    out.update(_wilcoxon_paired(np.nanmean(a, axis=0), np.nanmean(b, axis=0)))
    return out


def _wilcoxon_paired(a: np.ndarray, b: np.ndarray) -> dict[str, Any]:
    """Wilcoxon signed-rank test over scenarios, after averaging training seeds.

    Reported for reference only: it treats the seed-averaged score as fixed and so
    ignores training variance, which makes it anticonservative for learned models.
    Headline claims should use ``diff_ci_*``, which resamples seeds as well.
    """
    key = "p_wilcoxon_scenario_level"
    try:
        from scipy.stats import wilcoxon
    except Exception:
        return {key: float("nan")}
    mask = np.isfinite(a) & np.isfinite(b)
    if mask.sum() < 3 or np.allclose(a[mask], b[mask]):
        return {key: float("nan")}
    try:
        return {key: float(wilcoxon(a[mask], b[mask]).pvalue)}
    except Exception:
        return {key: float("nan")}


def comparison_frame(
    frame: pd.DataFrame,
    metrics: list[str],
    reference: str,
    models: list[str] | None = None,
    **kwargs: Any,
) -> pd.DataFrame:
    """Paired comparisons of every model against ``reference``."""
    model_list = models if models is not None else list(dict.fromkeys(frame["model"]))
    rows = [
        compare_models(frame, metric, model, reference, **kwargs)
        for model in model_list
        if model != reference
        for metric in metrics
    ]
    return pd.DataFrame(rows)


def to_latex_ci(
    frame: pd.DataFrame,
    metrics: list[str],
    labels: dict[str, str] | None = None,
    models: list[str] | None = None,
    precision: int = 3,
    **kwargs: Any,
) -> str:
    """Benchmark table reporting mean with half-width of the bootstrap CI."""
    model_list = models if models is not None else list(dict.fromkeys(frame["model"]))
    header_cells = ["Model", "Seeds"] + [m.replace("_", r"\_") for m in metrics]
    rows = []
    for model in model_list:
        summaries = [summarize_metric(frame, metric, model, **kwargs) for metric in metrics]
        n_units = max((s.n_units for s in summaries if s.resampling_unit == "train_seed"), default=0)
        seed_cell = str(n_units) if n_units else "--"
        label = (labels or {}).get(model, model).replace("_", r"\_")
        rows.append(" & ".join([label, seed_cell] + [s.format(precision) for s in summaries]) + r" \\")
    return "\n".join(
        [
            r"\begin{tabular}{ll" + "c" * len(metrics) + "}",
            r"\toprule",
            " & ".join(header_cells) + r" \\",
            r"\midrule",
            *rows,
            r"\bottomrule",
            r"\end{tabular}",
        ]
    )
