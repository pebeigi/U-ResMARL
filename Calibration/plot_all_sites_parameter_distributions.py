"""Overlay KDE of *competitive* scored utility parameters across sites.

Default: best 10% of closed-loop scored trials by objective, with working-prior
and median markers. Pass --top-fraction / --no-median / --name for variants
without overwriting the paper default.

    python Calibration/plot_all_sites_parameter_distributions.py
    python Calibration/plot_all_sites_parameter_distributions.py --top-fraction 0.2 --no-median --name all_sites_parameter_distributions_top20
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde

ROOT = Path(__file__).resolve().parents[1]

PARAM_KEYS = (
    "S_theta",
    "S_v",
    "xi_i",
    "S_d",
    "gamma",
    "w_x",
    "w_y",
    "w_c",
    "w_ell",
    "beta",
    "sigma_long",
    "sigma_lat",
)
PARAM_LABELS = {
    "S_theta": r"$S_\theta$",
    "S_v": r"$S_v$",
    "xi_i": r"$\xi_i$",
    "S_d": r"$S_d$",
    "gamma": r"$\gamma$",
    "w_x": r"$w_x$",
    "w_y": r"$w_y$",
    "w_c": r"$w_c$",
    "w_ell": r"$w_\ell$",
    "beta": r"$\beta$",
    "sigma_long": r"$\sigma_{\parallel}$",
    "sigma_lat": r"$\sigma_{\perp}$",
}
_P99 = ROOT / "Calibration" / "runs" / "recalibration_20260922_p99"
SITES = (
    {
        "label": "Highway",
        "path": _P99 / "diagnostics_highway" / "top_trials.csv",
        "calibration": _P99 / "utility_calibration_highway.json",
        "color": "#0072B2",
    },
    {
        "label": "Jounieh",
        "path": _P99 / "diagnostics_jounieh" / "top_trials.csv",
        "calibration": _P99 / "utility_calibration_jounieh.json",
        "color": "#009E73",
    },
    {
        "label": "TGSIM",
        "path": _P99 / "diagnostics_tgsim" / "top_trials.csv",
        "calibration": _P99 / "utility_calibration_tgsim.json",
        "color": "#D55E00",
    },
)
OUTPUTS = (
    ROOT / "Calibration" / "paper_summary",
    ROOT / "Paper Draft" / "ICLR" / "Appendix" / "calibration",
    ROOT / "Paper Draft" / "ICLR" / "Appendix",
    ROOT / "Paper Draft" / "Appendix" / "calibration",
    ROOT / "Paper Draft" / "ICLR" / "Pics",
    ROOT / "Paper Draft" / "Pics",
)


def _values(frame: pd.DataFrame, key: str) -> np.ndarray:
    vals = pd.to_numeric(frame[key], errors="coerce").to_numpy(dtype=float)
    return vals[np.isfinite(vals)]


def _best_objective_cloud(
    frame: pd.DataFrame, *, top_fraction: float, min_trials: int
) -> pd.DataFrame:
    """Keep the best objective fraction so clouds reflect site-specific fits."""
    if "objective" not in frame.columns:
        raise ValueError("top_trials.csv must include an 'objective' column")
    n = max(min_trials, int(round(len(frame) * top_fraction)))
    n = min(n, len(frame))
    return frame.nsmallest(n, "objective").reset_index(drop=True)


def _working_params(path: Path) -> dict[str, float]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    params = payload.get("working_params") or payload.get("best_params") or {}
    return {k: float(params[k]) for k in PARAM_KEYS if k in params}


def _kde_curve(vals: np.ndarray, lo: float, hi: float) -> tuple[np.ndarray, np.ndarray] | None:
    if len(vals) < 3:
        return None
    span = max(hi - lo, 1e-9)
    xs = np.linspace(lo - 0.02 * span, hi + 0.02 * span, 320)
    spread = float(np.std(vals))
    if spread < 1e-12:
        dens = np.zeros_like(xs)
        dens[np.argmin(np.abs(xs - float(vals[0])))] = 1.0
        return xs, dens
    dens = gaussian_kde(vals)(xs)
    dens = np.clip(dens, 0.0, None)
    return xs, dens


def plot_all_sites(
    *,
    top_fraction: float = 0.10,
    min_trials: int = 12,
    show_median: bool = True,
    out_name: str = "all_sites_parameter_distributions",
) -> Path:
    frames = []
    for site in SITES:
        raw = pd.read_csv(site["path"])
        if len(raw) == 0:
            raise ValueError(f"empty scored-trial table: {site['path']}")
        cloud = _best_objective_cloud(
            raw, top_fraction=top_fraction, min_trials=min_trials
        )
        working = _working_params(site["calibration"])
        frames.append((site, cloud, working))
        print(
            f"{site['label']}: cloud n={len(cloud)}/{len(raw)} "
            f"(best {100 * top_fraction:.0f}% by objective)",
            flush=True,
        )

    plt.rcParams.update(
        {
            "font.size": 8.5,
            "axes.titlesize": 10,
            "axes.labelsize": 8,
            "legend.fontsize": 8.5,
            "xtick.labelsize": 7.5,
            "ytick.labelsize": 7.5,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    fig, axes = plt.subplots(3, 4, figsize=(12.4, 7.6), constrained_layout=True)
    for ax, key in zip(axes.ravel(), PARAM_KEYS):
        columns = [_values(cloud, key) for _, cloud, _ in frames]
        work_vals = [working[key] for _, _, working in frames if key in working]
        lo = min(float(np.min(col)) for col in columns)
        hi = max(float(np.max(col)) for col in columns)
        if work_vals:
            lo = min(lo, min(work_vals))
            hi = max(hi, max(work_vals))
        if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
            lo, hi = 0.0, 1.0
        span = hi - lo
        for (site, _), vals, working in zip(
            [(s, c) for s, c, _ in frames],
            columns,
            [w for _, _, w in frames],
        ):
            color = site["color"]
            curve = _kde_curve(vals, lo, hi)
            if curve is not None:
                xs, dens = curve
                ax.plot(xs, dens, color=color, lw=2.0, zorder=3)
                ax.fill_between(xs, dens, color=color, alpha=0.10, linewidth=0, zorder=2)
            if show_median:
                median = float(np.median(vals))
                ax.axvline(median, color=color, lw=1.15, ls="--", alpha=0.85, zorder=4)
            if key in working:
                ax.axvline(working[key], color=color, lw=2.0, ls="-", alpha=0.95, zorder=5)
        ax.set_xlim(lo - 0.05 * span, hi + 0.05 * span)
        ax.set_ylim(bottom=0.0)
        ax.set_title(PARAM_LABELS[key], pad=3)
        ax.grid(True, axis="y", alpha=0.28, lw=0.6)
        ax.tick_params(length=2.5)
        if ax.get_subplotspec().is_first_col():
            ax.set_ylabel("density")

    handles = [
        Line2D([0], [0], color=site["color"], lw=2.0, label=f"{site['label']} ($n={len(cloud)}$)")
        for site, cloud, _ in frames
    ]
    handles.append(Line2D([0], [0], color="#334155", lw=2.0, ls="-", label="working"))
    if show_median:
        handles.append(Line2D([0], [0], color="#334155", lw=1.3, ls="--", label="median"))
    fig.legend(
        handles=handles,
        loc="upper center",
        ncol=5 if show_median else 4,
        frameon=False,
        bbox_to_anchor=(0.5, 1.045),
        handlelength=2.2,
        columnspacing=1.4,
    )
    primary = OUTPUTS[0]
    primary.mkdir(parents=True, exist_ok=True)
    png = primary / f"{out_name}.png"
    pdf = primary / f"{out_name}.pdf"
    fig.savefig(png, dpi=300, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    payload = png.read_bytes()
    pdf_bytes = pdf.read_bytes()
    for folder in OUTPUTS[1:]:
        folder.mkdir(parents=True, exist_ok=True)
        try:
            (folder / png.name).write_bytes(payload)
            (folder / pdf.name).write_bytes(pdf_bytes)
        except OSError as exc:
            print(f"warn: could not write {folder / png.name}: {exc}", flush=True)
    return png


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--top-fraction",
        type=float,
        default=0.10,
        help="Keep the best this fraction of scored trials by objective",
    )
    parser.add_argument("--min-trials", type=int, default=12)
    parser.add_argument(
        "--no-median",
        action="store_true",
        help="Omit dashed median markers (working prior lines stay)",
    )
    parser.add_argument(
        "--name",
        default="all_sites_parameter_distributions",
        help="Output basename without extension",
    )
    args = parser.parse_args()
    path = plot_all_sites(
        top_fraction=args.top_fraction,
        min_trials=args.min_trials,
        show_median=not args.no_median,
        out_name=args.name,
    )
    print(path)


if __name__ == "__main__":
    main()
