"""Overlay KDE / mean / median of scored utility parameters across sites.

    python Calibration/plot_all_sites_parameter_distributions.py
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde

ROOT = Path(__file__).resolve().parents[1]
OUT_NAME = "all_sites_parameter_distributions"

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
SITES = (
    {
        "label": "Highway",
        "path": ROOT / "Calibration" / "diagnostics" / "top_trials.csv",
        "color": "#0072B2",
    },
    {
        "label": "Jounieh",
        "path": ROOT / "Calibration" / "diagnostics_jounieh" / "top_trials.csv",
        "color": "#009E73",
    },
    {
        "label": "TGSIM",
        "path": ROOT / "Calibration" / "diagnostics_tgsim" / "top_trials.csv",
        "color": "#D55E00",
    },
)
OUTPUTS = (
    ROOT / "Calibration" / "paper_summary",
    ROOT / "Paper Draft" / "ICLR" / "Appendix" / "calibration",
    ROOT / "Paper Draft" / "Appendix" / "calibration",
    ROOT / "Paper Draft" / "ICLR" / "Pics",
    ROOT / "Paper Draft" / "Pics",
)


def _values(frame: pd.DataFrame, key: str) -> np.ndarray:
    vals = pd.to_numeric(frame[key], errors="coerce").to_numpy(dtype=float)
    return vals[np.isfinite(vals)]


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


def plot_all_sites() -> Path:
    frames = []
    for site in SITES:
        frame = pd.read_csv(site["path"])
        if len(frame) == 0:
            raise ValueError(f"empty scored-trial table: {site['path']}")
        frames.append((site, frame))

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
        columns = [_values(frame, key) for _, frame in frames]
        lo = min(float(np.min(col)) for col in columns)
        hi = max(float(np.max(col)) for col in columns)
        if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
            lo, hi = 0.0, 1.0
        span = hi - lo
        for (site, _), vals in zip(frames, columns):
            color = site["color"]
            curve = _kde_curve(vals, lo, hi)
            if curve is not None:
                xs, dens = curve
                ax.plot(xs, dens, color=color, lw=2.0, zorder=3)
                ax.fill_between(xs, dens, color=color, alpha=0.06, linewidth=0, zorder=2)
            mean = float(np.mean(vals))
            median = float(np.median(vals))
            ax.axvline(mean, color=color, lw=1.25, ls="-", alpha=0.95, zorder=4)
            ax.axvline(median, color=color, lw=1.25, ls="--", alpha=0.95, zorder=4)
        ax.set_xlim(lo - 0.03 * span, hi + 0.03 * span)
        ax.set_ylim(bottom=0.0)
        ax.set_title(PARAM_LABELS[key], pad=3)
        ax.grid(True, axis="y", alpha=0.28, lw=0.6)
        ax.tick_params(length=2.5)
        if ax.get_subplotspec().is_first_col():
            ax.set_ylabel("density")

    handles = [
        Line2D([0], [0], color=site["color"], lw=2.0, label=f"{site['label']} ($n={len(frame)}$)")
        for site, frame in frames
    ]
    handles.extend(
        [
            Line2D([0], [0], color="#334155", lw=1.4, ls="-", label="mean"),
            Line2D([0], [0], color="#334155", lw=1.4, ls="--", label="median"),
        ]
    )
    fig.legend(
        handles=handles,
        loc="upper center",
        ncol=5,
        frameon=False,
        bbox_to_anchor=(0.5, 1.045),
        handlelength=2.2,
        columnspacing=1.4,
    )
    primary = OUTPUTS[0]
    primary.mkdir(parents=True, exist_ok=True)
    png = primary / f"{OUT_NAME}.png"
    pdf = primary / f"{OUT_NAME}.pdf"
    fig.savefig(png, dpi=300, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    payload = png.read_bytes()
    pdf_bytes = pdf.read_bytes()
    for folder in OUTPUTS[1:]:
        folder.mkdir(parents=True, exist_ok=True)
        (folder / png.name).write_bytes(payload)
        (folder / pdf.name).write_bytes(pdf_bytes)
    return png


if __name__ == "__main__":
    path = plot_all_sites()
    print(path)
