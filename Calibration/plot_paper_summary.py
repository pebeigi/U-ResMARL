#!/usr/bin/env python
"""Build paper-ready calibration tables and distribution plots from saved outputs."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde

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

PARAM_BOUNDS = {
    "S_theta": (0.05, 10.0),
    "S_v": (0.05, 10.0),
    "xi_i": (1.1, 10.0),
    "S_d": (0.05, 10.0),
    "gamma": (0.1, 10.0),
    "w_x": (0.1, 10.0),
    "w_y": (0.1, 10.0),
    "w_c": (0.01, 1000.0),
    "w_ell": (0.1, 1000.0),
    "beta": (0.01, 10.0),
    "sigma_long": (0.5, 5.0),
    "sigma_lat": (0.3, 2.5),
    "w_ell": (0.1, 1000.0),
    "beta": (0.01, 10.0),
    "sigma_long": (0.5, 5.0),
    "sigma_lat": (0.3, 2.5),
    "w_ell": (0.1, 1000.0),
    "beta": (0.01, 10.0),
    "sigma_long": (0.5, 5.0),
    "sigma_lat": (0.3, 2.5),
}


def latex_escape(text: str) -> str:
    return (
        text.replace("\\", "\\textbackslash{}")
        .replace("_", "\\_")
        .replace("%", "\\%")
        .replace("&", "\\&")
    )


def write_latex_table(df: pd.DataFrame, path: Path, caption: str, label: str) -> None:
    cols = list(df.columns)
    align = "l" + "r" * (len(cols) - 1)
    lines = [
        "\\begin{table}[t]",
        "\\centering",
        f"\\caption{{{caption}}}",
        f"\\label{{{label}}}",
        f"\\begin{{tabular}}{{{align}}}",
        "\\toprule",
        " & ".join(latex_escape(str(c)) for c in cols) + " \\\\",
        "\\midrule",
    ]
    for _, row in df.iterrows():
        cells = []
        for c in cols:
            val = row[c]
            if isinstance(val, (float, np.floating)):
                cells.append(f"{float(val):.3g}")
            else:
                cells.append(latex_escape(str(val)))
        lines.append(" & ".join(cells) + " \\\\")
    lines.extend(["\\bottomrule", "\\end{tabular}", "\\end{table}", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def percentile_interval(vals: np.ndarray, level: float = 0.95) -> tuple[float, float, float]:
    """Return (low, median, high) for a central percentile interval."""
    alpha = (1.0 - level) / 2.0
    lo, med, hi = np.quantile(vals, [alpha, 0.5, 1.0 - alpha])
    return float(lo), float(med), float(hi)


def overlay_kde(ax, vals: np.ndarray, color: str = "#E45756", x_range: tuple[float, float] | None = None) -> None:
    """Overlay a Gaussian KDE on a density-normalized histogram."""
    if len(vals) < 3 or float(np.std(vals)) < 1e-12:
        return
    if x_range is None:
        lo, hi = float(np.min(vals)), float(np.max(vals))
    else:
        lo, hi = float(x_range[0]), float(x_range[1])
    pad = 0.02 * (hi - lo + 1e-9)
    xs = np.linspace(lo - pad, hi + pad, 256)
    # KDE fitted on the provided values (already clipped to the 95% window by caller).
    dens = gaussian_kde(vals)(xs)
    ax.plot(xs, dens, color=color, lw=2.0, label="KDE")
    ax.fill_between(xs, dens, color=color, alpha=0.12)


def clip_to_percentile_interval(vals: np.ndarray, level: float = 0.95) -> tuple[np.ndarray, float, float, float]:
    """Return values inside the central percentile interval, plus (lo, median, hi)."""
    vals = np.asarray(vals, dtype=float)
    vals = vals[np.isfinite(vals)]
    if len(vals) == 0:
        return vals, float("nan"), float("nan"), float("nan")
    if len(vals) == 1:
        v = float(vals[0])
        return vals.copy(), v, v, v
    lo, med, hi = percentile_interval(vals, level=level)
    # Guard against a degenerate interval (all equal).
    if hi <= lo:
        return vals.copy(), lo, med, hi
    clipped = vals[(vals >= lo) & (vals <= hi)]
    if len(clipped) < 3:
        clipped = vals.copy()
    return clipped, lo, med, hi


def plot_parameter_distributions(
    per_id: pd.DataFrame,
    out_dir: Path,
    *,
    title: str = "Per-vehicle calibrated utility-parameter distributions with KDE and 95% intervals",
    outfile: str = "fig_parameter_distributions.png",
    n_label: str | None = None,
    mark_values: dict[str, dict[str, float]] | None = None,
) -> None:
    n = len(per_id)
    n_txt = n_label or f"n={n}"
    fig, axes = plt.subplots(2, 5, figsize=(14.5, 6.2), constrained_layout=True)
    for ax, key in zip(axes.ravel(), PARAM_KEYS):
        vals = per_id[key].dropna().to_numpy(dtype=float)
        if len(vals) == 0:
            ax.set_title(PARAM_LABELS[key])
            continue
        clipped, p025, p50, p975 = clip_to_percentile_interval(vals, level=0.95)
        span = max(p975 - p025, 1e-9)
        n_bins = min(18, max(6, int(np.sqrt(len(clipped)))))
        ax.hist(
            clipped,
            bins=n_bins,
            range=(p025, p975),
            density=True,
            color="#4C78A8",
            edgecolor="white",
            linewidth=0.6,
            alpha=0.75,
            label="histogram (95%)",
        )
        overlay_kde(ax, clipped, x_range=(p025, p975))
        ax.axvline(p025, color="#54A24B", lw=1.2, ls=":", label="p2.5 / p97.5")
        ax.axvline(p975, color="#54A24B", lw=1.2, ls=":")
        ax.axvline(p50, color="#E45756", lw=1.6, label="median")
        if mark_values:
            if "best" in mark_values and key in mark_values["best"]:
                bv = mark_values["best"][key]
                if p025 - 0.02 * span <= bv <= p975 + 0.02 * span:
                    ax.axvline(bv, color="#4C78A8", lw=1.5, ls="--", label="best")
            if "robust" in mark_values and key in mark_values["robust"]:
                rv = mark_values["robust"][key]
                if p025 - 0.02 * span <= rv <= p975 + 0.02 * span:
                    ax.axvline(rv, color="#F58518", lw=1.8, label="robust")
        ax.set_xlim(p025 - 0.02 * span, p975 + 0.02 * span)
        ax.set_title(PARAM_LABELS[key])
        ax.set_xlabel("")
        ax.set_ylabel("density")
        ax.grid(True, axis="y", alpha=0.25)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    seen = set()
    uniq = [(h, lab) for h, lab in zip(handles, labels) if not (lab in seen or seen.add(lab))]
    fig.legend(
        [h for h, _ in uniq],
        [lab for _, lab in uniq],
        loc="upper center",
        ncol=min(6, len(uniq)),
        frameon=False,
        bbox_to_anchor=(0.5, 1.05),
    )
    fig.suptitle(f"{title} ({n_txt})", y=1.10)
    fig.savefig(out_dir / outfile, dpi=220, bbox_inches="tight")
    plt.close(fig)


def robust_cloud_from_trials(tdf: pd.DataFrame, result: dict, k: int = 10) -> pd.DataFrame:
    """Trials used for robust_params: flagged near-optimal, else top-k by objective."""
    if "near_optimal" in tdf.columns and int(tdf["near_optimal"].sum()) >= 3:
        return tdf[tdf["near_optimal"]].copy()
    return tdf.nsmallest(min(k, len(tdf)), "objective").copy()


def plot_parameter_ranges_forest(result: dict, per_id: pd.DataFrame, out_dir: Path) -> None:
    ranges = result["recommended_ranges_from_top_trials"]
    best = result["best_params"]
    small_keys = [k for k in PARAM_KEYS if PARAM_BOUNDS[k][1] <= 10.0]
    large_keys = [k for k in PARAM_KEYS if PARAM_BOUNDS[k][1] > 10.0]

    fig, axes = plt.subplots(2, 2, figsize=(12.5, 7.2), constrained_layout=True)

    def _draw_global(ax, keys):
        # Global JSON stores p05/p50/p95 (central 90%). Highlight p95 as the
        # upper recommended bound used for GSA.
        y = np.arange(len(keys))
        for i, key in enumerate(keys):
            lo, med, hi = ranges[key]["p05"], ranges[key]["p50"], ranges[key]["p95"]
            ax.hlines(i, lo, hi, color="#4C78A8", lw=2.2)
            ax.plot(med, i, "o", color="#4C78A8", ms=6)
            ax.plot(hi, i, "^", color="#54A24B", ms=6)
            ax.plot(best[key], i, "D", color="#E45756", ms=5)
        ax.set_yticks(y)
        ax.set_yticklabels([PARAM_LABELS[k] for k in keys])
        ax.invert_yaxis()
        ax.grid(True, axis="x", alpha=0.25)

    def _draw_local(ax, keys):
        y = np.arange(len(keys))
        for i, key in enumerate(keys):
            vals = per_id[key].dropna().to_numpy(dtype=float)
            lo95, med, hi95 = percentile_interval(vals, level=0.95)
            ax.hlines(i, lo95, hi95, color="#4C78A8", lw=3.0)
            ax.plot(med, i, "o", color="#E45756", ms=5)
            ax.plot(hi95, i, "^", color="#54A24B", ms=6)
        ax.set_yticks(y)
        ax.set_yticklabels([PARAM_LABELS[k] for k in keys])
        ax.invert_yaxis()
        ax.grid(True, axis="x", alpha=0.25)

    _draw_global(axes[0, 0], small_keys)
    axes[0, 0].set_title("Global top trials (p05–p50–p95)")
    axes[0, 0].set_xlabel("parameter value")
    axes[0, 0].plot([], [], "o", color="#4C78A8", label="p50")
    axes[0, 0].plot([], [], "^", color="#54A24B", label="p95")
    axes[0, 0].plot([], [], "D", color="#E45756", label="best")
    axes[0, 0].legend(loc="lower right", frameon=False)

    _draw_local(axes[0, 1], small_keys)
    axes[0, 1].set_title("Per-vehicle local fits (95% interval)")
    axes[0, 1].set_xlabel("parameter value")
    axes[0, 1].plot([], [], "o", color="#E45756", label="median")
    axes[0, 1].plot([], [], "^", color="#54A24B", label="p97.5")
    axes[0, 1].legend(loc="lower right", frameon=False)

    _draw_global(axes[1, 0], large_keys)
    axes[1, 0].set_title(r"Global top trials ($w_c$, $w_\ell$)")
    axes[1, 0].set_xlabel("parameter value")

    _draw_local(axes[1, 1], large_keys)
    axes[1, 1].set_title(r"Per-vehicle 95% intervals ($w_c$, $w_\ell$)")
    axes[1, 1].set_xlabel("parameter value")

    fig.suptitle("Calibrated utility-parameter uncertainty with 95% percentiles", y=1.02)
    fig.savefig(out_dir / "fig_parameter_ranges_forest.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_fit_quality(qdf: pd.DataFrame, per_id: pd.DataFrame, out_dir: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 8.0), constrained_layout=True)

    ax = axes[0, 0]
    ranks = np.sort(qdf["target_rank"].to_numpy(dtype=float))
    cdf = np.arange(1, len(ranks) + 1) / len(ranks)
    ax.plot(ranks, cdf, color="#4C78A8", lw=2)
    ax.axhline(0.5, color="k", ls=":", lw=1)
    ax.set_xlabel("target candidate rank (1 = best)")
    ax.set_ylabel("empirical CDF")
    ax.set_title("One-step ranking of observed-like action")
    ax.grid(True, alpha=0.25)

    ax = axes[0, 1]
    ax.hist(qdf["target_rank"], bins=np.arange(1, qdf["target_rank"].max() + 2) - 0.5, color="#4C78A8", edgecolor="white")
    ax.set_xlabel("rank")
    ax.set_ylabel("count")
    ax.set_title("Rank histogram (global samples)")
    ax.grid(True, axis="y", alpha=0.25)

    ax = axes[1, 0]
    margins = qdf["target_margin"].dropna().to_numpy(dtype=float)
    # Full central 95% still includes a long left tail (p2.5 ~ -4), which
    # collapses the bulk near zero into one bar. Zoom to the dense core
    # (p10–p99) so binning resolves the main mass; report excluded tails.
    lo_core, hi_core = np.quantile(margins, [0.10, 0.99])
    if hi_core <= lo_core:
        lo_core, hi_core = float(np.min(margins)), float(np.max(margins))
    core = margins[(margins >= lo_core) & (margins <= hi_core)]
    med = float(np.median(margins))
    n_bins = min(50, max(20, int(np.sqrt(len(core)) * 2)))
    ax.hist(
        core,
        bins=n_bins,
        range=(float(lo_core), float(hi_core)),
        color="#72B7B2",
        edgecolor="white",
        linewidth=0.5,
        label="core window",
    )
    ax.axvline(0.0, color="k", ls="--", lw=1, label="zero margin")
    ax.axvline(med, color="#E45756", lw=1.5, label="median")
    ax.set_xlim(float(lo_core), float(hi_core))
    n_out = int(len(margins) - len(core))
    ax.set_xlabel(r"$U_{\mathrm{target}}-\max U_{\mathrm{other}}$ (p10–p99 window)")
    ax.set_ylabel("count")
    ax.set_title(f"Utility margin of observed-like candidate\n({n_out} tail points outside window)")
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(frameon=False, fontsize=8)

    ax = axes[1, 1]
    losses = per_id["local_closed_loop_loss"].dropna().to_numpy(dtype=float)
    loss_clip, llo, lmed, lhi = clip_to_percentile_interval(losses, level=0.95)
    ax.hist(
        loss_clip,
        bins=min(25, max(8, int(np.sqrt(len(loss_clip))))),
        range=(llo, lhi),
        color="#F58518",
        edgecolor="white",
    )
    ax.axvline(lmed, color="#E45756", lw=1.4)
    ax.set_xlim(llo, lhi)
    ax.set_xlabel("local closed-loop loss (95% window)")
    ax.set_ylabel("vehicles")
    ax.set_title("Per-vehicle closed-loop tracking loss")
    ax.grid(True, axis="y", alpha=0.25)

    fig.suptitle("Calibration fit diagnostics", y=1.02)
    fig.savefig(out_dir / "fig_fit_quality.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def build_tables(result: dict, per_id: pd.DataFrame, qdf: pd.DataFrame, out_dir: Path) -> dict:
    ranges = result.get("near_optimal_ranges") or result["recommended_ranges_from_top_trials"]
    best = result["best_params"]
    robust = result.get("robust_params") or result.get("working_params") or best

    param_rows = []
    for key in PARAM_KEYS:
        vals = per_id[key].dropna().to_numpy(dtype=float)
        lo95, med, hi95 = percentile_interval(vals, level=0.95)
        lo_b, hi_b = PARAM_BOUNDS[key]
        param_rows.append(
            {
                "parameter": key,
                "search_low": lo_b,
                "search_high": hi_b,
                "best": best[key],
                "robust": robust[key],
                "top_p05": ranges[key].get("p05", ranges[key].get("p025")),
                "top_p50": ranges[key]["p50"],
                "top_p95": ranges[key].get("p95", ranges[key].get("p975")),
                "top_p025": ranges[key].get("p025", ranges[key].get("p05")),
                "top_p975": ranges[key].get("p975", ranges[key].get("p95")),
                "per_id_mean": float(np.mean(vals)),
                "per_id_std": float(np.std(vals, ddof=1)),
                "per_id_p025": lo95,
                "per_id_p50": med,
                "per_id_p975": hi95,
                "per_id_p05": float(np.quantile(vals, 0.05)),
                "per_id_p95": float(np.quantile(vals, 0.95)),
            }
        )
    param_df = pd.DataFrame(param_rows)
    param_df.to_csv(out_dir / "table_parameter_summary.csv", index=False)
    write_latex_table(
        param_df[
            [
                "parameter",
                "best",
                "robust",
                "top_p50",
                "top_p95",
                "per_id_p50",
                "per_id_p025",
                "per_id_p975",
                "per_id_p95",
            ]
        ].round(3),
        out_dir / "table_parameter_summary.tex",
        caption=(
            "Calibrated utility parameters with 95\\% percentiles: single best, robust "
            "(median of near-optimal trials), top-trial p50--p95, and per-vehicle local-fit "
            "median with central 95\\% interval (p2.5--p97.5) and upper p95."
        ),
        label="tab:utility_calibration_params",
    )

    setup_df = pd.DataFrame(
        [
            {"item": "Trajectory rows (valid)", "value": 190767},
            {"item": "One-step choice samples", "value": int(result["n_samples"])},
            {"item": "Random-search trials", "value": int(result["n_trials"])},
            {"item": "Closed-loop candidates retained", "value": int(result["closed_loop_candidates"])},
            {"item": "Closed-loop windows", "value": int(result["n_rollout_windows"])},
            {"item": "Closed-loop horizon (steps)", "value": int(result["closed_loop_horizon_steps"])},
            {"item": "Softmax temperature", "value": float(result["temperature"])},
            {"item": "One-step weight", "value": float(result["one_step_weight"])},
            {"item": "Closed-loop weight", "value": float(result["closed_loop_weight"])},
            {"item": "Tracking-rank weight", "value": float(result["tracking_weight"])},
            {"item": "Per-vehicle diagnostic plots", "value": int(len(per_id))},
            {"item": "Utility frame", "value": str(result["utility_frame"])},
        ]
    )
    setup_df.to_csv(out_dir / "table_calibration_setup.csv", index=False)
    write_latex_table(
        setup_df,
        out_dir / "table_calibration_setup.tex",
        caption="Utility calibration experimental setup.",
        label="tab:utility_calibration_setup",
    )

    ranks = qdf["target_rank"].to_numpy(dtype=float)
    fit_df = pd.DataFrame(
        [
            {"metric": "Best objective", "value": float(result["best_objective"])},
            {"metric": "Best closed-loop loss", "value": float(result["best_closed_loop_loss"])},
            {"metric": "Best one-step NLL", "value": float(result["best_nll"])},
            {"metric": "Best mean target rank", "value": float(result["best_tracking_rank"])},
            {"metric": "Mean target rank (held samples)", "value": float(np.mean(ranks))},
            {"metric": "Median target rank", "value": float(np.median(ranks))},
            {"metric": "Top-1 hit rate", "value": float(np.mean(ranks == 1))},
            {"metric": "Top-5 hit rate", "value": float(np.mean(ranks <= 5))},
            {"metric": "Top-10 hit rate", "value": float(np.mean(ranks <= 10))},
            {"metric": "Mean target probability", "value": float(qdf["target_probability"].mean())},
            {"metric": "Positive utility-margin fraction", "value": float(np.mean(qdf["target_margin"] > 0))},
            {"metric": "Per-ID mean closed-loop loss", "value": float(per_id["local_closed_loop_loss"].mean())},
            {"metric": "Per-ID median closed-loop loss", "value": float(per_id["local_closed_loop_loss"].median())},
            {"metric": "Per-ID mean local NLL", "value": float(per_id["local_nll"].mean())},
            {"metric": "Per-ID median tracking rank", "value": float(per_id["local_tracking_rank"].median())},
        ]
    )
    fit_df.to_csv(out_dir / "table_fit_quality.csv", index=False)
    write_latex_table(
        fit_df.round(4),
        out_dir / "table_fit_quality.tex",
        caption="Calibration fit-quality metrics for the global utility prior.",
        label="tab:utility_calibration_fit",
    )

    # Main paper table: best + robust + 95% percentiles
    compact = pd.DataFrame(
        [
            {
                "parameter": key,
                "best": best[key],
                "robust": robust[key],
                "top_p50": ranges[key]["p50"],
                "top_p95": ranges[key].get("p95", ranges[key].get("p975")),
                "per_id_p50": float(np.median(per_id[key].dropna())),
                "per_id_p025": percentile_interval(per_id[key].dropna().to_numpy(dtype=float), 0.95)[0],
                "per_id_p975": percentile_interval(per_id[key].dropna().to_numpy(dtype=float), 0.95)[2],
                "per_id_p95": float(np.quantile(per_id[key].dropna().to_numpy(dtype=float), 0.95)),
            }
            for key in PARAM_KEYS
        ]
    ).round(3)
    compact.to_csv(out_dir / "table_parameter_95pct.csv", index=False)
    compact.to_csv(out_dir / "table_recommended_ranges.csv", index=False)
    write_latex_table(
        compact,
        out_dir / "table_parameter_95pct.tex",
        caption=(
            "Utility-parameter summary with robust near-optimal median and 95\\% percentiles: "
            "single best, robust working point, top-trial p50/p95, plus per-vehicle median, "
            "central 95\\% interval (p2.5--p97.5), and p95."
        ),
        label="tab:utility_parameter_95pct",
    )
    write_latex_table(
        compact,
        out_dir / "table_recommended_ranges.tex",
        caption="Recommended utility-parameter values and 95\\% percentile ranges.",
        label="tab:utility_recommended_ranges",
    )

    return {
        "param_df": param_df,
        "setup_df": setup_df,
        "fit_df": fit_df,
        "compact": compact,
    }


def write_paper_notes(result: dict, tables: dict, out_dir: Path) -> None:
    fit = {row["metric"]: row["value"] for _, row in tables["fit_df"].iterrows()}
    lines = [
        "Utility Calibration Results — Paper Notes",
        "=" * 48,
        "",
        "Setup",
        "-----",
        (
            f"Utility parameters were calibrated from the Lebanon trajectory set using "
            f"{int(result['n_samples'])} one-step choice samples and "
            f"{int(result['n_rollout_windows'])} short closed-loop windows "
            f"({int(result['closed_loop_horizon_steps'])} steps each). "
            f"A random search over {int(result['n_trials'])} parameter vectors screened "
            f"candidates by one-step NLL and target rank; the best "
            f"{int(result['closed_loop_candidates'])} candidates were then scored with a "
            "closed-loop tracking objective in the corridor frame."
        ),
        "",
        "Fit quality",
        "-----------",
        (
            f"The selected parameter set achieved objective {fit['Best objective']:.3f}, "
            f"closed-loop loss {fit['Best closed-loop loss']:.3f}, and mean target rank "
            f"{fit['Best mean target rank']:.2f}. On the 2000 held choice samples, the "
            f"observed-like candidate had median rank {fit['Median target rank']:.1f} "
            f"(mean {fit['Mean target rank (held samples)']:.2f}), with top-5 / top-10 hit "
            f"rates of {100*fit['Top-5 hit rate']:.1f}% / {100*fit['Top-10 hit rate']:.1f}%."
        ),
        "",
        "Parameter interpretation",
        "------------------------",
        (
            "Directional and speed weights (S_theta, S_v) and proximity weights (S_d, w_x, w_y) "
            "concentrate in moderate-to-high values, while lane-keeping (w_ell, beta) and "
            "collision (w_c) exhibit broader per-vehicle dispersion. Table summaries report "
            "95% percentiles (per-vehicle central 95% interval p2.5–p97.5 and upper p95; "
            "global top-trial p95) for use as GSA bounds."
        ),
        "",
        "Suggested figure/table use",
        "--------------------------",
        "- Table: table_parameter_95pct.(csv|tex)     -> main 95% percentile parameter table",
        "- Table: table_fit_quality.(csv|tex)         -> calibration quality metrics",
        "- Table: table_calibration_setup.(csv|tex)   -> methods appendix",
        "- Fig: fig_parameter_ranges_forest.png       -> uncertainty with 95% intervals",
        "- Fig: fig_parameter_distributions.png       -> hist + KDE + 95% band",
        "- Fig: fig_fit_quality.png                   -> ranking / tracking diagnostics",
        "",
        "Files written under: " + str(out_dir.resolve()),
        "",
    ]
    (out_dir / "paper_notes.txt").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    import Calibration._paths  # noqa: F401
    from Calibration._paths import REPO_ROOT

    root = REPO_ROOT / "Calibration"
    out_dir = root / "paper_summary"
    out_dir.mkdir(parents=True, exist_ok=True)

    result = json.loads((root / "utility_calibration.json").read_text(encoding="utf-8"))
    per_id = pd.read_csv(root / "diagnostics" / "per_id" / "per_id_best_params.csv")
    qdf = pd.read_csv(root / "diagnostics" / "calibration_quality_samples.csv")
    trials_path = root / "diagnostics" / "top_trials.csv"
    trials = pd.read_csv(trials_path) if trials_path.exists() else None

    marks = {
        "best": result["best_params"],
        "robust": result.get("robust_params") or result.get("working_params") or result["best_params"],
    }

    # Primary paper figure: global scored-trial cloud (multi-restart calibration).
    if trials is not None and len(trials) > 0:
        plot_parameter_distributions(
            trials,
            out_dir,
            title="Global scored-trial parameter distributions with KDE and 95% intervals",
            outfile="fig_parameter_distributions.png",
            n_label=f"n={len(trials)} scored trials, {int(result.get('n_restarts', 1))} restarts",
            mark_values=marks,
        )
        cloud = robust_cloud_from_trials(trials, result, k=10)
        plot_parameter_distributions(
            cloud,
            out_dir,
            title="Near-optimal / robust-cloud parameter distributions with KDE",
            outfile="fig_parameter_distributions_near_optimal.png",
            n_label=f"n={len(cloud)} trials used for robust_params",
            mark_values=marks,
        )
        # Also keep a copy of the identifiability diagnostic if present.
        src = root / "diagnostics" / "identifiability_diagnostics.png"
        if src.exists():
            (out_dir / "fig_identifiability_diagnostics.png").write_bytes(src.read_bytes())
    else:
        plot_parameter_distributions(per_id, out_dir, mark_values=marks)

    # Secondary: per-vehicle local-fit heterogeneity.
    plot_parameter_distributions(
        per_id,
        out_dir,
        title="Per-vehicle local-fit parameter distributions with KDE and 95% intervals",
        outfile="fig_parameter_distributions_per_id.png",
        mark_values=marks,
    )
    plot_parameter_ranges_forest(result, per_id, out_dir)
    plot_fit_quality(qdf, per_id, out_dir)
    tables = build_tables(result, per_id, qdf, out_dir)
    write_paper_notes(result, tables, out_dir)

    print(f"Wrote paper summary artifacts to {out_dir.resolve()}")
    print("Figures:")
    for name in sorted(out_dir.glob("fig_*.png")):
        print(f"  {name.name}")
    print("Tables:")
    for name in sorted(out_dir.glob("table_*")):
        print(f"  {name.name}")
    idinfo = result.get("identifiability") or {}
    print(
        "Identifiability:",
        f"verdict={idinfo.get('verdict')},",
        f"near-optimal={idinfo.get('n_near_optimal')}/{idinfo.get('n_scored_trials')},",
        f"robust_obj={result.get('robust_objective')}, best_obj={result.get('best_objective')}",
    )
    print("Notes: paper_notes.txt")


if __name__ == "__main__":
    main()
