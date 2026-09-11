"""Run checklist A2–A6: multi-seed train → ablation/stress → sparse benchmark → figures/stats.

Usage (from repo root):
    python -u -m Baselines._run_A_pipeline
    python -u -m Baselines._run_A_pipeline --skip-train   # eval only once checkpoints exist
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import Baselines._paths  # noqa: F401
from Baselines._paths import REPO_ROOT

SEEDS = [0, 1, 2]
LOG_DIR = Path("RL/logs/v2/paper_rerun")
PIPELINE_LOG = LOG_DIR / "A_pipeline.log"
STATS_OUT = Path("Baselines/results/v2/paper_rerun/A6_stats_summary.md")

SPARSE_MODELS = [
    "orca",
    "social_force",
    "dwa",
    "mppi",
    "frenet",
    "utility_pt",
    "direct_discrete_rl",
    "mappo",
    "residual_marl",
]


def log(msg: str) -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    line = f"{datetime.now(timezone.utc).isoformat()}  {msg}"
    print(line, flush=True)
    with PIPELINE_LOG.open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def run(cmd: list[str], step: str) -> None:
    log(f"START {step}: {' '.join(cmd)}")
    t0 = time.time()
    proc = subprocess.run(cmd, cwd=REPO_ROOT)
    elapsed_h = (time.time() - t0) / 3600.0
    if proc.returncode != 0:
        log(f"FAIL  {step} (exit {proc.returncode}) after {elapsed_h:.2f} h")
        raise SystemExit(proc.returncode)
    log(f"OK    {step} in {elapsed_h:.2f} h")


def write_stats_summary() -> None:
    import pandas as pd

    paths = {
        "sparse_benchmark": Path("Baselines/results/v2/benchmark_summary.csv"),
        "sparse_paired": Path("Baselines/results/v2/benchmark_comparisons.csv"),
        "ablation": Path("Baselines/results/v2/paper_ablation/ablation/ablation_summary.csv"),
        "ablation_paired": Path("Baselines/results/v2/paper_ablation/ablation/ablation_paired.csv"),
        "ablation_no_lookahead": Path(
            "Baselines/results/v2/paper_ablation/ablation_no_lookahead/ablation_no_lookahead_summary.csv"
        ),
        "stress": Path("Baselines/results/v2/paper_ablation/stress/stress_summary.csv"),
        "stress_paired": Path("Baselines/results/v2/paper_ablation/stress/stress_paired.csv"),
    }

    lines = [
        "# A6 stats summary",
        "",
        f"Generated: {datetime.now(timezone.utc).isoformat()}",
        f"Train seeds: {SEEDS}",
        "",
        "## Honesty notes",
        "",
        "- Report mean ± bootstrap 95% CI over training seeds (and scenario units as coded).",
        "- Do not claim significance without paired bootstrap / Wilcoxon rows in `*_paired.csv`.",
        "- Residual MARL is freeway-only; urban sites are calibration evidence only.",
        "- Matched discrete RL (`direct_discrete_rl`) uses the same action grid + OBB filter.",
        "",
    ]

    for name, path in paths.items():
        lines.append(f"## {name} (`{path.as_posix()}`)")
        lines.append("")
        if not path.exists():
            lines.append("_missing_")
            lines.append("")
            continue
        df = pd.read_csv(path)
        lines.append("```")
        lines.append(df.head(40).to_string(index=False))
        if len(df) > 40:
            lines.append(f"... ({len(df)} rows total)")
        lines.append("```")
        lines.append("")

    STATS_OUT.parent.mkdir(parents=True, exist_ok=True)
    STATS_OUT.write_text("\n".join(lines), encoding="utf-8")
    log(f"Wrote {STATS_OUT}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Checklist A2–A6 pipeline")
    parser.add_argument("--skip-train", action="store_true")
    parser.add_argument("--skip-eval", action="store_true")
    parser.add_argument("--skip-benchmark", action="store_true")
    parser.add_argument("--skip-figures", action="store_true")
    parser.add_argument("--jobs", type=int, default=2)
    parser.add_argument("--scenarios", type=int, default=30)
    args = parser.parse_args()

    seed_args = [str(s) for s in SEEDS]
    py = sys.executable

    if not args.skip_train:
        run(
            [
                py,
                "-u",
                "-m",
                "Baselines.paper_rerun",
                "train",
                "--models",
                "residual_marl",
                "residual_param",
                "residual_nominal",
                "direct_discrete_rl",
                "mappo",
                "--seeds",
                *seed_args,
                "--jobs",
                str(args.jobs),
                "--log-dir",
                str(LOG_DIR),
            ],
            "A2+A5 multi-seed train",
        )

    if not args.skip_eval:
        run(
            [
                py,
                "-u",
                "-m",
                "Baselines.paper_rerun",
                "eval",
                "--seeds",
                *seed_args,
                "--scenarios",
                str(args.scenarios),
            ],
            "A3 ablation+stress eval",
        )

    if not args.skip_benchmark:
        run(
            [
                py,
                "-u",
                "-m",
                "Baselines.benchmark",
                "--models",
                *SPARSE_MODELS,
                "--scenarios",
                str(args.scenarios),
                "--train-seeds",
                *seed_args,
                "--output-dir",
                "Baselines/results/v2",
            ],
            "A4 sparse benchmark",
        )

    if not args.skip_figures:
        run(
            [py, "-u", "-m", "Baselines.paper_figures", "--all"],
            "A6 paper figures",
        )

    write_stats_summary()
    log("ALL A-CHECKLIST STEPS COMPLETE")


if __name__ == "__main__":
    main()
