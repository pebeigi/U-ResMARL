"""Summarize available 3-site RL/baseline results."""
from __future__ import annotations

from pathlib import Path
import pandas as pd

KEEP = [
    "model",
    "mean_collision_events",
    "mean_arrival_rate",
    "mean_closed_loop_score",
    "mean_min_ttc_s",
    "mean_realism_score",
]


def sparse(label: str, path: Path) -> None:
    print(f"\n=== {label} ===")
    if not path.exists():
        print(f"MISSING: {path}")
        return
    df = pd.read_csv(path)
    cols = [c for c in KEEP if c in df.columns]
    out = df[cols].copy()
    for c in cols:
        if c != "model":
            out[c] = out[c].map(lambda x: f"{float(x):.3f}" if pd.notna(x) else "")
    print(out.to_string(index=False))


def delta(label: str, path: Path) -> None:
    print(f"\n=== {label} ===")
    if not path.exists():
        print(f"MISSING: {path}")
        return
    df = pd.read_csv(path)
    if "residual_model" in df.columns:
        df = df[df["residual_model"] == "residual_marl"]
    if "metric" in df.columns:
        df = df[df["metric"].isin(["closed_loop_score", "collision_events", "arrival_rate"])]
    print(df.to_string(index=False))


def main() -> None:
    sparse("FREEWAY sparse (no CTG++)", Path("Baselines/results/revision6/paper_2day/benchmark_summary.csv"))
    sparse("TGSIM sparse", Path("TGSIM Case/runs/paper_2day/results/benchmark_summary.csv"))

    rb = list(Path("Roundabout Case/runs").rglob("benchmark_summary.csv")) if Path("Roundabout Case/runs").exists() else []
    if not rb:
        print("\n=== ROUNDABOUT ===")
        print("No benchmark_summary.csv (paper_2day deleted after 3x cleanup).")
        print("runs present:", [p.name for p in Path("Roundabout Case/runs").iterdir()] if Path("Roundabout Case/runs").exists() else [])
    else:
        for p in rb:
            sparse(f"ROUNDABOUT {p}", p)

    delta("FREEWAY residual vs prior (param/stress)", Path("Baselines/results/revision6/param_lookahead_stress/ablation_stress_delta.csv"))
    delta("FREEWAY gate", Path("Baselines/results/revision6/gate_ablation/ablation_stress_delta.csv"))
    delta("TGSIM residual vs prior (param/stress)", Path("TGSIM Case/runs/paper_2day/results/param_lookahead_stress/ablation_stress_delta.csv"))
    delta("TGSIM gate", Path("TGSIM Case/runs/paper_2day/results/gate_ablation/ablation_stress_delta.csv"))

    log = Path("Baselines/results/revision6/paper_2day/pipeline_no_ctg.out.log")
    if log.exists():
        text = log.read_text(encoding="utf-8", errors="ignore")
        print("\n=== FREEWAY pipeline log stage ===")
        for key in ["START sparse", "OK    sparse", "START param", "OK    param", "START gate", "OK    gate", "2-day pipeline complete", "=== stress ==="]:
            if key in text:
                print(" found:", key)
        print(" tail:")
        print("\n".join(text.strip().splitlines()[-8:]))


if __name__ == "__main__":
    main()
