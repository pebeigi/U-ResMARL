"""Compare old live calibration JSONs to recalibration_20260922_p99."""
from __future__ import annotations

import json
from pathlib import Path

KEYS = [
    "S_theta", "S_v", "xi_i", "S_d", "gamma", "w_x", "w_y",
    "w_c", "w_ell", "beta", "sigma_long", "sigma_lat",
]


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def grid_shape(result: dict) -> tuple[int, int]:
    grid = result.get("candidate_grid") or {}
    return len(grid.get("acceleration") or []), len(grid.get("steering") or [])


def main() -> None:
    pairs = [
        (
            "Highway",
            Path("Calibration/utility_calibration.json"),
            Path("Calibration/runs/recalibration_20260922_p99/utility_calibration_highway.json"),
        ),
        (
            "Jounieh",
            Path("Calibration/utility_calibration_jounieh.json"),
            Path("Calibration/runs/recalibration_20260922_p99/utility_calibration_jounieh.json"),
        ),
        (
            "TGSIM",
            Path("Calibration/utility_calibration_tgsim.json"),
            Path("Calibration/runs/recalibration_20260922_p99/utility_calibration_tgsim.json"),
        ),
    ]

    print("PROTOCOL / LOSS (old live paper numbers -> new p99 run)")
    for name, old_path, new_path in pairs:
        old, new = load(old_path), load(new_path)
        oa, os_ = grid_shape(old)
        na, ns = grid_shape(new)
        print(f"\n{name}")
        print(
            f"  trials: {old.get('n_trials')}x{old.get('n_restarts')} -> "
            f"{new.get('n_trials')}x{new.get('n_restarts')}"
        )
        print(
            f"  CL cand scored: {old.get('closed_loop_candidates')} -> "
            f"{new.get('closed_loop_candidates')}"
        )
        print(
            f"  train windows: {old.get('n_rollout_windows')} -> "
            f"{new.get('n_rollout_windows')}"
        )
        print(f"  action grid: {oa}x{os_} -> {na}x{ns}")
        print(f"  vmax: {old.get('max_agent_speed')} -> {new.get('max_agent_speed')}")
        print(
            f"  working src: {old.get('working_params_source')} -> "
            f"{new.get('working_params_source')}"
        )
        print(
            f"  test CL: {old['test_closed_loop_loss']:.3f} -> "
            f"{new['test_closed_loop_loss']:.3f}"
        )
        print(
            f"  best CL: {old['best_closed_loop_loss']:.3f} -> "
            f"{new['best_closed_loop_loss']:.3f}"
        )
        print(
            f"  robust CL: {old['robust_closed_loop_loss']:.3f} -> "
            f"{new['robust_closed_loop_loss']:.3f}"
        )
        idinfo = new.get("identifiability", {})
        print(
            f"  near-opt (new): {idinfo.get('n_near_optimal')}/"
            f"{idinfo.get('n_scored_trials')}; top_fraction={new.get('top_fraction')}"
        )

    print("\nWORKING POINT + NEAR-OPTIMAL CLOUD p50/p95")
    print("(paper tables use working point; paper_summary also reports 95% percentiles)")
    for name, old_path, new_path in pairs:
        old, new = load(old_path), load(new_path)
        owp, nwp = old["working_params"], new["working_params"]
        orange = old.get("near_optimal_ranges") or old["recommended_ranges_from_top_trials"]
        nrange = new.get("near_optimal_ranges") or new["recommended_ranges_from_top_trials"]
        print("=" * 78)
        print(name)
        header = (
            f"{'param':12s} {'old_work':>10s} {'new_work':>10s} {'d':>8s} | "
            f"{'old_p50':>8s} {'old_p95':>8s} | {'new_p50':>8s} {'new_p95':>8s}"
        )
        print(header)
        for key in KEYS:
            ow, nw = owp[key], nwp[key]
            print(
                f"{key:12s} {ow:10.3f} {nw:10.3f} {nw - ow:+8.3f} | "
                f"{orange[key]['p50']:8.3f} {orange[key]['p95']:8.3f} | "
                f"{nrange[key]['p50']:8.3f} {nrange[key]['p95']:8.3f}"
            )


if __name__ == "__main__":
    main()
