"""Quick Jounieh sim-vs-obs previews with current p99 / 7x9 site config."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from Calibration.calibrate_utility_from_data import (  # noqa: E402
    SceneTimeIndex,
    apply_urban_site_config,
    infer_default_class_id,
    load_and_prepare,
    nll,
    plot_all_trajectories,
    plot_vehicle_simulated_vs_observed,
    samples_for_vehicle,
    simulate_vehicle,
    working_params_from_result,
)
from RL.traffic_env import EnvConfig  # noqa: E402

DEFAULT_IDS = [12549, 41901, 54664, 6567, 11936, 3999, 72328, 4955]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--params-json",
        type=Path,
        default=ROOT / "Calibration" / "utility_calibration_jounieh.json",
        help="Calibration JSON providing working_params (live or fresh run).",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT / "Calibration" / "runs" / "recalibration_20260922_p99" / "preview_jounieh",
    )
    parser.add_argument("--ids", type=int, nargs="*", default=DEFAULT_IDS)
    args = parser.parse_args()

    csv = ROOT / "data" / "Lebanon_Jounieh" / "prepared" / "trajectories_calibration.csv"
    polygon = ROOT / "data" / "Lebanon_Jounieh" / "Jounieh_Road_Boundaries.csv"
    result = json.loads(args.params_json.read_text(encoding="utf-8"))
    params = working_params_from_result(result)

    ns = argparse.Namespace(
        csv=csv,
        site_polygon_csv=polygon,
        neighbor_radius=60.0,
        max_neighbors=6,
        max_agent_speed=None,
        boundary_csv=None,
        vehicle_length=4.5,
        vehicle_width=1.8,
        wheelbase=2.8,
        temperature=0.25,
        per_id_samples=80,
        verbose=True,
        class_id=None,
        seed=7,
        n_samples=2000,
    )
    class_id = infer_default_class_id(csv)
    df, scene_df = load_and_prepare(csv, class_id)
    cfg = EnvConfig()
    apply_urban_site_config(cfg, ns, df)
    print(
        f"vmax={cfg.sim_config['max_agent_speed']:.2f} "
        f"frame={cfg.sim_config.get('utility_frame')} "
        f"actions="
        f"{len(cfg.sim_config['candidate_accel_grid'])}x"
        f"{len(cfg.sim_config['candidate_steering_grid'])} "
        f"params_from={args.params_json.name}",
        flush=True,
    )

    out = args.out_dir
    out.mkdir(parents=True, exist_ok=True)
    per_id = out / "per_id"
    per_id.mkdir(parents=True, exist_ok=True)
    plot_all_trajectories(df, out / "all_agents_xy.png", site_polygon_csv=polygon)

    grouped = SceneTimeIndex(scene_df)
    wanted = set(int(i) for i in args.ids)
    rows = []
    for (run_id, vehicle_id), group in df.groupby(["run_id", "id"], sort=True):
        if int(vehicle_id) not in wanted:
            continue
        local = samples_for_vehicle(group, grouped, cfg.sim_config, ns, {})
        local_nll = nll(local, params, ns.temperature) if local else float("nan")
        sim_df = simulate_vehicle(
            group,
            grouped,
            params,
            cfg.sim_config,
            neighbor_radius=ns.neighbor_radius,
            max_neighbors=ns.max_neighbors,
            boundary_map={},
        )
        path = per_id / f"run_{int(run_id):02d}_id_{int(vehicle_id):06d}_sim_vs_obs.png"
        plot_vehicle_simulated_vs_observed(
            group,
            sim_df,
            params,
            local_nll,
            path,
            boundary_map={},
            site_polygon_csv=polygon,
            scene_df=scene_df,
        )
        disp = float(
            np.sum(
                np.linalg.norm(
                    np.diff(group.sort_values("time")[["xloc_kf", "yloc_kf"]].to_numpy(float), axis=0),
                    axis=1,
                )
            )
        )
        rows.append(
            {
                "run_id": int(run_id),
                "id": int(vehicle_id),
                "local_nll": local_nll,
                "disp_m": disp,
                "path": str(path),
            }
        )
        print(f"wrote {path.name} nll={local_nll:.3f} disp={disp:.1f}m", flush=True)

    pd.DataFrame(rows).to_csv(out / "preview_ids.csv", index=False)
    print(f"Done: {out}", flush=True)


if __name__ == "__main__":
    main()
