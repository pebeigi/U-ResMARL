"""Build TGSIM recorded-scene caches for CtRL-Sim / CTG++ (no PCA lanes)."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from Baselines.data_evaluation import (
    DATA_PROTOCOL_VERSION,
    RecordedScene,
    file_sha256,
    horizon_steps,
    resample_track,
    temporal_partition,
)
from Baselines.scenario import AgentInit, Scenario
from RL.boundary import BoundaryInfeasibleError, filter_boundary_control
from RL.corridor import boxes_overlap, load_corridor
from RL.traffic_env import EnvConfig


def _last_finite(points: np.ndarray) -> np.ndarray | None:
    finite = np.isfinite(points).all(axis=1)
    if not finite.any():
        return None
    return points[int(np.flatnonzero(finite)[-1])]


def make_tgsim_recorded_scene(
    tracks: dict[int, pd.DataFrame],
    start: float,
    cfg: EnvConfig,
    *,
    seed: int,
    history_steps: int,
    max_gap: float,
    max_agents: int = 16,
) -> RecordedScene | None:
    corridor = load_corridor(cfg.run_id, cfg.lane_kf)
    history_times = start + np.arange(-history_steps, 1) * cfg.dt
    future_times = start + np.arange(1, cfg.max_steps + 1) * cfg.dt
    initial, references, excluded = [], [], {}
    for vehicle_id, track in sorted(tracks.items()):
        history = resample_track(track, history_times, max_gap, cutoff=start, lane_kf=None)
        if not np.isfinite(history[-1]).all() or not np.isfinite(history).all():
            excluded[str(vehicle_id)] = "missing_history"
            continue
        if not corridor.inside(history[-1], margin=0.5):
            excluded[str(vehicle_id)] = "off_curb"
            continue
        velocity = (history[-1] - history[-2]) / cfg.dt
        speed = float(np.linalg.norm(velocity))
        if speed > cfg.sim_config["max_agent_speed"]:
            excluded[str(vehicle_id)] = "initial_speed_outside_model_range"
            continue
        heading = float(np.arctan2(velocity[1], velocity[0])) if speed > 0.05 else 0.0
        future = resample_track(track, future_times, max_gap, lane_kf=None)
        future[~np.logical_and.accumulate(np.isfinite(future).all(axis=1))] = np.nan
        on_road_future = future.copy()
        for t, point in enumerate(on_road_future):
            if np.isfinite(point).all() and not corridor.inside(point, margin=1.):
                on_road_future[t] = np.nan
        dest = _last_finite(on_road_future)
        if dest is None:
            dest = history[-1].copy()
        if float(np.linalg.norm(dest - history[-1])) < 4.0:
            excluded[str(vehicle_id)] = "no_travel"
            continue
        init = AgentInit(
            vehicle_id,
            history[-1].copy(),
            velocity,
            heading,
            dest.copy(),
            0.0,
            0.0,
            cfg.base_desired_speed,
        )
        temporary = Scenario(
            seed, cfg.run_id, cfg.lane_kf, cfg.dt, cfg.max_steps,
            dict(cfg.sim_config), [init], corridor,
        )
        try:
            filter_boundary_control(temporary.spawn_agents()[0], (0.0, 0.0), cfg.sim_config, corridor)
        except BoundaryInfeasibleError:
            excluded[str(vehicle_id)] = "no_boundary_feasible_initial_control"
            continue
        initial.append(init)
        references.append(np.concatenate([history, future]))
    if len(initial) < 2:
        return None
    if len(initial) > max_agents:
        centroid = np.mean([a.pos for a in initial], axis=0)
        order = np.argsort([float(np.linalg.norm(a.pos - centroid)) for a in initial])[:max_agents]
        initial = [initial[i] for i in order]
        references = [references[i] for i in order]
    overlaps = sum(
        boxes_overlap(a.pos, a.heading, b.pos, b.heading, cfg.vehicle_length, cfg.vehicle_width)
        for i, a in enumerate(initial)
        for b in initial[i + 1 :]
    )
    scenario = Scenario(
        seed, cfg.run_id, cfg.lane_kf, cfg.dt, cfg.max_steps,
        dict(cfg.sim_config), initial, corridor,
    )
    return RecordedScene(scenario, np.stack(references, axis=1), history_steps, {
        "start_time_s": float(start),
        "agent_ids": [a.agent_id for a in initial],
        "excluded_initial_agents": excluded,
        "initial_overlap_pairs": int(overlaps),
        "initial_positions": [a.pos.tolist() for a in initial],
        "initial_velocities": [a.vel.tolist() for a in initial],
    })


def build_tgsim_recorded_scenes(
    csv_path: Path,
    *,
    count: int,
    seed: int,
    dt: float,
    horizons,
    split: str,
    history_s: float,
    base_desired_speed: float = 8.0,
) -> tuple[list[RecordedScene], dict]:
    steps = horizon_steps(horizons, dt)
    history_steps = horizon_steps([history_s], dt)[0]
    source = Path(csv_path).resolve()
    frame = pd.read_csv(source, usecols=["run_id", "lane_kf", "id", "time", "xloc_kf", "yloc_kf", "keep_ego"])
    frame = frame.sort_values(["id", "time"])
    intervals, membership = temporal_partition(frame)
    lo, hi = intervals[split]
    tracks = {int(i): g.drop(columns=["keep_ego"], errors="ignore") for i, g in frame.groupby("id", sort=True)}
    ego_ids = set(frame.loc[frame["keep_ego"] == True, "id"].astype(int).unique())
    native_dt = float(frame.groupby("id").time.diff().median())
    max_gap = 1.5 * native_dt + 1e-8
    cfg = EnvConfig(
        dt=dt, max_steps=max(steps), run_id=1, lane_kf=0,
        base_desired_speed=base_desired_speed, obb_safety_filter=True,
    )
    span = history_s + max(steps) * dt
    block_s = max(30.0, span)
    origin = float(frame.time.min())
    starts = []
    block_start = lo
    while block_start + span <= hi + 1e-8:
        first = origin + np.ceil((block_start + history_s - origin) / dt) * dt
        starts.extend(np.arange(first, min(block_start + block_s, hi) - max(steps) * dt + 1e-8, span))
        block_start += block_s
    scenes, rejected = [], {"cross_partition_context": 0, "insufficient_valid_agents": 0}
    for start in np.random.default_rng(seed).permutation(starts):
        present = {}
        for i, track in tracks.items():
            past = track[track.time <= start + 1e-8]
            if past.empty or start - float(past.time.iloc[-1]) > max_gap:
                continue
            if i not in ego_ids:
                continue
            present[i] = track
        if any(membership.get(i) != split for i in present):
            rejected["cross_partition_context"] += 1
            continue
        scene = make_tgsim_recorded_scene(
            present, float(start), cfg, seed=seed + len(scenes),
            history_steps=history_steps, max_gap=max_gap,
        )
        if scene is None:
            rejected["insufficient_valid_agents"] += 1
            continue
        scene.metadata.update(scene_seed=scene.scenario.seed, split=split,
                              time_block=int((start - lo) // block_s))
        scenes.append(scene)
        if len(scenes) == count:
            break
    if len(scenes) != count:
        raise ValueError(
            f"Only {len(scenes)} eligible TGSIM scenes for {count} requested; "
            f"rejections={rejected}"
        )
    manifest = {
        "data_protocol_version": DATA_PROTOCOL_VERSION,
        "site_protocol": cfg.sim_config['site_protocol'],
        "data_csv": str(source),
        "data_sha256": file_sha256(source),
        "run_id": 0,
        "lane_kf": 0,
        "dt": dt,
        "horizons_s": [s * dt for s in steps],
        "history_s": history_s,
        "split": split,
        "split_intervals_s": intervals,
        "split_fractions": [0.6, 0.2, 0.2],
        "split_track_ids": {
            name: [i for i, part in membership.items() if part == name] for name in intervals
        },
        "purged_cross_partition_track_ids": sorted(set(tracks) - set(membership)),
        "holdout_status": "retrospective_partition_prior_exposure_unverified",
        "holdout_note": "TGSIM site-curb recorded scenes; lanes unused.",
        "initialization": "observed position and backward velocity from history only",
        "goal": "last observed on-road point in the window",
        "desired_speed_mps": base_desired_speed,
        "rejected_candidates": rejected,
        "scenes": [scene.metadata for scene in scenes],
    }
    return scenes, manifest


def prepare_tgsim_new_baselines(output_dir: Path, csv_path: Path, cfg, *, train_scenes: int, val_scenes: int, seed: int = 0):
    from new_baselines.data import scene_arrays

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    future_steps = cfg.context + cfg.horizon + 1
    for split, count in (("train", train_scenes), ("validation", val_scenes)):
        scenes, manifest = build_tgsim_recorded_scenes(
            csv_path, count=count, seed=seed, dt=cfg.dt,
            horizons=[future_steps * cfg.dt], split=split,
            history_s=cfg.dt * max(cfg.history, 2),
        )
        arrays, diagnostics = {}, []
        for i, scene in enumerate(scenes):
            values, diag = scene_arrays(scene, cfg)
            arrays.update({f"{i}_{k}": v for k, v in values.items()})
            diagnostics.append(diag)
        path = output_dir / f"{split}.npz"
        np.savez_compressed(path, **arrays)
        manifest.update(
            config=cfg.to_dict(),
            scene_count=len(scenes),
            inverse_dynamics=diagnostics,
            new_baselines_data_version=1,
        )
        path.with_suffix(".json").write_text(json.dumps(manifest, indent=2) + "\n")
        print(f"{split}: {len(scenes)} TGSIM recorded scenes -> {path}", flush=True)
    return output_dir


def generation_config():
    from new_baselines.config import Config
    from config import (MAX_AGENT_SPEED, PERCEPTION_RADIUS, VEHICLE_LENGTH,
                        VEHICLE_WIDTH, VEHICLE_WHEELBASE)
    return Config(length=VEHICLE_LENGTH, width=VEHICLE_WIDTH,
                  wheelbase=VEHICLE_WHEELBASE, max_speed=MAX_AGENT_SPEED,
                  steer_from_rest=True, perception_radius=PERCEPTION_RADIUS).validate()


if __name__ == '__main__':
    from config import (NEW_BASELINE_DATA, TRAJECTORIES_CSV, NEW_BASELINE_TRAIN_SCENES,
                        NEW_BASELINE_VAL_SCENES)
    prepare_tgsim_new_baselines(NEW_BASELINE_DATA, TRAJECTORIES_CSV, generation_config(),
        train_scenes=NEW_BASELINE_TRAIN_SCENES, val_scenes=NEW_BASELINE_VAL_SCENES)
