"""Recorded-scene evaluation, with references kept outside controller inputs.

The retrospective time split is reproducible, but cannot certify that an old
calibration/checkpoint never used the source recordings. The manifest says so.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

from Baselines.scenario import AgentInit, Scenario
from RL.boundary import BoundaryInfeasibleError, filter_boundary_control
from RL.corridor import boxes_overlap, load_corridor
from RL.traffic_env import EnvConfig

DATA_PROTOCOL_VERSION = 1
DEFAULT_HORIZONS = (1.0, 3.0, 5.0)
REQUIRED_COLUMNS = ("run_id", "lane_kf", "id", "time", "xloc_kf", "yloc_kf")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def horizon_steps(horizons, dt: float) -> list[int]:
    values = np.asarray(horizons, dtype=float)
    if (not np.isfinite(dt) or dt <= 0 or values.size == 0
            or not np.isfinite(values).all() or (values <= 0).any()):
        raise ValueError("Horizons and dt must be positive and finite")
    steps = np.rint(values / dt).astype(int)
    if (steps < 1).any() or not np.allclose(steps * dt, values, rtol=0, atol=1e-8):
        raise ValueError("Every evaluation horizon must be an integer multiple of dt")
    return sorted(set(steps.tolist()))


def temporal_partition(frame: pd.DataFrame) -> tuple[dict, dict[int, str]]:
    """60/20/20 time split; whole tracks crossing a cut belong to no split.

    Call on the whole recording (all lanes), so lane changes cannot put the
    same vehicle into different partitions.
    """
    lo, hi = float(frame.time.min()), float(frame.time.max())
    if not np.isfinite([lo, hi]).all() or hi <= lo:
        raise ValueError("A recording needs at least two distinct timestamps")
    cuts = [lo, lo + .6 * (hi - lo), lo + .8 * (hi - lo), hi]
    intervals = dict(zip(("train", "validation", "test"), zip(cuts[:-1], cuts[1:])))
    membership = {}
    for vehicle_id, group in frame.groupby("id", sort=True):
        start, end = float(group.time.min()), float(group.time.max())
        for name, (left, right) in intervals.items():
            if start >= left and (end < right or (name == "test" and end <= right)):
                membership[int(vehicle_id)] = name
                break
    return intervals, membership


def resample_track(track: pd.DataFrame, times: np.ndarray, max_gap: float,
                   *, cutoff: float | None = None, lane_kf: int | None = None) -> np.ndarray:
    """Linear interpolation with no extrapolation or interpolation across gaps.

    A cutoff is used for initialization: even interpolation may not read a
    sample after the current time. Lane exits are missing observations.
    """
    if cutoff is not None:
        track = track[track.time <= cutoff + 1e-8]
    out = np.full((len(times), 2), np.nan)
    if track.empty:
        return out
    t = track.time.to_numpy(float)
    xy = track[["xloc_kf", "yloc_kf"]].to_numpy(float)
    right = np.searchsorted(t, times)
    right = np.clip(right, 0, len(t) - 1)
    exact = np.abs(t[right] - times) < 1e-7
    left = np.where(exact, right, np.maximum(right - 1, 0))
    valid = ((times >= t[0] - 1e-8) & (times <= t[-1] + 1e-8)
             & (exact | ((t[right] - t[left]) <= max_gap)))
    if lane_kf is not None:
        lane = track.lane_kf.to_numpy()
        valid &= (lane[left] == lane_kf) & (lane[right] == lane_kf)
    for dim in range(2):
        out[valid, dim] = np.interp(times[valid], t, xy[:, dim])
    return out


@dataclass
class RecordedScene:
    scenario: Scenario
    # Includes history and t=0. Controllers only receive scenario, never this.
    reference_positions: np.ndarray
    history_steps: int
    metadata: dict


def make_recorded_scene(tracks: dict[int, pd.DataFrame], start: float,
                        cfg: EnvConfig, *, seed: int, history_steps: int,
                        max_gap: float) -> RecordedScene | None:
    corridor = load_corridor(cfg.run_id, cfg.lane_kf)
    history_times = start + np.arange(-history_steps, 1) * cfg.dt
    future_times = start + np.arange(1, cfg.max_steps + 1) * cfg.dt
    initial, references, excluded = [], [], {}
    destination_s = corridor.length - 5.0
    destination, _ = corridor.xy_from_frenet(destination_s, 0.0)
    for vehicle_id, track in sorted(tracks.items()):
        history = resample_track(track, history_times, max_gap, cutoff=start,
                                 lane_kf=cfg.lane_kf)
        if not np.isfinite(history[-1]).all():
            continue
        if not np.isfinite(history).all():
            excluded[str(vehicle_id)] = "missing_history"
            continue
        velocity = (history[-1] - history[-2]) / cfg.dt
        speed = float(np.linalg.norm(velocity))
        station, _, tangent, _, _ = corridor.project(history[-1])
        heading = float(np.arctan2(*(velocity if speed > .05 else tangent)[::-1]))
        if speed > cfg.sim_config["max_agent_speed"]:
            excluded[str(vehicle_id)] = "initial_speed_outside_model_range"
            continue
        if station >= destination_s - cfg.sim_config.get("destination_threshold", 1.0):
            excluded[str(vehicle_id)] = "at_corridor_exit"
            continue
        if velocity @ tangent < -.1:
            excluded[str(vehicle_id)] = "opposite_travel_direction"
            continue
        # Desired speed is fixed before evaluation; never a future track quantile.
        init = AgentInit(vehicle_id, history[-1].copy(), velocity, heading,
                         destination.copy(), destination_s, float(station),
                         cfg.base_desired_speed)
        temporary = Scenario(seed, cfg.run_id, cfg.lane_kf, cfg.dt, cfg.max_steps,
                             dict(cfg.sim_config), [init], corridor)
        try:
            filter_boundary_control(temporary.spawn_agents()[0], (0., 0.),
                                    cfg.sim_config, corridor)
        except BoundaryInfeasibleError:
            excluded[str(vehicle_id)] = "no_boundary_feasible_initial_control"
            continue
        future = resample_track(track, future_times, max_gap, lane_kf=cfg.lane_kf)
        # Once a track exits/has a gap, do not resume scoring it later.
        future[~np.logical_and.accumulate(np.isfinite(future).all(axis=1))] = np.nan
        initial.append(init)
        references.append(np.concatenate([history, future]))
    if len(initial) < 2:
        return None
    # Preserve initial data contacts instead of removing cars to improve scores.
    overlaps = sum(boxes_overlap(a.pos, a.heading, b.pos, b.heading,
                                 cfg.vehicle_length, cfg.vehicle_width)
                   for i, a in enumerate(initial) for b in initial[i + 1:])
    scenario = Scenario(seed, cfg.run_id, cfg.lane_kf, cfg.dt, cfg.max_steps,
                        dict(cfg.sim_config), initial, corridor)
    return RecordedScene(scenario, np.stack(references, axis=1), history_steps, {
        "start_time_s": float(start),
        "agent_ids": [a.agent_id for a in initial],
        "excluded_initial_agents": excluded,
        "initial_overlap_pairs": int(overlaps),
        "initial_positions": [a.pos.tolist() for a in initial],
        "initial_velocities": [a.vel.tolist() for a in initial],
    })


def build_recorded_scenes(csv_path: Path, *, count: int, seed: int, run_id: int,
                          lane_kf: int, dt: float = .5, horizons=DEFAULT_HORIZONS,
                          split: str = "test", history_s: float = 1.,
                          base_desired_speed: float = 8.,
                          obb_safety_filter: bool = True) -> tuple[list[RecordedScene], dict]:
    steps = horizon_steps(horizons, dt)
    history_steps = horizon_steps([history_s], dt)[0]
    if count <= 0 or split not in ("train", "validation", "test"):
        raise ValueError("Positive scene count and train/validation/test split required")
    source = Path(csv_path).resolve()
    frame = pd.read_csv(source, usecols=list(REQUIRED_COLUMNS))
    frame = frame[frame.run_id == run_id].sort_values(["id", "time"])
    if frame.empty or not np.isfinite(frame[list(REQUIRED_COLUMNS)].to_numpy()).all():
        raise ValueError("Recording is empty or contains non-finite required values")
    if frame.duplicated(["id", "time"]).any():
        raise ValueError("Duplicate vehicle/timestamp rows in recording")
    intervals, membership = temporal_partition(frame)
    lo, hi = intervals[split]
    tracks = {int(i): g for i, g in frame.groupby("id", sort=True)}
    native_dt = float(frame.groupby("id").time.diff().median())
    if not np.isfinite(native_dt) or native_dt <= 0:
        raise ValueError("Cannot infer recording sample interval")
    max_gap = 1.5 * native_dt + 1e-8
    cfg = EnvConfig(dt=dt, max_steps=max(steps), run_id=run_id, lane_kf=lane_kf,
                    base_desired_speed=base_desired_speed, obb_safety_filter=obb_safety_filter)
    # Non-overlapping history+future windows; cluster statistics by time block.
    span = history_s + max(steps) * dt
    block_s = max(30., span)
    origin = float(frame.time.min())
    starts = []
    block_start = lo
    while block_start + span <= hi + 1e-8:
        first = origin + np.ceil((block_start + history_s - origin) / dt) * dt
        starts.extend(np.arange(first, min(block_start + block_s, hi) - max(steps) * dt + 1e-8, span))
        block_start += block_s
    if not starts:
        raise ValueError("Requested split is too short for the history and horizons")
    scenes, rejected = [], {"cross_partition_context": 0, "insufficient_valid_agents": 0}
    # Presence comes from the last recorded sample at/before start, no future.
    for start in np.random.default_rng(seed).permutation(starts):
        present = {}
        for i, track in tracks.items():
            past = track[track.time <= start + 1e-8]
            if (not past.empty and start - float(past.time.iloc[-1]) <= max_gap
                    and int(past.lane_kf.iloc[-1]) == lane_kf):
                present[i] = track
        if any(membership.get(i) != split for i in present):
            rejected["cross_partition_context"] += 1
            continue
        scene = make_recorded_scene(present, float(start), cfg, seed=seed + len(scenes),
                                    history_steps=history_steps, max_gap=max_gap)
        if scene is None:
            rejected["insufficient_valid_agents"] += 1
            continue
        scene.metadata.update(scene_seed=scene.scenario.seed, split=split,
                              time_block=int((start - lo) // block_s))
        scenes.append(scene)
        if len(scenes) == count:
            break
    if len(scenes) != count:
        raise ValueError(f"Only {len(scenes)} eligible scenes for {count} requested; "
                         f"reduce --scenarios or use more recordings. Rejections: {rejected}")
    manifest = {
        "data_protocol_version": DATA_PROTOCOL_VERSION,
        "data_csv": str(source), "data_sha256": file_sha256(source),
        "run_id": run_id, "lane_kf": lane_kf, "dt": dt,
        "horizons_s": [s * dt for s in steps], "history_s": history_s,
        "split": split, "split_intervals_s": intervals, "split_fractions": [.6, .2, .2],
        "split_track_ids": {name: [i for i, part in membership.items() if part == name]
                            for name in intervals},
        "purged_cross_partition_track_ids": sorted(set(tracks) - set(membership)),
        "holdout_status": "retrospective_partition_prior_exposure_unverified",
        "holdout_note": "Disjoint data partitions do not establish independence from existing "
                        "calibration or policy fitting. Historical calibration sampled one-step "
                        "choices before splitting rollout windows and saved no exposure IDs.",
        "initialization": "observed position and backward velocity from history only",
        "goal": "fixed corridor exit, 5 m before end; no observed future endpoints",
        "desired_speed_mps": base_desired_speed,
        "cohort": "all eligible agents present at start; no later entrants; reference "
                  "ends at first gap/lane exit; all simulated agents remain interactive",
        "vehicle_length_m": cfg.vehicle_length, "vehicle_width_m": cfg.vehicle_width,
        "footprint_note": "Shared model footprint, not individual measured vehicle dimensions",
        "sampling": "non-overlapping history+future windows, sampled without replacement",
        "statistics_block_s": block_s,
        "rejected_candidates": rejected,
        "scenes": [scene.metadata for scene in scenes],
    }
    return scenes, manifest


def motion_arrays(positions: np.ndarray, dt: float, initial_headings: np.ndarray) -> dict:
    """Same causal finite differences for observations and generated positions."""
    valid = np.isfinite(positions).all(axis=-1)
    velocity = np.full_like(positions, np.nan)
    velocity[1:] = np.diff(positions, axis=0) / dt
    speed = np.linalg.norm(velocity, axis=-1)
    heading = np.broadcast_to(initial_headings, valid.shape).copy()
    for t in range(1, len(positions)):
        heading[t] = heading[t - 1]
        moving = np.isfinite(speed[t]) & (speed[t] > .05)
        heading[t, moving] = np.arctan2(velocity[t, moving, 1], velocity[t, moving, 0])
    accel = np.full_like(speed, np.nan)
    accel[1:] = np.diff(speed, axis=0) / dt
    yaw_rate = np.full_like(speed, np.nan)
    yaw_rate[1:] = np.arctan2(np.sin(np.diff(heading, axis=0)),
                             np.cos(np.diff(heading, axis=0))) / dt
    yaw_rate[~(valid & np.roll(valid, 1, axis=0))] = np.nan
    return dict(positions=positions, valid=valid, velocity=velocity, speed=speed,
                heading=heading, accel=accel, yaw_rate=yaw_rate)


def aligned_motion(result, scene: RecordedScene) -> tuple[dict, dict]:
    if result.steps != scene.scenario.max_steps or result.num_agents != scene.scenario.num_agents:
        raise ValueError("Recorded evaluation requires the entire fixed horizon and cohort")
    if not np.isclose(result.dt, scene.scenario.dt):
        raise ValueError("Simulation and reference dt differ")
    if not np.allclose(result.positions[0], scene.reference_positions[scene.history_steps]):
        raise ValueError("Simulation and reference initial states/agent order differ")
    if not np.isfinite(result.positions).all():
        raise ValueError("Non-finite generated trajectory; cannot omit failed predictions")
    history = scene.reference_positions[:scene.history_steps]
    headings = np.array([a.heading for a in scene.scenario.agents])
    simulated = motion_arrays(np.concatenate([history, result.positions]), result.dt, headings)
    observed = motion_arrays(scene.reference_positions, result.dt, headings)
    return simulated, observed


def trajectory_metrics(result, scene: RecordedScene, horizons=DEFAULT_HORIZONS) -> dict:
    simulated, observed = aligned_motion(result, scene)
    start = scene.history_steps
    out = {}
    for steps in horizon_steps(horizons, result.dt):
        if steps > result.steps:
            raise ValueError("Requested trajectory horizon exceeds rollout length")
        suffix = f"{steps * result.dt:g}s"
        window = slice(start + 1, start + steps + 1)
        # Only full reference tracks, independent of model arrival/collision.
        eligible = observed["valid"][start:start + steps + 1].all(axis=0)
        out[f"trajectory_agents_{suffix}"] = int(eligible.sum())
        out[f"trajectory_coverage_{suffix}"] = float(eligible.mean())
        delta = simulated["positions"][window, eligible] - observed["positions"][window, eligible]
        distances = np.linalg.norm(delta, axis=-1)
        out[f"ade_{suffix}_m"] = float(distances.mean()) if eligible.any() else float("nan")
        out[f"fde_{suffix}_m"] = float(distances[-1].mean()) if eligible.any() else float("nan")
        speed_error = simulated["speed"][window, eligible] - observed["speed"][window, eligible]
        out[f"speed_rmse_{suffix}_mps"] = float(np.sqrt(np.mean(speed_error ** 2))) if eligible.any() else float("nan")
        angle = simulated["heading"][window, eligible] - observed["heading"][window, eligible]
        # Observed heading is undefined at rest; a stopped prediction against a
        # moving observation still counts using its last generated heading.
        moving = observed["speed"][window, eligible] > .05
        out[f"heading_mae_{suffix}_rad"] = (
            float(np.abs(np.arctan2(np.sin(angle[moving]), np.cos(angle[moving]))).mean())
            if moving.any() else float("nan"))
        out[f"heading_samples_{suffix}"] = int(moving.sum())
    return out
