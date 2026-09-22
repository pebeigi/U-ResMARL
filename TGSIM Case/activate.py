"""Point the existing RL/benchmark stack at TGSIM without editing repo files.

Geometry is the extracted Foggy Bottom curb polygon (same as TGSIM calibration),
not a PCA lane tube. Decision, local observations, OBB checks, and the bicycle
grid stay on the highway protocol in RL.decision / Baselines.
"""
from __future__ import annotations

import numpy as np

from config import (
    BASE_DESIRED_SPEED,
    CALIBRATION,
    CHECKPOINT_DIR,
    DESTINATION_THRESHOLD,
    MIN_INITIAL_SPACING,
    MAX_AGENT_SPEED,
    PERCEPTION_RADIUS,
    REPO_ROOT,
    STREET_BOUNDARIES,
    TRAJECTORIES_CSV,
    VEHICLE_LENGTH,
    VEHICLE_WIDTH,
    VEHICLE_WHEELBASE,
)

_APPLIED = False


def apply() -> None:
    """Monkeypatch corridor, spawn, utility frame, and checkpoint locations."""
    global _APPLIED
    if _APPLIED:
        return
    import sys

    root = str(REPO_ROOT)
    if root not in sys.path:
        sys.path.insert(0, root)

    from network import load_site_corridor, road_region_from_site

    import RL.calibration_io as calibration_io
    import RL.candidate_policy as candidate_policy
    import RL.corridor as corridor
    import RL.traffic_env as traffic_env
    import RL.transition as transition
    import RL.boundary as boundary
    import utility_model
    from Baselines import registry
    from Calibration.calibrate_utility_from_data import (
        destination_choice_features,
        polygon_path_error,
    )
    from RL.boundary import BoundaryInfeasibleError

    site = load_site_corridor(str(STREET_BOUNDARIES))
    vehicle_length, vehicle_width = VEHICLE_LENGTH, VEHICLE_WIDTH
    import hashlib
    from pathlib import Path
    source_paths = [CALIBRATION, STREET_BOUNDARIES, TRAJECTORIES_CSV, *sorted(Path(__file__).parent.glob('*.py'))]
    site_protocol = dict(version=5, name='tgsim_recorded_initialization', arrival='euclidean_own_goal',
        routing='clearance_visibility_graph', spawn='simultaneous_recorded_poses_with_braking_backup',
        goal='recorded_endpoint_given_as_navigation_intent',
        traffic_split='60_20_20_time_with_cross_partition_tracks_purged',
        validation_seed_block=[910000, 920000], test_seed_block=[810000, 820000],
        hashes={p.relative_to(REPO_ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in source_paths})

    def load_tgsim_corridor(*_args, **_kwargs):
        return site

    corridor.load_corridor = load_tgsim_corridor
    traffic_env.load_corridor = load_tgsim_corridor
    boundary.load_corridor = load_tgsim_corridor
    import RL.behavior_reference as behavior_reference
    import Baselines.scenario as scenario_mod
    behavior_reference.load_corridor = load_tgsim_corridor
    scenario_mod.load_corridor = load_tgsim_corridor
    traffic_env.DEFAULT_RUN_ID = 0
    traffic_env.DEFAULT_LANE_KF = 0
    corridor.DEFAULT_RUN_ID = 0
    corridor.DEFAULT_LANE_KF = 0

    original_road_region = boundary.road_region

    def tgsim_road_region(corridor_obj, margin=0.0):
        if getattr(corridor_obj, "roadway", None) is not None:
            return road_region_from_site(corridor_obj, margin=margin)
        return original_road_region(corridor_obj, margin=margin)

    boundary.road_region = tgsim_road_region

    original_post = traffic_env.EnvConfig.__post_init__

    def tgsim_post_init(self):
        self.run_id = self.lane_kf = 0
        self.min_initial_spacing = MIN_INITIAL_SPACING
        self.vehicle_length = vehicle_length
        self.vehicle_width = vehicle_width
        self.base_desired_speed = BASE_DESIRED_SPEED
        original_post(self)
        sim = self.sim_config
        sim["_site_roadway"] = site.roadway
        sim['site_protocol'] = site_protocol
        sim["_on_road_dest"] = site.on_road
        sim["utility_frame"] = "destination"
        sim["path_mode"] = "boundary"
        sim["perception_radius"] = PERCEPTION_RADIUS
        sim["steer_from_rest"] = True
        sim["min_steer_speed"] = 0.5
        sim["vehicle_length"] = vehicle_length
        sim["vehicle_width"] = vehicle_width
        sim["wheelbase"] = VEHICLE_WHEELBASE
        sim["destination_threshold"] = DESTINATION_THRESHOLD
        sim["max_agent_speed"] = MAX_AGENT_SPEED
        minx, miny, maxx, maxy = site.roadway.bounds
        sim["road_x_min"], sim["road_x_max"] = float(minx), float(maxx)
        sim["road_y_min"], sim["road_y_max"] = float(miny), float(maxy)
        sim["corridor_length"] = float(site.length)

    traffic_env.EnvConfig.__post_init__ = tgsim_post_init

    from recorded_spawns import RecordedSpawnPool, split_for_seed

    original_env_init = traffic_env.MultiAgentTrafficEnv.__init__
    spawn_pool = None

    def recorded_env_init(self, config=None, seed=None):
        self._recorded_split = split_for_seed(seed)
        original_env_init(self, config, seed)

    def recorded_spawn_once(self):
        nonlocal spawn_pool
        if spawn_pool is None:
            spawn_pool = RecordedSpawnPool(site, self.config.sim_config)
        return spawn_pool.sample(self)

    traffic_env.MultiAgentTrafficEnv.__init__ = recorded_env_init
    traffic_env.MultiAgentTrafficEnv._spawn_agents_once = recorded_spawn_once

    original_path = utility_model.path_adherence_penalty

    def tgsim_path_adherence(candidate_pos, nominal_y, params, sim_config=None):
        roadway = None if sim_config is None else sim_config.get("_site_roadway")
        if roadway is not None:
            ell_i = float(
                polygon_path_error(
                    np.atleast_2d(np.asarray(candidate_pos, dtype=float)),
                    roadway,
                    boundary_buffer=float(sim_config.get("boundary_buffer", 1.5)),
                )[0]
            )
            return params["w_ell"] * (1.0 - np.exp(-params["beta"] * ell_i ** 2))
        return original_path(candidate_pos, nominal_y, params, sim_config)

    utility_model.path_adherence_penalty = tgsim_path_adherence

    original_dir = utility_model.directional_alignment_utility
    original_dist = utility_model.distance_reward_utility
    original_evaluate = utility_model.evaluate_candidate_utility
    _tls = {}

    def tgsim_evaluate(*args, **kwargs):
        sim = args[5] if len(args) > 5 else kwargs.get("sim_config")
        _tls["sim"] = sim
        try:
            return original_evaluate(*args, **kwargs)
        finally:
            _tls.clear()

    def tgsim_dir(current_pos, candidate_pos, destination_pos, current_heading_vector, params):
        sim = _tls.get("sim")
        on_road = None if sim is None else sim.get("_on_road_dest")
        if on_road is None:
            return original_dir(
                current_pos, candidate_pos, destination_pos, current_heading_vector, params
            )
        heading = float(np.arctan2(current_heading_vector[1], current_heading_vector[0]))
        dir_cos, _ = destination_choice_features(
            np.asarray(current_pos, dtype=float),
            np.atleast_2d(np.asarray(candidate_pos, dtype=float)),
            np.asarray(destination_pos, dtype=float),
            heading=heading,
            on_road=on_road,
        )
        return params["S_theta"] * float(dir_cos[0])

    def tgsim_dist(candidate_pos, destination_pos, current_speed, params, sim_config):
        on_road = sim_config.get("_on_road_dest") if sim_config is not None else None
        if on_road is None:
            return original_dist(
                candidate_pos, destination_pos, current_speed, params, sim_config
            )
        heading = None
        _, dist_abs = destination_choice_features(
            np.asarray(candidate_pos, dtype=float),
            np.atleast_2d(np.asarray(candidate_pos, dtype=float)),
            np.asarray(destination_pos, dtype=float),
            heading=heading,
            on_road=on_road,
        )
        remaining, lat = float(dist_abs[0, 0]), float(dist_abs[0, 1])
        d_eff = params["w_x"] * remaining + params["w_y"] * lat
        h_p = sim_config["kappa_perception_horizon"] * current_speed
        h_p = max(h_p, sim_config["min_perception_horizon"])
        if h_p < 1e-6:
            return 0.0
        ratio = d_eff / h_p
        return params["S_d"] / (1.0 + ratio ** params["gamma"])

    utility_model.evaluate_candidate_utility = tgsim_evaluate
    utility_model.directional_alignment_utility = tgsim_dir
    utility_model.distance_reward_utility = tgsim_dist

    calibration_io.DEFAULT_CALIBRATION_PATH = CALIBRATION
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    registry.LEARNED_CHECKPOINTS["residual_marl"] = CHECKPOINT_DIR / "residual_policy.pt"
    registry.LEARNED_CHECKPOINTS["residual_no_gate"] = CHECKPOINT_DIR / "residual_policy.pt"
    registry.LEARNED_CHECKPOINTS["residual_param"] = CHECKPOINT_DIR / "residual_param_policy.pt"
    registry.LEARNED_CHECKPOINTS["residual_weights_only"] = CHECKPOINT_DIR / "residual_param_policy.pt"
    registry.LEARNED_CHECKPOINTS["residual_sigma_only"] = CHECKPOINT_DIR / "residual_param_policy.pt"
    registry.LEARNED_CHECKPOINTS["residual_sigma_frozen"] = CHECKPOINT_DIR / "residual_param_policy.pt"
    registry.LEARNED_CHECKPOINTS["direct_discrete_rl"] = CHECKPOINT_DIR / "direct_discrete_policy.pt"
    registry.LEARNED_CHECKPOINTS["mappo"] = CHECKPOINT_DIR / "mappo_policy.pt"
    registry.LEARNED_CHECKPOINTS["ctrl_sim"] = CHECKPOINT_DIR / "ctrl_sim_policy.pt"
    registry.LEARNED_CHECKPOINTS["ctg_plus_plus"] = CHECKPOINT_DIR / "ctg_plus_plus_policy.pt"

    sys.path.insert(0, str(REPO_ROOT / "New Baselines"))
    _APPLIED = True
