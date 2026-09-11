"""2D multi-agent traffic environment with utility-based action selection."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

import RL._paths  # noqa: F401 — put repo root on sys.path
from RL.calibration_io import RESIDUAL_PARAM_KEYS, apply_residual
from RL.corridor import (
    DEFAULT_LANE_KF,
    DEFAULT_RUN_ID,
    DEFAULT_VEHICLE_LENGTH,
    DEFAULT_VEHICLE_WIDTH,
    boundary_reward,
    boxes_overlap,
    corridor_sim_defaults,
    load_corridor,
)
from RL.behavior_reference import FEATURES as BEHAVIOR_FEATURES
from RL.behavior_reference import load_behavior_reference
from RL.obs import local_observation, observation_dim
from RL.transition import advance_agents
from utility_model import (
    DEFAULT_BASE_PARAMS,
    DEFAULT_SIM_CONFIG,
    TrafficAgent,
    kinematic_bicycle_rollout,
    sanitize_control_command,
    select_candidate_with_logit_residual,
)


@dataclass
class EnvConfig:
    dt: float = 0.5
    max_steps: int = 240
    num_agents: int = 10
    base_desired_speed: float = 8.0
    highway_length: float = 500.0  # overridden by corridor length when available
    spawn_s_range: tuple[float, float] = (20.0, 120.0)
    # Destination station: random near the corridor end (absolute s, not offset from start).
    dest_s_range_from_end: tuple[float, float] = (5.0, 40.0)
    spawn_lateral_frac: float = 0.35  # fraction of half-width used at spawn
    min_initial_spacing: float = 8.0
    run_id: int = DEFAULT_RUN_ID
    lane_kf: int = DEFAULT_LANE_KF
    vehicle_length: float = DEFAULT_VEHICLE_LENGTH
    vehicle_width: float = DEFAULT_VEHICLE_WIDTH
    reward_weights: dict[str, float] | None = None
    # Explicit OBB collision penalty added to the per-step reward of every agent
    # involved in a collision this step. Soft proximity (r_safety) alone is too
    # weak for residual learning to care about vehicle-scale overlaps.
    collision_penalty: float = 0.0
    # Behavioral data terms. ``behavior_coef`` weights a per-agent penalty for
    # leaving the observed band of each marginal. ``behavior_shaping_coef``
    # weights potential-based shaping on the episode's distance to the measured
    # marginals, which telescopes to the realism score actually reported, so the
    # distribution shape is optimized during training instead of only measured.
    behavior_coef: float = 0.0
    behavior_shaping_coef: float = 0.0
    behavior_warmup_steps: int = 10
    behavior_csv: str | None = None
    sim_config: dict[str, Any] | None = None
    base_params: dict[str, float] | None = None
    # ``candidate_logits``: residual adds to discrete utilities (default, continuous credit).
    # ``param_delta``: residual edits Θ (legacy / ablation).
    residual_mode: str = "candidate_logits"
    # Terms that align the dense reward with leftover-distance / arrival metrics.
    leftover_coef: float = 0.05
    arrival_bonus: float = 5.0
    # When set, overrides ``sim_config["obb_safety_filter"]`` (train default: False).
    obb_safety_filter: bool | None = None

    def __post_init__(self) -> None:
        if self.reward_weights is None:
            self.reward_weights = {
                "progress": 1.0,
                "safety": 0.5,
                "smooth": 0.2,
                "traj": 0.0,
            }
        corridor = load_corridor(self.run_id, self.lane_kf)
        self.highway_length = float(corridor.length)
        if self.sim_config is None:
            self.sim_config = dict(DEFAULT_SIM_CONFIG)
            self.sim_config["dt"] = self.dt
            self.sim_config.update(corridor_sim_defaults(corridor))
            self.sim_config.update(
                {
                    "perception_radius": 60.0,
                    "max_neighbors": 6,
                    "max_agent_speed": 16.0,
                    "collision_threshold": 1.5,  # unused when OBB collisions enabled
                    "use_obb_collisions": True,
                    "wheelbase": 2.8,
                    "candidate_accel_grid": [-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0],
                    "candidate_steering_grid": [
                        -0.45,
                        -0.3375,
                        -0.225,
                        -0.1125,
                        0.0,
                        0.1125,
                        0.225,
                        0.3375,
                        0.45,
                    ],
                    # Closed-loop only: 1-step OBB at dt=0.5 misses closing pairs.
                    # Calibration leaves these unset (defaults to 1 x dt).
                    "conflict_substeps": 4,
                    "conflict_horizon": 1.5,
                    "obb_safety_filter": True,
                    "steering_penalty_weight": 0.5,
                    "vehicle_length": self.vehicle_length,
                    "vehicle_width": self.vehicle_width,
                }
            )
        else:
            self.sim_config.setdefault("run_id", self.run_id)
            self.sim_config.setdefault("lane_kf", self.lane_kf)
            self.sim_config.setdefault("path_mode", "polyline")
            self.sim_config.setdefault("utility_frame", "corridor")
        if self.obb_safety_filter is not None:
            self.sim_config["obb_safety_filter"] = bool(self.obb_safety_filter)
        if self.base_params is None:
            self.base_params = dict(DEFAULT_BASE_PARAMS)
        if self.residual_mode not in ("candidate_logits", "param_delta"):
            raise ValueError(f"Unknown residual_mode={self.residual_mode!r}")


class MultiAgentTrafficEnv:
    """Decentralized multi-agent environment on the measured highway corridor."""

    def __init__(self, config: EnvConfig | None = None, seed: int | None = None):
        self.config = config or EnvConfig()
        self.corridor = load_corridor(self.config.run_id, self.config.lane_kf)
        self.rng = np.random.default_rng(seed)
        self.agents: list[TrafficAgent] = []
        self.step_count = 0
        self.collision_count = 0
        self.behavior_reference = None
        if self.config.behavior_coef > 0.0 or self.config.behavior_shaping_coef > 0.0:
            self.behavior_reference = load_behavior_reference(
                self.config.run_id,
                self.config.lane_kf,
                float(self.config.dt),
                self.config.behavior_csv,
            )
        self._behavior_samples: dict[str, list[float]] = {k: [] for k in BEHAVIOR_FEATURES}
        self._behavior_potential: float | None = None

    @property
    def obs_dim(self) -> int:
        return observation_dim(int(self.config.sim_config["max_neighbors"]))

    @property
    def residual_dim(self) -> int:
        if self.config.residual_mode == "candidate_logits":
            from utility_model import n_candidate_actions

            return n_candidate_actions(self.config.sim_config)
        return len(RESIDUAL_PARAM_KEYS)

    def reset(self) -> list[np.ndarray]:
        self.step_count = 0
        self.collision_count = 0
        self._dest_s = []
        self._behavior_samples = {k: [] for k in BEHAVIOR_FEATURES}
        self._behavior_potential = None
        self.agents = self._spawn_agents()
        return [self.get_observation(i) for i in range(len(self.agents))]

    def _spawn_agents(self) -> list[TrafficAgent]:
        agents: list[TrafficAgent] = []
        n = self.config.num_agents
        base_v = self.config.base_desired_speed
        self._dest_s: list[float] = []
        for i in range(n):
            pos, tangent, s0 = self._sample_start_pose(agents)
            speed = max(1.0, self.rng.normal(base_v, 1.2))
            heading = float(np.arctan2(tangent[1], tangent[0]) + self.rng.normal(0.0, 0.05))
            vel = np.array([speed * np.cos(heading), speed * np.sin(heading)], dtype=float)
            # Random destination near the end of the highway corridor.
            end_margin = float(self.rng.uniform(*self.config.dest_s_range_from_end))
            dest_s = max(s0 + 30.0, self.corridor.length - end_margin)
            dest_s = min(dest_s, self.corridor.length - 1.0)
            dest, _ = self.corridor.xy_from_frenet(dest_s, 0.0)
            self._dest_s.append(float(dest_s))
            agents.append(
                TrafficAgent(
                    agent_id=i,
                    pos=np.asarray(pos, dtype=float),
                    vel=vel,
                    dest=np.asarray(dest, dtype=float),
                    desired_speed=float(np.clip(self.rng.normal(base_v, 1.5), 4.0, 14.0)),
                    nominal_y=float(pos[1]),
                    run_id=self.config.run_id,
                    lane_kf=self.config.lane_kf,
                )
            )
        return agents

    def _sample_start_pose(
        self, existing_agents: list[TrafficAgent]
    ) -> tuple[np.ndarray, np.ndarray, float]:
        """Sample a start pose inside the corridor with minimum spacing."""
        s_lo, s_hi = self.config.spawn_s_range
        s_hi = min(s_hi, max(s_lo + 1.0, self.corridor.length * 0.35))
        margin = 0.5 * float(self.config.sim_config.get("vehicle_width", self.config.vehicle_width))
        for _ in range(300):
            s = float(self.rng.uniform(s_lo, s_hi))
            # Probe local half-width
            mid, tangent = self.corridor.xy_from_frenet(s, 0.0)
            c_lo, c_hi, _ = self.corridor.clearances(mid)
            half = 0.5 * (c_lo + c_hi)
            max_lat = max(0.0, self.config.spawn_lateral_frac * half - margin)
            lateral = float(self.rng.uniform(-max_lat, max_lat)) if max_lat > 0 else 0.0
            pos, tangent = self.corridor.xy_from_frenet(s, lateral)
            if not self.corridor.inside(pos, margin=margin):
                continue
            if all(
                np.linalg.norm(pos - agent.pos) >= self.config.min_initial_spacing
                for agent in existing_agents
            ):
                return pos, tangent, s
        pos, tangent = self.corridor.xy_from_frenet(float(self.rng.uniform(s_lo, s_hi)), 0.0)
        return pos, tangent, float(self.corridor.project(pos)[0])

    def get_neighbors(self, agent_idx: int) -> list[int]:
        ego = self.agents[agent_idx]
        rp = self.config.sim_config["perception_radius"]
        neighbors: list[tuple[float, int]] = []
        for j, other in enumerate(self.agents):
            if j == agent_idx or other.reached_destination:
                continue
            d = float(np.linalg.norm(other.pos - ego.pos))
            if d <= rp:
                neighbors.append((d, j))
        neighbors.sort(key=lambda x: x[0])
        max_n = self.config.sim_config["max_neighbors"]
        return [j for _, j in neighbors[:max_n]]

    def get_observation(self, agent_idx: int) -> np.ndarray:
        """Frenet ego state + remaining station + body-frame neighbors."""
        dest_s = (
            self._dest_s[agent_idx]
            if agent_idx < len(getattr(self, "_dest_s", []))
            else None
        )
        return local_observation(
            self.agents[agent_idx],
            self.agents,
            self.get_neighbors(agent_idx),
            self.corridor,
            int(self.config.sim_config["max_neighbors"]),
            dest_s=dest_s,
        )

    def _remaining_station(self, agent_idx: int) -> float:
        ego = self.agents[agent_idx]
        s, _, _, _, _ = self.corridor.project(ego.pos)
        dest_s = (
            self._dest_s[agent_idx]
            if agent_idx < len(getattr(self, "_dest_s", []))
            else float(self.corridor.project(ego.dest)[0])
        )
        return max(float(dest_s) - float(s), 0.0)

    def step(
        self,
        residual_actions: list[Any] | None = None,
    ) -> tuple[list[np.ndarray], list[float], bool, dict[str, Any]]:
        if residual_actions is None:
            residual_actions = [None for _ in self.agents]

        if len(residual_actions) != len(self.agents):
            raise ValueError("Expected one residual action per agent")
        controls = []
        flipped = considered = 0
        for i, agent in enumerate(self.agents):
            if agent.reached_destination:
                controls.append((0.0, 0.0))
                continue
            action = residual_actions[i]
            residual = None
            params = self.config.base_params
            if self.config.residual_mode == "candidate_logits":
                if action is not None:
                    residual = np.asarray(action, dtype=float)
            else:
                if action is not None and not isinstance(action, dict):
                    raise ValueError("param_delta actions must be parameter dictionaries")
                params = apply_residual(params, action)
            chosen, idx, prior_idx = select_candidate_with_logit_residual(
                i, agent, self.agents, params, self.config.sim_config, residual)
            considered += 1
            flipped += int(idx != prior_idx)
            controls.append((float(chosen["accel_longitudinal"]), float(chosen["steering_angle"])))

        transition = advance_agents(
            self.agents, controls, self.corridor, self.config.sim_config, self._dest_s,
            reward_weights=self.config.reward_weights, leftover_coef=self.config.leftover_coef,
            arrival_bonus=self.config.arrival_bonus, collision_penalty=self.config.collision_penalty,
        )
        rewards = transition.rewards
        selected_controls = [{"accel": a, "steering": d} for a, d in transition.controls]
        hit = transition.colliding_agents
        self.collision_count += len(transition.collision_pairs)
        self.step_count += 1
        if self.behavior_reference is not None:
            for i, move in enumerate(transition.candidates):
                if move is None or self.agents[i].reached_destination:
                    continue
                lateral = float(self.corridor.project(self.agents[i].pos)[1])
                # Use realized acceleration, including speed-cap saturation.
                accel = float(move["realized_accel"])
                speed = float(move["speed"])
                for key, value in (("speed", speed), ("accel", accel), ("lateral", lateral)):
                    self._behavior_samples[key].append(value)
                rewards[i] -= self.config.behavior_coef * self.behavior_reference.step_deviation(speed, accel, lateral)

        observations = [self.get_observation(i) for i in range(len(self.agents))]
        truncated = self.step_count >= self.config.max_steps
        all_arrived = all(a.reached_destination for a in self.agents)
        done = truncated or all_arrived

        realism_distance = float("nan")
        if self.behavior_reference is not None:
            realism_distance = self.behavior_distance()
            if (
                self.config.behavior_shaping_coef > 0.0
                and self.step_count > self.config.behavior_warmup_steps
            ):
                previous = self._behavior_potential
                if previous is not None:
                    shared = self.config.behavior_shaping_coef * (realism_distance - previous)
                    rewards = [r - shared for r in rewards]
            self._behavior_potential = realism_distance

        info = {
            "collision_count": self.collision_count,
            "steps": self.step_count,
            "destinations_reached": sum(a.reached_destination for a in self.agents),
            "selected_controls": selected_controls,
            "run_id": self.config.run_id,
            "lane_kf": self.config.lane_kf,
            "colliding_agents": sorted(hit),
            "realism_distance": realism_distance,
            "truncated": bool(truncated and not all_arrived),
            "control_flips": int(flipped),
            "control_decisions": int(considered),
            "control_flip_rate": float(flipped / considered) if considered else 0.0,
        }
        return observations, rewards, done, info

    def behavior_distance(self) -> float:
        """Normalized W1 between this episode's marginals and the measured ones."""
        if self.behavior_reference is None:
            return float("nan")
        samples = {k: np.asarray(v, dtype=float) for k, v in self._behavior_samples.items()}
        return self.behavior_reference.distribution_distance(samples)

    def _update_destination_flag(self, agent_idx: int) -> None:
        """
        Mark arrival by corridor progress: once along-track s reaches dest_s,
        stop the agent. Euclidean 1 m checks fail when cars are laterally offset
        from the centerline destination star.
        """
        agent = self.agents[agent_idx]
        if agent.reached_destination:
            agent.vel[:] = 0.0
            return
        s, _, _, _, _ = self.corridor.project(agent.pos)
        dest_s = self._dest_s[agent_idx] if agent_idx < len(getattr(self, "_dest_s", [])) else None
        if dest_s is None:
            dest_s = float(self.corridor.project(agent.dest)[0])
        tol = float(self.config.sim_config.get("destination_threshold", 1.0))
        # Arrive when we reach/pass the destination station (with small tolerance).
        if s >= dest_s - tol:
            agent.reached_destination = True
            agent.vel[:] = 0.0
            agent.prev_control = {"accel": 0.0, "steering": 0.0}

    def _check_collisions(self) -> set[int]:
        """Count pairwise OBB overlaps; return the set of agents involved this step."""
        sim = self.config.sim_config
        length = float(sim.get("vehicle_length", self.config.vehicle_length))
        width = float(sim.get("vehicle_width", self.config.vehicle_width))
        use_obb = bool(sim.get("use_obb_collisions", True))
        threshold = float(sim.get("collision_threshold", 1.5))
        colliding: set[int] = set()
        for i in range(len(self.agents)):
            for j in range(i + 1, len(self.agents)):
                if self.agents[i].reached_destination or self.agents[j].reached_destination:
                    continue
                if use_obb:
                    hit = boxes_overlap(
                        self.agents[i].pos,
                        self.agents[i].heading,
                        self.agents[j].pos,
                        self.agents[j].heading,
                        length=length,
                        width=width,
                    )
                else:
                    hit = np.linalg.norm(self.agents[i].pos - self.agents[j].pos) < threshold
                if hit:
                    self.collision_count += 1
                    colliding.update((i, j))
        return colliding

    def rollout_metric(self) -> float:
        total_dist = sum(
            float(np.linalg.norm(a.pos - a.dest))
            for a in self.agents
            if not a.reached_destination
        )
        penalty = 10.0 * self.collision_count
        return total_dist + penalty
