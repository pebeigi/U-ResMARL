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
    boxes_overlap,
    corridor_sim_defaults,
    load_corridor,
)
from RL.behavior_reference import FEATURES as BEHAVIOR_FEATURES
from RL.behavior_reference import load_behavior_reference
from RL.obs import local_observation, observation_dim
from RL.transition import advance_agents, DEFAULT_REWARD_WEIGHTS, DRIVING_REWARD_REVISION
from RL.decision import select_candidate_with_logit_residual
from utility_model import (
    DEFAULT_BASE_PARAMS,
    DEFAULT_SIM_CONFIG,
    TrafficAgent,
    sanitize_control_command,
)


SPAWN_PROTOCOL_VERSION = 3


class SpawnPackingError(ValueError):
    """The requested spawn layout could not be packed without overlaps."""


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
    # Explicit OBB collision penalties. Soft proximity alone is too weak for
    # residual learning to care about vehicle-scale overlaps. Duration is off
    # by default (CaRL); event penalty is the terminal contact cost.
    collision_penalty: float = 0.0
    collision_event_penalty: float = 1.0
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
    # candidate_logits: add to discrete utilities before argmax.
    # param_delta: edit utility parameters (ablation).
    residual_mode: str = "candidate_logits"
    # Terms that align the dense reward with leftover-distance / arrival metrics.
    leftover_coef: float = 0.08
    arrival_bonus: float = 8.0
    # When set, overrides ``sim_config["obb_safety_filter"]``.
    obb_safety_filter: bool | None = None
    # PDM-Closed: at inference, execute a residual candidate only if its local
    # closed-loop score strictly beats the frozen utility proposal. Training
    # keeps sampled actions so PPO stays on-policy.
    accept_residual_if_better: bool = False

    def __post_init__(self) -> None:
        if self.reward_weights is None:
            self.reward_weights = dict(DEFAULT_REWARD_WEIGHTS)
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
        self.sim_config.setdefault("boundary_safety_filter", True)
        self.sim_config.setdefault("boundary_margin", 0.1)
        self.sim_config["decision_protocol_version"] = 1
        self.sim_config["spawn_protocol_version"] = SPAWN_PROTOCOL_VERSION
        self.sim_config["collision_filter_revision"] = 2
        self.sim_config["driving_reward_revision"] = DRIVING_REWARD_REVISION
        self.sim_config["leftover_coef"] = float(self.leftover_coef)
        self.sim_config["arrival_bonus"] = float(self.arrival_bonus)
        self.sim_config["collision_event_penalty"] = float(self.collision_event_penalty)
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
        self._candidate_cache = {}
        self.step_count = 0
        self.collision_count = 0
        self.collision_events = 0
        self._collision_pairs: set[tuple[int, int]] = set()
        self.collided_agents: set[int] = set()
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
        self._candidate_cache = {}
        self.step_count = 0
        self.collision_count = 0
        self.collision_events = 0
        self._collision_pairs = set()
        self.collided_agents = set()
        self._dest_s = []
        self._behavior_samples = {k: [] for k in BEHAVIOR_FEATURES}
        self._behavior_potential = None
        self.agents = self._spawn_agents()
        return [self.get_observation(i) for i in range(len(self.agents))]

    def _spawn_agents(self) -> list[TrafficAgent]:
        # Random sequential packing can get stuck even when a valid layout fits.
        # Restart the layout instead of silently dropping spacing constraints.
        for attempt in range(64):
            self._spawn_stations = None
            if attempt >= 8 and self.config.num_agents > 1:
                lo, hi = self.config.spawn_s_range
                hi = min(hi, max(lo + 1.0, self.corridor.length * .35))
                # A structured proposal can fit near-capacity single-file
                # layouts that random sequential placement rarely discovers.
                # All geometric and spacing checks below still apply.
                if (hi - lo) / (self.config.num_agents - 1) > self.config.min_initial_spacing:
                    stations = np.linspace(lo, hi, self.config.num_agents)
                    self._spawn_stations = self.rng.permutation(stations)
            try:
                return self._spawn_agents_once()
            except SpawnPackingError:
                continue
            finally:
                self._spawn_stations = None
        raise SpawnPackingError("Unable to pack non-overlapping vehicles in the spawn window")

    def _spawn_agents_once(self) -> list[TrafficAgent]:
        from RL.spawn_safety import has_straight_braking_backup
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
            from RL.boundary import footprint_clearance
            agent = agents[-1]
            if footprint_clearance(self.corridor, agent.pos, agent.heading,
                                   self.config.sim_config["vehicle_length"],
                                   self.config.sim_config["vehicle_width"]) < self.config.sim_config["boundary_margin"]:
                # Heading noise must not invalidate a verified spawn footprint.
                agent.heading_angle = float(np.arctan2(tangent[1], tangent[0]))
                agent.vel = speed * tangent
                agent._sync_heading_vector()
            if any(boxes_overlap(agent.pos, agent.heading, other.pos, other.heading,
                                 self.config.sim_config.get("vehicle_length", self.config.vehicle_length),
                                 self.config.sim_config.get("vehicle_width", self.config.vehicle_width))
                   for other in agents[:-1]):
                raise SpawnPackingError("Heading noise caused an initial vehicle overlap")
            # Rejection-sample velocity as well as position. Otherwise a dense
            # layout may look valid while its independently sampled closing
            # speeds make contact unavoidable under the available backup.
            for velocity_attempt in range(32):
                if has_straight_braking_backup(agents, self.config.sim_config):
                    break
                speed = max(1., self.rng.normal(base_v, 1.2))
                agent.vel = speed * np.array([np.cos(agent.heading), np.sin(agent.heading)])
            else:
                raise SpawnPackingError("Unable to sample compatible initial velocities")
        if not has_straight_braking_backup(agents, self.config.sim_config, self.corridor):
            raise SpawnPackingError("Spawn has no collision-free straight-braking backup")
        return agents

    def _sample_start_pose(
        self, existing_agents: list[TrafficAgent]
    ) -> tuple[np.ndarray, np.ndarray, float]:
        """Sample a start pose inside the corridor with minimum spacing."""
        s_lo, s_hi = self.config.spawn_s_range
        s_hi = min(s_hi, max(s_lo + 1.0, self.corridor.length * 0.35))
        margin = 0.5 * float(self.config.sim_config.get("vehicle_width", self.config.vehicle_width))
        for _ in range(300):
            stations = getattr(self, "_spawn_stations", None)
            s = (float(self.rng.uniform(s_lo, s_hi)) if stations is None
                 else float(stations[len(existing_agents)]))
            # Probe local half-width
            mid, tangent = self.corridor.xy_from_frenet(s, 0.0)
            c_lo, c_hi, _ = self.corridor.clearances(mid)
            half = 0.5 * (c_lo + c_hi)
            max_lat = max(0.0, self.config.spawn_lateral_frac * half - margin)
            lateral = float(self.rng.uniform(-max_lat, max_lat)) if max_lat > 0 else 0.0
            pos, tangent = self.corridor.xy_from_frenet(s, lateral)
            if not self.corridor.inside(pos, margin=margin):
                continue
            from RL.boundary import footprint_clearance
            if footprint_clearance(self.corridor, pos, float(np.arctan2(tangent[1], tangent[0])),
                                   self.config.sim_config.get("vehicle_length", self.config.vehicle_length),
                                   self.config.sim_config.get("vehicle_width", self.config.vehicle_width)) < self.config.sim_config["boundary_margin"]:
                continue
            if all(
                np.linalg.norm(pos - agent.pos) >= self.config.min_initial_spacing
                for agent in existing_agents
            ):
                return pos, tangent, s
        raise SpawnPackingError("No road-contained start pose with the requested spacing")

    def get_neighbors(self, agent_idx: int) -> list[int]:
        from RL.decision import neighbor_indices
        return neighbor_indices(self.agents, agent_idx, self.config.sim_config)

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
        from RL.routing import agent_station
        s = agent_station(self.corridor, ego)
        dest_s = (
            self._dest_s[agent_idx]
            if agent_idx < len(getattr(self, "_dest_s", []))
            else float(self.corridor.project(ego.dest)[0])
        )
        return max(float(dest_s) - float(s), 0.0)

    def _keep_residual_candidate(self, agent_idx, residual_candidate, prior_candidate) -> bool:
        if not self.config.accept_residual_if_better:
            return True
        from RL.closed_loop_score import residual_proposal_improves
        return residual_proposal_improves(
            self.agents[agent_idx], residual_candidate, prior_candidate,
            self.agents, agent_idx, self.config.sim_config, self.corridor)

    def candidate_context(self, agent_idx):
        from RL.candidate_policy import candidate_context
        if self.config.residual_mode != "candidate_logits":
            raise ValueError("Discrete candidate context requires candidate_logits mode")
        if agent_idx not in self._candidate_cache:
            self._candidate_cache[agent_idx] = candidate_context(
                agent_idx, self.agents, self.config.base_params, self.config.sim_config)
        return self._candidate_cache[agent_idx]

    def step(
        self,
        residual_actions: list[Any] | None = None,
    ) -> tuple[list[np.ndarray], list[float], bool, dict[str, Any]]:
        if residual_actions is None:
            residual_actions = [None for _ in self.agents]

        if len(residual_actions) != len(self.agents):
            raise ValueError("Expected one residual action per agent")
        controls = []
        prior_controls = []
        from RL.candidate_policy import CandidateIndex
        flipped = considered = 0
        for i, agent in enumerate(self.agents):
            if agent.reached_destination:
                controls.append((0.0, 0.0))
                prior_controls.append((0.0, 0.0))
                continue
            action = residual_actions[i]
            if isinstance(action, CandidateIndex):
                context = self.candidate_context(i)
                idx = action.index
                if idx < 0 or idx >= len(context.mask) or not context.mask[idx]:
                    raise ValueError("Sampled candidate is not eligible in the current state")
                prior = context.candidates[context.prior_index]
                chosen = context.candidates[idx]
                if idx != context.prior_index and not self._keep_residual_candidate(i, chosen, prior):
                    idx = context.prior_index
                    chosen = prior
                considered += 1
                flipped += int(idx != context.prior_index)
                controls.append((float(chosen["accel_longitudinal"]), float(chosen["steering_angle"])))
                prior_controls.append((float(prior["accel_longitudinal"]), float(prior["steering_angle"])))
                continue
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
            if self.config.residual_mode == "candidate_logits":
                context = self.candidate_context(i)
                prior = context.candidates[context.prior_index]
                if idx != prior_idx and not self._keep_residual_candidate(i, chosen, prior):
                    idx = prior_idx
                    chosen = prior
                prior_controls.append((float(prior["accel_longitudinal"]), float(prior["steering_angle"])))
            else:
                prior, _, _ = select_candidate_with_logit_residual(
                    i, agent, self.agents, self.config.base_params, self.config.sim_config, None)
                if idx != prior_idx and not self._keep_residual_candidate(i, chosen, prior):
                    idx = prior_idx
                    chosen = prior
                prior_controls.append((float(prior["accel_longitudinal"]), float(prior["steering_angle"])))
            considered += 1
            flipped += int(idx != prior_idx)
            controls.append((float(chosen["accel_longitudinal"]), float(chosen["steering_angle"])))

        active_before = [not a.reached_destination for a in self.agents]
        prior_executed = [sanitize_control_command(i, a, self.agents, c, self.config.sim_config)
                          if active_before[i] else (0., 0.)
                          for i, (a, c) in enumerate(zip(self.agents, prior_controls))]

        transition = advance_agents(
            self.agents, controls, self.corridor, self.config.sim_config, self._dest_s,
            reward_weights=self.config.reward_weights, leftover_coef=self.config.leftover_coef,
            arrival_bonus=self.config.arrival_bonus, collision_penalty=self.config.collision_penalty,
            collision_event_penalty=self.config.collision_event_penalty,
            previous_collision_pairs=self._collision_pairs,
        )
        self._candidate_cache = {}
        executed_flips = sum(active_before[i] and not np.allclose(c, prior_executed[i], rtol=0., atol=1e-8)
                             for i, c in enumerate(transition.controls))
        rewards = transition.rewards
        selected_controls = [{"accel": a, "steering": d} for a, d in transition.controls]
        hit = transition.colliding_agents
        self.collision_count += len(transition.collision_pairs)
        self.collision_events += len(transition.collision_pairs - self._collision_pairs)
        self._collision_pairs = transition.collision_pairs
        self.collided_agents |= set(transition.colliding_agents)
        self.step_count += 1
        if self.behavior_reference is not None:
            for i, move in enumerate(transition.candidates):
                if move is None or self.agents[i].reached_destination:
                    continue
                from RL.routing import agent_route
                lateral = float(agent_route(self.corridor, self.agents[i]).project(self.agents[i].pos)[1])
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
            "collision_events": self.collision_events,
            "steps": self.step_count,
            "destinations_reached": sum(a.reached_destination for a in self.agents),
            "selected_controls": selected_controls,
            "proposed_controls": controls,
            "run_id": self.config.run_id,
            "lane_kf": self.config.lane_kf,
            "colliding_agents": sorted(hit),
            "realism_distance": realism_distance,
            "truncated": bool(truncated and not all_arrived),
            "candidate_flips": int(flipped),
            "control_flips": int(executed_flips),
            "control_decisions": int(considered),
            "control_flip_rate": float(executed_flips / considered) if considered else 0.0,
        }
        return observations, rewards, done, info

    def behavior_distance(self) -> float:
        """Normalized W1 between this episode's marginals and the measured ones."""
        if self.behavior_reference is None:
            return float("nan")
        samples = {k: np.asarray(v, dtype=float) for k, v in self._behavior_samples.items()}
        return self.behavior_reference.distribution_distance(samples)

    def rollout_metric(self) -> float:
        total_dist = sum(
            float(np.linalg.norm(a.pos - a.dest))
            for a in self.agents
            if not a.reached_destination
        )
        penalty = 10.0 * self.collision_count
        return total_dist + penalty
