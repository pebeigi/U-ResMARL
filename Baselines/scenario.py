"""Shared benchmark scenarios.

Every model is evaluated on byte-identical initial conditions: the scenario is
generated once from the RL environment's spawn logic for a given seed, then
replayed for each controller.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

import Baselines._paths  # noqa: F401
from RL.corridor import DEFAULT_LANE_KF, DEFAULT_RUN_ID, HighwayCorridor, load_corridor
from RL.traffic_env import EnvConfig, MultiAgentTrafficEnv
from utility_model import TrafficAgent


@dataclass
class AgentInit:
    agent_id: int
    pos: np.ndarray
    vel: np.ndarray
    heading: float
    dest: np.ndarray
    dest_s: float
    start_s: float
    desired_speed: float


@dataclass
class Scenario:
    seed: int
    run_id: int
    lane_kf: int
    dt: float
    max_steps: int
    sim_config: dict[str, Any]
    agents: list[AgentInit]
    corridor: HighwayCorridor = field(repr=False)

    @property
    def num_agents(self) -> int:
        return len(self.agents)

    @property
    def vehicle_length(self) -> float:
        return float(self.sim_config.get("vehicle_length", 4.5))

    @property
    def vehicle_width(self) -> float:
        return float(self.sim_config.get("vehicle_width", 1.8))

    def spawn_agents(self) -> list[TrafficAgent]:
        """Fresh mutable agent state for one rollout."""
        return [
            TrafficAgent(
                agent_id=a.agent_id,
                pos=a.pos.copy(),
                vel=a.vel.copy(),
                dest=a.dest.copy(),
                desired_speed=a.desired_speed,
                nominal_y=float(a.pos[1]),
                run_id=self.run_id,
                lane_kf=self.lane_kf,
                heading_angle=float(a.heading),
            )
            for a in self.agents
        ]


def build_scenario(
    seed: int,
    num_agents: int = 10,
    max_steps: int = 240,
    dt: float = 0.5,
    run_id: int = DEFAULT_RUN_ID,
    lane_kf: int = DEFAULT_LANE_KF,
    base_desired_speed: float = 8.0,
    spawn_s_range: tuple[float, float] | None = None,
    spawn_lateral_frac: float | None = None,
    min_initial_spacing: float | None = None,
    obb_safety_filter: bool | None = None,
    conflict_lookahead: str = "full",
) -> Scenario:
    """Sample one benchmark scenario using the RL environment's spawn rules."""
    cfg_kwargs: dict[str, Any] = {
        "dt": dt,
        "max_steps": max_steps,
        "num_agents": num_agents,
        "base_desired_speed": base_desired_speed,
        "run_id": run_id,
        "lane_kf": lane_kf,
    }
    if spawn_s_range is not None:
        cfg_kwargs["spawn_s_range"] = spawn_s_range
    if spawn_lateral_frac is not None:
        cfg_kwargs["spawn_lateral_frac"] = spawn_lateral_frac
    if min_initial_spacing is not None:
        cfg_kwargs["min_initial_spacing"] = min_initial_spacing
    cfg = EnvConfig(**cfg_kwargs)
    env = MultiAgentTrafficEnv(cfg, seed=seed)
    env.reset()

    scenario = scenario_from_env(env, seed)
    scenario.sim_config = _finalize_sim_config(
        scenario.sim_config, obb_safety_filter, conflict_lookahead, dt)
    return scenario


def scenario_from_env(env: MultiAgentTrafficEnv, seed: int) -> Scenario:
    """Snapshot an already-reset environment for the shared rollout recorder."""
    cfg, corridor = env.config, env.corridor
    from RL.routing import agent_station
    agents = [AgentInit(
        agent_id=a.agent_id, pos=a.pos.copy(), vel=a.vel.copy(), heading=float(a.heading),
        dest=a.dest.copy(), dest_s=float(env._dest_s[i]),
        start_s=agent_station(corridor, a), desired_speed=float(a.desired_speed),
    ) for i, a in enumerate(env.agents)]
    return Scenario(seed=seed, run_id=cfg.run_id, lane_kf=cfg.lane_kf, dt=cfg.dt,
                    max_steps=cfg.max_steps, sim_config=dict(cfg.sim_config),
                    agents=agents, corridor=corridor)


def _finalize_sim_config(
    sim_config: dict[str, Any],
    obb_safety_filter: bool | None,
    conflict_lookahead: str,
    dt: float,
) -> dict[str, Any]:
    """Attach benchmark-wide safety settings shared by every controller."""
    sim_config.setdefault("obb_safety_filter", True)
    if obb_safety_filter is not None:
        sim_config["obb_safety_filter"] = bool(obb_safety_filter)
    if conflict_lookahead == "none":
        sim_config["conflict_horizon"] = float(dt)
        sim_config["conflict_substeps"] = 1
    else:
        sim_config.setdefault("conflict_horizon", 1.5)
        sim_config.setdefault("conflict_substeps", 4)
    return sim_config


def build_scenarios(
    seeds: list[int],
    **kwargs: Any,
) -> list[Scenario]:
    return [build_scenario(seed=s, **kwargs) for s in seeds]
