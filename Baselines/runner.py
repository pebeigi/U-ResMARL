"""Common rollout loop: one integrator, one collision test, one arrival rule."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

import Baselines._paths  # noqa: F401
from Baselines.controllers import Controller
from Baselines.dynamics import apply_control, hold_still, project_and_clearances, sanitize_control
from Baselines.scenario import Scenario
from RL.corridor import boxes_overlap
from RL.transition import advance_agents
from utility_model import TrafficAgent


@dataclass
class RolloutResult:
    model: str
    seed: int
    run_id: int
    lane_kf: int
    dt: float
    vehicle_length: float
    vehicle_width: float
    steps: int
    num_agents: int
    positions: np.ndarray = field(repr=False)  # (T+1, n, 2)
    headings: np.ndarray = field(repr=False)  # (T+1, n)
    speeds: np.ndarray = field(repr=False)  # (T+1, n)
    accels: np.ndarray = field(repr=False)  # (T, n) longitudinal command
    steerings: np.ndarray = field(repr=False)  # (T, n)
    active: np.ndarray = field(repr=False)  # (T+1, n) bool, still driving
    lateral: np.ndarray = field(repr=False)  # (T+1, n) offset from centreline
    clearance: np.ndarray = field(repr=False)  # (T+1, n) min corridor clearance
    station: np.ndarray = field(repr=False)  # (T+1, n) along-corridor s
    collision_steps: int = 0
    collision_events: int = 0
    colliding_agents: set = field(default_factory=set, repr=False)
    offroad_steps: int = 0
    offroad_agents: set = field(default_factory=set, repr=False)
    arrival_step: np.ndarray = field(default_factory=lambda: np.array([]), repr=False)
    dest_s: np.ndarray = field(default_factory=lambda: np.array([]), repr=False)
    start_s: np.ndarray = field(default_factory=lambda: np.array([]), repr=False)
    wall_time: float = 0.0
    extra: dict[str, Any] = field(default_factory=dict, repr=False)


def _agent_state(
    agents: list[TrafficAgent], scenario: Scenario
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    n = len(agents)
    pos = np.zeros((n, 2))
    head = np.zeros(n)
    spd = np.zeros(n)
    lat = np.zeros(n)
    clr = np.zeros(n)
    sta = np.zeros(n)
    for i, a in enumerate(agents):
        pos[i] = a.pos
        head[i] = a.heading
        spd[i] = a.speed
        s, lateral, _, c_lo, c_hi = project_and_clearances(scenario.corridor, a.pos)
        sta[i] = s
        lat[i] = lateral
        from RL.boundary import footprint_clearance
        clr[i] = footprint_clearance(scenario.corridor, a.pos, a.heading,
                                     scenario.vehicle_length, scenario.vehicle_width)
    return pos, head, spd, lat, clr, sta


def rollout(
    scenario: Scenario,
    controller: Controller,
    stop_when_all_arrived: bool = True,
) -> RolloutResult:
    """Simulate one scenario under one controller."""
    import time

    t0 = time.perf_counter()
    agents = scenario.spawn_agents()
    controller.reset(scenario)

    dest_s = np.array([a.dest_s for a in scenario.agents], dtype=float)
    start_s = np.array([a.start_s for a in scenario.agents], dtype=float)
    tol = float(scenario.sim_config.get("destination_threshold", 1.0))
    length = scenario.vehicle_length
    width = scenario.vehicle_width
    n = len(agents)

    positions, headings, speeds, laterals, clearances, stations = [], [], [], [], [], []
    accels, steerings, actives = [], [], []
    arrival_step = np.full(n, -1, dtype=int)

    collision_steps = 0
    collision_events = 0
    colliding_agents: set[int] = set()
    offroad_steps = 0
    offroad_agents: set[int] = set()
    active_pairs: set[tuple[int, int]] = set()

    p, h, v, lat, clr, sta = _agent_state(agents, scenario)
    positions.append(p)
    headings.append(h)
    speeds.append(v)
    laterals.append(lat)
    clearances.append(clr)
    stations.append(sta)
    actives.append(np.array([not a.reached_destination for a in agents]))

    steps = 0
    for step in range(scenario.max_steps):
        controls = controller.compute_controls(agents, scenario, step)

        transition = advance_agents(agents, controls, scenario.corridor, scenario.sim_config, dest_s)
        step_accel = np.array([c[0] for c in transition.controls])
        step_steer = np.array([c[1] for c in transition.controls])
        for i in transition.arrived:
            arrival_step[i] = step + 1
        current_pairs = transition.collision_pairs
        hit_this_step = transition.colliding_agents
        collision_events += len(current_pairs - active_pairs)
        collision_steps += len(current_pairs)
        colliding_agents.update(hit_this_step)
        active_pairs = current_pairs

        p, h, v, lat, clr, sta = _agent_state(agents, scenario)
        for i in range(n):
            if transition.active_before[i] and clr[i] < 0.0:
                offroad_steps += 1
                offroad_agents.add(i)

        positions.append(p)
        headings.append(h)
        speeds.append(v)
        laterals.append(lat)
        clearances.append(clr)
        stations.append(sta)
        accels.append(step_accel)
        steerings.append(step_steer)
        actives.append(np.array([not a.reached_destination for a in agents]))
        steps = step + 1

        if stop_when_all_arrived and all(a.reached_destination for a in agents):
            break

    return RolloutResult(
        model=getattr(controller, "name", controller.__class__.__name__),
        seed=scenario.seed,
        run_id=scenario.run_id,
        lane_kf=scenario.lane_kf,
        dt=scenario.dt,
        vehicle_length=length,
        vehicle_width=width,
        steps=steps,
        num_agents=n,
        positions=np.asarray(positions),
        headings=np.asarray(headings),
        speeds=np.asarray(speeds),
        accels=np.asarray(accels) if accels else np.zeros((0, n)),
        steerings=np.asarray(steerings) if steerings else np.zeros((0, n)),
        active=np.asarray(actives),
        lateral=np.asarray(laterals),
        clearance=np.asarray(clearances),
        station=np.asarray(stations),
        collision_steps=collision_steps,
        collision_events=collision_events,
        colliding_agents=colliding_agents,
        offroad_steps=offroad_steps,
        offroad_agents=offroad_agents,
        arrival_step=arrival_step,
        dest_s=dest_s,
        start_s=start_s,
        wall_time=time.perf_counter() - t0,
    )
