"""Common rollout loop: one integrator, one collision test, one arrival rule."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

import Baselines._paths  # noqa: F401
from Baselines.controllers import Controller
from Baselines.dynamics import project_and_clearances
from Baselines.scenario import Scenario
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
        from RL.routing import agent_route, agent_station
        s, lateral, _, c_lo, c_hi = project_and_clearances(agent_route(scenario.corridor, a), a.pos)
        sta[i] = agent_station(scenario.corridor, a)
        lat[i] = lateral
        from RL.boundary import footprint_clearance
        clr[i] = footprint_clearance(scenario.corridor, a.pos, a.heading,
                                     scenario.vehicle_length, scenario.vehicle_width)
    return pos, head, spd, lat, clr, sta


class RolloutRecorder:
    """One trace format for training validation and the standalone benchmark."""

    def __init__(self, scenario: Scenario, agents: list[TrafficAgent], model: str):
        import time
        self.started = time.perf_counter()
        self.scenario, self.model = scenario, model
        self.states = [_agent_state(agents, scenario)]
        self.actives = [np.array([not a.reached_destination for a in agents])]
        self.controls = []
        self.proposed_controls = []
        self.interventions = self.decisions = 0
        self.arrival_step = np.full(len(agents), -1, dtype=int)
        self.collision_steps = self.collision_events = self.offroad_steps = 0
        self.colliding_agents, self.offroad_agents, self.active_pairs = set(), set(), set()

    def record(self, agents, controls, collision_pairs, proposed_controls=None):
        proposed = np.asarray(controls if proposed_controls is None else proposed_controls, dtype=float)
        self.proposed_controls.append(proposed)
        self.decisions += int(self.actives[-1].sum())
        changed = np.any(np.abs(proposed-np.asarray(controls)) > 1e-8, axis=-1)
        self.interventions += int((changed & self.actives[-1]).sum())
        self.controls.append(np.asarray(controls, dtype=float))
        self.states.append(_agent_state(agents, self.scenario))
        active = np.array([not a.reached_destination for a in agents])
        active_before = self.actives[-1]
        self.arrival_step[active_before & ~active] = len(self.controls)
        self.actives.append(active)
        self.collision_events += len(collision_pairs - self.active_pairs)
        self.collision_steps += len(collision_pairs)
        self.colliding_agents.update(i for pair in collision_pairs for i in pair)
        self.active_pairs = set(collision_pairs)
        offroad = set(np.flatnonzero(active_before & (self.states[-1][4] < 0.0)))
        self.offroad_steps += len(offroad)
        self.offroad_agents.update(offroad)

    def result(self, *, extra=None):
        import time
        scenario = self.scenario
        n = scenario.num_agents
        positions, headings, speeds, lateral, clearance, station = (
            np.asarray(values) for values in zip(*self.states))
        controls = np.asarray(self.controls) if self.controls else np.zeros((0, n, 2))
        return RolloutResult(
            model=self.model, seed=scenario.seed, run_id=scenario.run_id,
            lane_kf=scenario.lane_kf, dt=scenario.dt,
            vehicle_length=scenario.vehicle_length, vehicle_width=scenario.vehicle_width,
            steps=len(self.controls), num_agents=n,
            positions=positions, headings=headings, speeds=speeds,
            accels=controls[:, :, 0], steerings=controls[:, :, 1],
            active=np.asarray(self.actives), lateral=lateral, clearance=clearance,
            station=station, collision_steps=self.collision_steps,
            collision_events=self.collision_events, colliding_agents=self.colliding_agents,
            offroad_steps=self.offroad_steps, offroad_agents=self.offroad_agents,
            arrival_step=self.arrival_step,
            dest_s=np.array([a.dest_s for a in scenario.agents]),
            start_s=np.array([a.start_s for a in scenario.agents]),
            wall_time=time.perf_counter() - self.started,
            extra={"spawn_protocol_version": scenario.sim_config.get("spawn_protocol_version", 1),
                   "decision_protocol_version": scenario.sim_config.get("decision_protocol_version", 1),
                   "shield_interventions": self.interventions,
                   "shield_decisions": self.decisions,
                   "shield_intervention_rate": self.interventions/max(1, self.decisions),
                   "proposed_controls": np.asarray(self.proposed_controls).tolist(),
                   "collision_filter_revision": scenario.sim_config.get("collision_filter_revision", 1),
                   **(extra or {})},
        )


def rollout(
    scenario: Scenario,
    controller: Controller,
    stop_when_all_arrived: bool = True,
) -> RolloutResult:
    """Simulate one scenario under one controller."""
    agents = scenario.spawn_agents()
    controller.reset(scenario)
    recorder = RolloutRecorder(scenario, agents, getattr(controller, "name", controller.__class__.__name__))
    dest_s = np.array([a.dest_s for a in scenario.agents], dtype=float)
    for step in range(scenario.max_steps):
        controls = controller.compute_controls(agents, scenario, step)
        transition = advance_agents(agents, controls, scenario.corridor, scenario.sim_config, dest_s)
        recorder.record(agents, transition.controls, transition.collision_pairs, proposed_controls=controls)
        if stop_when_all_arrived and all(a.reached_destination for a in agents):
            break
    return recorder.result(extra={
        "selection_status": getattr(getattr(controller, "policy", None), "selection_status", "not_applicable"),
        "training_revision": getattr(getattr(controller, "policy", None), "training_revision", None),
    })
