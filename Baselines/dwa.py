"""Dynamic Window Approach baseline.

Reference: Fox, Burgard & Thrun, "The Dynamic Window Approach to Collision
Avoidance", IEEE Robotics & Automation Magazine 4(1), 1997.

Commands reachable within one step given the acceleration and steering-rate
limits are enumerated, each is forward-simulated with the true bicycle model
against constant-velocity predictions of the neighbours, and the classical
heading / clearance / velocity objective picks the winner.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

import Baselines._paths  # noqa: F401
from Baselines.controllers import BaseController
from Baselines.dynamics import MAX_STEERING, control_obb_conflict, simulate_bicycle_batch
from Baselines.local_frame import build_local_frame, frenet_conflict, predict_neighbours
from utility_model import TrafficAgent

if TYPE_CHECKING:  # pragma: no cover
    from Baselines.scenario import Scenario


def _normalise(values: np.ndarray) -> np.ndarray:
    total = float(np.sum(values))
    if total <= 1e-9:
        return np.zeros_like(values)
    return values / total


class DWAController(BaseController):
    """Sample the dynamic window, score by heading / clearance / velocity."""

    name = "dwa"

    def __init__(
        self,
        horizon: int = 6,
        accel_samples: int = 7,
        steering_samples: int = 11,
        steering_rate: float = 0.6,  # rad/s
        accel_rate: float = 4.0,  # m/s^3
        heading_weight: float = 1.0,
        clearance_weight: float = 0.8,
        velocity_weight: float = 1.5,
        clearance_cap: float = 12.0,
        lookahead: float = 20.0,
        name: str | None = None,
    ):
        self.horizon = int(horizon)
        self.accel_samples = int(accel_samples)
        self.steering_samples = int(steering_samples)
        self.steering_rate = float(steering_rate)
        self.accel_rate = float(accel_rate)
        self.heading_weight = float(heading_weight)
        self.clearance_weight = float(clearance_weight)
        self.velocity_weight = float(velocity_weight)
        self.clearance_cap = float(clearance_cap)
        self.lookahead = float(lookahead)
        if name:
            self.name = name
        self._radius = 2.4
        self._dest_s: np.ndarray = np.array([])

    def reset(self, scenario: "Scenario") -> None:
        self._radius = 0.5 * float(np.hypot(scenario.vehicle_length, scenario.vehicle_width))
        self._dest_s = np.array([a.dest_s for a in scenario.agents], dtype=float)

    def _window(self, agent: TrafficAgent, scenario: "Scenario") -> tuple[np.ndarray, np.ndarray]:
        dt = float(scenario.dt)
        max_accel = float(scenario.sim_config.get("max_accel", 4.0))
        prev_steering = float(agent.prev_control.get("steering", 0.0))
        prev_accel = float(agent.prev_control.get("accel", 0.0))

        # The dynamic window is the set reachable under the actuator rate limits.
        accel_lo = max(-max_accel, prev_accel - self.accel_rate * dt)
        accel_hi = min(max_accel, prev_accel + self.accel_rate * dt)
        steer_lo = max(-MAX_STEERING, prev_steering - self.steering_rate * dt)
        steer_hi = min(MAX_STEERING, prev_steering + self.steering_rate * dt)

        accels = np.linspace(accel_lo, accel_hi, self.accel_samples)
        steerings = np.linspace(steer_lo, steer_hi, self.steering_samples)
        grid_a, grid_s = np.meshgrid(accels, steerings, indexing="ij")
        return grid_a.ravel(), grid_s.ravel()

    def compute_controls(
        self,
        agents: list[TrafficAgent],
        scenario: "Scenario",
        step: int,
    ) -> list[tuple[float, float]]:
        dt = float(scenario.dt)
        controls: list[tuple[float, float]] = []

        for i, agent in enumerate(agents):
            if agent.reached_destination:
                controls.append((0.0, 0.0))
                continue

            accels, steerings = self._window(agent, scenario)
            k = accels.shape[0]
            traj, speeds, headings = simulate_bicycle_batch(
                agent.pos,
                float(agent.heading),
                float(agent.speed),
                np.repeat(accels[:, None], self.horizon, axis=1),
                np.repeat(steerings[:, None], self.horizon, axis=1),
                scenario,
            )

            s_now, _, _, _, _ = scenario.corridor.project(agent.pos)
            frame = build_local_frame(scenario.corridor, float(s_now), ahead=self.lookahead + 60.0)
            station, lateral, clearance = frame.project_many(traj.reshape(-1, 2))
            station = station.reshape(k, self.horizon + 1)
            lateral = lateral.reshape(k, self.horizon + 1)
            corridor_clearance = (
                clearance.reshape(k, self.horizon + 1)[:, 1:].min(axis=1) - 0.5 * scenario.vehicle_width
            )

            predictions = predict_neighbours(agents, i, self.horizon, dt)
            if predictions.shape[0]:
                shape = predictions.shape
                n_station, n_lateral, _ = frame.project_many(predictions.reshape(-1, 2))
                separation, _ = frenet_conflict(
                    station,
                    lateral,
                    n_station.reshape(shape[0], shape[1]),
                    n_lateral.reshape(shape[0], shape[1]),
                    scenario.vehicle_length,
                    scenario.vehicle_width,
                )
                obstacle_clearance = separation.min(axis=(1, 2))
            else:
                obstacle_clearance = np.full(k, self.clearance_cap)

            clearance_score = np.minimum(
                np.minimum(corridor_clearance, obstacle_clearance), self.clearance_cap
            )

            # Heading toward a carrot point on the corridor.
            target_s = min(float(s_now) + self.lookahead, float(self._dest_s[i]))
            target = frame.point_at(np.array([target_s]), np.array([0.0]))[0]
            to_target = target - traj[:, -1, :]
            desired = np.arctan2(to_target[:, 1], to_target[:, 0])
            heading_error = np.abs(
                np.arctan2(np.sin(desired - headings[:, -1]), np.cos(desired - headings[:, -1]))
            )
            heading_score = np.pi - heading_error

            admissible = clearance_score > 0.0
            obb_ok = np.array(
                [
                    not control_obb_conflict(i, agent, agents, float(a), float(s), scenario)
                    for a, s in zip(accels, steerings)
                ]
            )
            admissible = admissible & obb_ok
            if not np.any(admissible):
                # Everything is unsafe: fall back to the safest command.
                best = int(np.argmax(clearance_score))
                controls.append((float(accels[best]), float(steerings[best])))
                continue

            velocity_score = speeds[:, -1]
            score = (
                self.heading_weight * _normalise(np.where(admissible, heading_score, 0.0))
                + self.clearance_weight * _normalise(np.where(admissible, clearance_score, 0.0))
                + self.velocity_weight * _normalise(np.where(admissible, velocity_score, 0.0))
            )
            score[~admissible] = -np.inf

            # Stop at the destination station.
            remaining = float(self._dest_s[i]) - float(s_now)
            if remaining < max(agent.speed * dt * 2.0, 3.0):
                controls.append((float(np.clip(-agent.speed / dt, -4.0, 0.0)), 0.0))
                continue

            best = int(np.argmax(score))
            controls.append((float(accels[best]), float(steerings[best])))
        return controls
