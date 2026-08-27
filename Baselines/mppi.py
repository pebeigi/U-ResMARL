"""Model Predictive Path Integral control baseline.

Reference: Williams, Aldrich & Theodorou, "Model Predictive Path Integral
Control: From Theory to Parallel Computation", JGCD 2017; and "Aggressive
driving with model predictive path integral control", ICRA 2016.

Each agent keeps a nominal control sequence, perturbs it with Gaussian noise,
rolls the perturbations out through the true bicycle model against
constant-velocity predictions of its neighbours, and updates the nominal
sequence with the exponentially weighted average of the samples.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

import Baselines._paths  # noqa: F401
from Baselines.controllers import BaseController
from Baselines.dynamics import MAX_STEERING, control_obb_conflict, simulate_bicycle_batch, sanitize_control
from Baselines.local_frame import build_local_frame, frenet_conflict, predict_neighbours
from utility_model import TrafficAgent

if TYPE_CHECKING:  # pragma: no cover
    from Baselines.scenario import Scenario


class MPPIController(BaseController):
    """Sampling-based MPC with a path-integral update of the control sequence."""

    name = "mppi"

    def __init__(
        self,
        horizon: int = 10,
        samples: int = 256,
        temperature: float = 1.0,
        accel_std: float = 1.0,
        steering_std: float = 0.06,
        w_progress: float = 1.0,
        w_speed: float = 0.6,
        w_collision: float = 400.0,
        w_boundary: float = 200.0,
        w_control: float = 0.5,
        w_steering: float = 40.0,
        seed: int = 0,
        name: str | None = None,
    ):
        self.horizon = int(horizon)
        self.samples = int(samples)
        self.temperature = float(temperature)
        self.noise_std = np.array([float(accel_std), float(steering_std)], dtype=float)
        self.w_progress = float(w_progress)
        self.w_speed = float(w_speed)
        self.w_collision = float(w_collision)
        self.w_boundary = float(w_boundary)
        self.w_control = float(w_control)
        self.w_steering = float(w_steering)
        self.rng = np.random.default_rng(seed)
        if name:
            self.name = name
        self._radius = 2.4
        self._dest_s: np.ndarray = np.array([])
        self._nominal: dict[int, np.ndarray] = {}

    def reset(self, scenario: "Scenario") -> None:
        self._radius = 0.5 * float(np.hypot(scenario.vehicle_length, scenario.vehicle_width))
        self._dest_s = np.array([a.dest_s for a in scenario.agents], dtype=float)
        self._nominal = {a.agent_id: np.zeros((self.horizon, 2)) for a in scenario.agents}

    def _rollout_cost(
        self,
        traj: np.ndarray,
        speeds: np.ndarray,
        frame,
        predictions: np.ndarray,
        desired_speed: float,
        dest_s: float,
        length: float,
        width: float,
    ) -> np.ndarray:
        k = traj.shape[0]
        station, lateral, clearance = frame.project_many(traj.reshape(-1, 2))
        station = station.reshape(k, -1)
        lateral = lateral.reshape(k, -1)
        clearance = clearance.reshape(k, -1)

        # Reward along-corridor progress, capped at the destination station.
        reached = np.minimum(station, dest_s)
        progress = reached[:, -1] - reached[:, 0]
        cost = -self.w_progress * progress

        cost += self.w_speed * np.sum((speeds - desired_speed) ** 2, axis=1)

        violation = np.maximum(0.0, 0.5 * width - clearance)
        cost += self.w_boundary * np.sum(violation**2, axis=1)

        if predictions.shape[0]:
            shape = predictions.shape
            n_station, n_lateral, _ = frame.project_many(predictions.reshape(-1, 2))
            _, overlap = frenet_conflict(
                station,
                lateral,
                n_station.reshape(shape[0], shape[1]),
                n_lateral.reshape(shape[0], shape[1]),
                length,
                width,
            )
            cost += self.w_collision * np.sum(overlap, axis=(1, 2))
        return cost

    def compute_controls(
        self,
        agents: list[TrafficAgent],
        scenario: "Scenario",
        step: int,
    ) -> list[tuple[float, float]]:
        dt = float(scenario.dt)
        max_accel = float(scenario.sim_config.get("max_accel", 4.0))
        controls: list[tuple[float, float]] = []

        for i, agent in enumerate(agents):
            if agent.reached_destination:
                controls.append((0.0, 0.0))
                continue

            nominal = self._nominal.setdefault(agent.agent_id, np.zeros((self.horizon, 2)))
            noise = self.rng.normal(0.0, 1.0, size=(self.samples, self.horizon, 2)) * self.noise_std
            candidates = nominal[None, :, :] + noise
            candidates[:, :, 0] = np.clip(candidates[:, :, 0], -max_accel, max_accel)
            candidates[:, :, 1] = np.clip(candidates[:, :, 1], -MAX_STEERING, MAX_STEERING)

            traj, speeds, _ = simulate_bicycle_batch(
                agent.pos,
                float(agent.heading),
                float(agent.speed),
                candidates[:, :, 0],
                candidates[:, :, 1],
                scenario,
            )

            s_now, _, _, _, _ = scenario.corridor.project(agent.pos)
            reach = float(agent.speed) * dt * self.horizon + 40.0
            frame = build_local_frame(scenario.corridor, float(s_now), ahead=reach)
            predictions = predict_neighbours(agents, i, self.horizon, dt)

            cost = self._rollout_cost(
                traj,
                speeds,
                frame,
                predictions,
                float(agent.desired_speed),
                float(self._dest_s[i]),
                scenario.vehicle_length,
                scenario.vehicle_width,
            )
            cost += self.w_control * np.sum(candidates[:, :, 0] ** 2, axis=1)
            cost += self.w_steering * np.sum(candidates[:, :, 1] ** 2, axis=1)

            weights = np.exp(-(cost - cost.min()) / max(self.temperature, 1e-6))
            weights /= max(float(weights.sum()), 1e-12)
            nominal = nominal + np.einsum("k,khc->hc", weights, noise)
            nominal[:, 0] = np.clip(nominal[:, 0], -max_accel, max_accel)
            nominal[:, 1] = np.clip(nominal[:, 1], -MAX_STEERING, MAX_STEERING)

            accel, steering = float(nominal[0, 0]), float(nominal[0, 1])
            # Receding horizon: shift the sequence and repeat the last command.
            self._nominal[agent.agent_id] = np.vstack([nominal[1:], nominal[-1:]])

            remaining = float(self._dest_s[i]) - float(s_now)
            if remaining < max(agent.speed * dt * 2.0, 3.0):
                accel = float(np.clip(-agent.speed / dt, -max_accel, 0.0))
                steering = 0.0
            controls.append(sanitize_control(i, agent, agents, (accel, steering), scenario))
        return controls
