"""Frenet-frame trajectory generation baseline.

Reference: Werling, Ziegler, Kammel & Thrun, "Optimal trajectory generation for
dynamic street scenarios in a Frenet Frame", ICRA 2010.

The canonical trajectory-generation planner for car-like robots: quintic
polynomials in the lateral coordinate and quartic velocity-keeping polynomials
in the longitudinal coordinate are sampled over terminal offsets, terminal
speeds and horizons; infeasible or colliding candidates are discarded and the
lowest jerk/time/deviation cost survivor is tracked.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

import Baselines._paths  # noqa: F401
from Baselines.controllers import BaseController
from Baselines.dynamics import goal_approach_control
from Baselines.dynamics import velocity_to_control, wrap_angle
from Baselines.local_frame import build_local_frame, frenet_conflict, predict_neighbours
from utility_model import TrafficAgent
from RL.routing import agent_route

if TYPE_CHECKING:  # pragma: no cover
    from Baselines.scenario import Scenario


def _quintic_coefficients(
    start: np.ndarray,  # (n, 3) = d, d', d''
    end_position: np.ndarray,  # (n,)
    horizon: np.ndarray,  # (n,)
) -> np.ndarray:
    """Coefficients a0..a5 of the minimum-jerk lateral polynomial."""
    n = start.shape[0]
    a0, a1, a2 = start[:, 0], start[:, 1], 0.5 * start[:, 2]
    t = horizon
    t2, t3, t4, t5 = t**2, t**3, t**4, t**5

    matrix = np.empty((n, 3, 3))
    matrix[:, 0] = np.stack([t3, t4, t5], axis=1)
    matrix[:, 1] = np.stack([3 * t2, 4 * t3, 5 * t4], axis=1)
    matrix[:, 2] = np.stack([6 * t, 12 * t2, 20 * t3], axis=1)

    rhs = np.stack(
        [
            end_position - (a0 + a1 * t + a2 * t2),
            -(a1 + 2 * a2 * t),
            -2 * a2,
        ],
        axis=1,
    )
    tail = np.linalg.solve(matrix, rhs)
    return np.concatenate([np.stack([a0, a1, a2], axis=1), tail], axis=1)


def _quartic_coefficients(
    start: np.ndarray,  # (n, 3) = s, s', s''
    end_speed: np.ndarray,  # (n,)
    horizon: np.ndarray,  # (n,)
) -> np.ndarray:
    """Coefficients a0..a4 of the velocity-keeping longitudinal polynomial."""
    n = start.shape[0]
    a0, a1, a2 = start[:, 0], start[:, 1], 0.5 * start[:, 2]
    t = horizon
    t2, t3 = t**2, t**3

    matrix = np.empty((n, 2, 2))
    matrix[:, 0] = np.stack([3 * t2, 4 * t3], axis=1)
    matrix[:, 1] = np.stack([6 * t, 12 * t2], axis=1)
    rhs = np.stack([end_speed - (a1 + 2 * a2 * t), -2 * a2], axis=1)
    tail = np.linalg.solve(matrix, rhs)
    return np.concatenate([np.stack([a0, a1, a2], axis=1), tail], axis=1)


def _polyval(coeffs: np.ndarray, times: np.ndarray, derivative: int = 0) -> np.ndarray:
    """Evaluate a batch of polynomials (n, k) on a time grid (n, m)."""
    order = coeffs.shape[1]
    result = np.zeros_like(times)
    for power in range(derivative, order):
        factor = 1.0
        for d in range(derivative):
            factor *= power - d
        result = result + factor * coeffs[:, power : power + 1] * times ** (power - derivative)
    return result


def integrated_squared_jerk(coeffs, horizons):
    """Exact jerk integral over each polynomial's own horizon."""
    derivative = np.stack([coeffs[:, p] * p * (p - 1) * (p - 2)
                           for p in range(3, coeffs.shape[1])], axis=1)
    result = np.zeros(len(coeffs))
    for i in range(derivative.shape[1]):
        for j in range(derivative.shape[1]):
            power = i + j + 1
            result += derivative[:, i] * derivative[:, j] * horizons**power / power
    return np.maximum(result, 0.)


def cartesian_feasible(world, times, max_speed, max_accel, max_curvature):
    """Check sampled Cartesian speed, total acceleration and curvature."""
    feasible = np.ones(len(world), dtype=bool)
    for k, path in enumerate(world):
        unique = np.r_[True, np.diff(times[k]) > 1e-9]
        t, xy = times[k, unique], path[unique]
        if len(t) < 3:
            feasible[k] = False
            continue
        velocity = np.gradient(xy, t, axis=0, edge_order=2)
        acceleration = np.gradient(velocity, t, axis=0, edge_order=2)
        speed = np.linalg.norm(velocity, axis=1)
        cross = velocity[:, 0] * acceleration[:, 1] - velocity[:, 1] * acceleration[:, 0]
        curvature = np.abs(cross) / np.maximum(speed**3, 1e-8)
        feasible[k] = (np.max(speed) <= max_speed + 1e-6
                       and np.max(np.linalg.norm(acceleration, axis=1)) <= max_accel + 1e-6
                       and np.max(curvature) <= max_curvature + 1e-6)
    return feasible


def trajectory_collision_free(s_path, d_path, times, predictions, frame, length, width, dt):
    """Match constant-velocity predictions to each candidate's own time support."""
    if not predictions.shape[0]:
        return np.ones(len(s_path), dtype=bool)
    valid = np.diff(times, axis=1) > 1e-9
    velocity = (predictions[:, 1] - predictions[:, 0]) / dt
    predicted = predictions[None, :, :1, :] + velocity[None, :, None, :] * times[:, None, :, None]
    shape = predicted.shape[:-1]
    ns, nd, _ = frame.project_many(predicted.reshape(-1, 2))
    separation = np.maximum(np.abs(s_path[:, None, :] - ns.reshape(shape)) - length,
                            np.abs(d_path[:, None, :] - nd.reshape(shape)) - width)
    return np.all(np.where(valid[:, None, :], separation[:, :, 1:] > 0, True), axis=(1, 2))


class FrenetPlannerController(BaseController):
    """Sampled polynomial trajectories in corridor coordinates."""

    name = "frenet"

    def __init__(
        self,
        lateral_samples: int = 9,
        lateral_range: float = 4.5,
        horizons: tuple[float, ...] = (2.0, 3.0, 4.0),
        speed_samples: int = 5,
        speed_range: float = 2.5,
        k_jerk: float = 0.1,
        k_time: float = 0.6,
        k_offset: float = 1.5,
        k_centre: float = 0.1,
        k_speed: float = 1.0,
        k_lateral: float = 1.0,
        k_longitudinal: float = 1.0,
        max_accel: float = 4.0,
        safety_margin: float = 0.6,
        name: str | None = None,
    ):
        self.lateral_samples = int(lateral_samples)
        self.lateral_range = float(lateral_range)
        self.horizons = tuple(float(h) for h in horizons)
        self.speed_samples = int(speed_samples)
        self.speed_range = float(speed_range)
        self.k_jerk = float(k_jerk)
        self.k_time = float(k_time)
        self.k_offset = float(k_offset)
        self.k_centre = float(k_centre)
        self.k_speed = float(k_speed)
        self.k_lateral = float(k_lateral)
        self.k_longitudinal = float(k_longitudinal)
        self.max_accel = float(max_accel)
        self.safety_margin = float(safety_margin)
        if name:
            self.name = name
        self._radius = 2.4
        self._dest_s: np.ndarray = np.array([])
        self._grid: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None

    def reset(self, scenario: "Scenario") -> None:
        self._radius = 0.5 * float(np.hypot(scenario.vehicle_length, scenario.vehicle_width))
        self._dest_s = np.array([a.dest_s for a in scenario.agents], dtype=float)
        offsets = np.linspace(-self.lateral_range, self.lateral_range, self.lateral_samples)
        horizons = np.array(self.horizons)
        speed_deltas = np.linspace(-self.speed_range, self.speed_range, self.speed_samples)
        grid = np.meshgrid(offsets, horizons, speed_deltas, indexing="ij")
        self._grid = tuple(g.ravel() for g in grid)

    def compute_controls(
        self,
        agents: list[TrafficAgent],
        scenario: "Scenario",
        step: int,
    ) -> list[tuple[float, float]]:
        from RL.decision import control_feasible, trajectory_obb_free
        dt = float(scenario.dt)
        max_speed = float(scenario.sim_config.get("max_agent_speed", 16.0))
        offsets, horizons, speed_deltas = self._grid
        n_candidates = offsets.shape[0]

        controls: list[tuple[float, float]] = []
        for i, agent in enumerate(agents):
            if agent.reached_destination:
                controls.append((0.0, 0.0))
                continue

            route = agent_route(scenario.corridor, agent)
            s0, d0, tangent, _, _ = route.project(agent.pos)
            tangent_angle = float(np.arctan2(tangent[1], tangent[0]))
            heading_error = wrap_angle(float(agent.heading) - tangent_angle)
            speed = float(agent.speed)
            s_dot = max(speed * np.cos(heading_error), 0.1)
            d_dot = speed * np.sin(heading_error)

            dest_s = float(self._dest_s[i])
            approach = goal_approach_control(agent, scenario, dest_s)
            if approach is not None and control_feasible(i, agents, *approach, scenario.sim_config, scenario.corridor):
                controls.append(approach)
                continue

            target_speed = np.clip(
                float(agent.desired_speed) + speed_deltas,
                0.5,
                max_speed,
            )
            lateral_start = np.tile(np.array([d0, d_dot, 0.0]), (n_candidates, 1))
            longitudinal_start = np.tile(np.array([float(s0), s_dot, 0.0]), (n_candidates, 1))

            lat_coeffs = _quintic_coefficients(lateral_start, offsets, horizons)
            lon_coeffs = _quartic_coefficients(longitudinal_start, target_speed, horizons)

            steps = int(np.ceil(max(self.horizons) / dt))
            time_grid = np.arange(steps + 1) * dt
            times = np.minimum(time_grid[None, :], horizons[:, None])

            d_path = _polyval(lat_coeffs, times)
            s_path = _polyval(lon_coeffs, times)
            s_rate = _polyval(lon_coeffs, times, derivative=1)
            lon_accel = _polyval(lon_coeffs, times, derivative=2)
            lat_jerk = _polyval(lat_coeffs, times, derivative=3)
            lon_jerk = _polyval(lon_coeffs, times, derivative=3)

            frame = build_local_frame(
                route,
                float(s0),
                ahead=float(np.max(s_path) - s0) + 20.0,
            )
            world = frame.point_at(s_path.ravel(), d_path.ravel()).reshape(n_candidates, -1, 2)
            _, _, clearance = frame.project_many(world.reshape(-1, 2))
            clearance = clearance.reshape(n_candidates, -1)

            feasible = (
                (np.max(np.abs(lon_accel), axis=1) <= self.max_accel)
                & (np.min(s_rate, axis=1) >= -0.1)
                & (np.min(clearance[:, 1:], axis=1) >= 0.5 * scenario.vehicle_width)
            )
            feasible &= cartesian_feasible(
                world, times, max_speed,
                min(self.max_accel, float(scenario.sim_config.get("max_accel", 4.))),
                np.tan(.45) / float(scenario.sim_config.get("wheelbase", 2.8)))

            predictions = predict_neighbours(agents, i, steps, dt, sim_config=scenario.sim_config)
            if predictions.shape[0]:
                # Match neighbours to each candidate's actual times, including a
                # partial final interval. Ignore repeated padded endpoints.
                feasible &= trajectory_collision_free(
                    s_path, d_path, times, predictions, frame,
                    scenario.vehicle_length + self.safety_margin,
                    scenario.vehicle_width + self.safety_margin, dt)

            # The corridor has no lanes, so the reference the planner should hold
            # is the agent's own lateral position, with only a weak pull to the
            # centreline. Penalising |offset| alone crowds every agent onto d = 0.
            cost_lat = (
                self.k_jerk * integrated_squared_jerk(lat_coeffs, horizons)
                + self.k_time * horizons
                + self.k_offset * (offsets - float(d0)) ** 2
                + self.k_centre * offsets**2
            )
            cost_lon = (
                self.k_jerk * integrated_squared_jerk(lon_coeffs, horizons)
                + self.k_time * horizons
                + self.k_speed * (target_speed - float(agent.desired_speed)) ** 2
            )
            cost = self.k_lateral * cost_lat + self.k_longitudinal * cost_lon

            delta = np.diff(world, axis=1)
            headings = np.concatenate([np.full((n_candidates, 1), agent.heading),
                                       np.arctan2(delta[..., 1], delta[..., 0])], axis=1)
            feasible &= trajectory_obb_free(world, headings, times, agents, i, scenario.sim_config)
            commands = [None] * n_candidates
            for k in np.flatnonzero(feasible):
                commands[k] = velocity_to_control(agent, (world[k, 1]-agent.pos)/dt, scenario)
                feasible[k] = control_feasible(i, agents, *commands[k], scenario.sim_config, scenario.corridor)

            if not np.any(feasible):
                # No feasible trajectory in the sample set: emergency brake, the
                # standard fallback for this planner family.
                controls.append((-self.max_accel, 0.0))
                continue
            best = int(np.argmin(np.where(feasible, cost, np.inf)))

            controls.append(commands[best])
        return controls
