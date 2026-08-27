"""ORCA baseline (Optimal Reciprocal Collision Avoidance).

Reference: van den Berg et al., "Reciprocal n-Body Collision Avoidance",
ISRR 2011. Each agent solves a 2-D linear program for the velocity closest to
its preferred velocity subject to one reciprocal half-plane per neighbour, plus
two static half-planes for the corridor edges. The resulting holonomic velocity
is mapped onto the shared bicycle model so that ORCA is judged with exactly the
same actuation limits as every other model.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

import Baselines._paths  # noqa: F401
from Baselines.controllers import BaseController
from Baselines.dynamics import preferred_velocity, velocity_to_control
from utility_model import TrafficAgent

if TYPE_CHECKING:  # pragma: no cover
    from Baselines.scenario import Scenario

EPS = 1e-5


@dataclass
class Line:
    point: np.ndarray
    direction: np.ndarray


def _det(a: np.ndarray, b: np.ndarray) -> float:
    return float(a[0] * b[1] - a[1] * b[0])


def _normalize(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    return v / n if n > 1e-12 else np.array([1.0, 0.0])


def _linear_program1(
    lines: list[Line],
    line_no: int,
    radius: float,
    opt_velocity: np.ndarray,
    direction_opt: bool,
    result: np.ndarray,
) -> tuple[bool, np.ndarray]:
    """Optimise along a single constraint line, subject to the previous ones."""
    line = lines[line_no]
    dot = float(line.point @ line.direction)
    discriminant = dot * dot + radius * radius - float(line.point @ line.point)
    if discriminant < 0.0:
        return False, result

    sqrt_disc = float(np.sqrt(discriminant))
    t_left = -dot - sqrt_disc
    t_right = -dot + sqrt_disc

    for i in range(line_no):
        denominator = _det(line.direction, lines[i].direction)
        numerator = _det(lines[i].direction, line.point - lines[i].point)
        if abs(denominator) <= EPS:
            if numerator < 0.0:
                return False, result
            continue
        t = numerator / denominator
        if denominator >= 0.0:
            t_right = min(t_right, t)
        else:
            t_left = max(t_left, t)
        if t_left > t_right:
            return False, result

    if direction_opt:
        t = t_right if float(opt_velocity @ line.direction) > 0.0 else t_left
    else:
        t = float(line.direction @ (opt_velocity - line.point))
        t = float(np.clip(t, t_left, t_right))
    return True, line.point + t * line.direction


def _linear_program2(
    lines: list[Line],
    radius: float,
    opt_velocity: np.ndarray,
    direction_opt: bool,
) -> tuple[int, np.ndarray]:
    if direction_opt:
        result = opt_velocity * radius
    elif float(opt_velocity @ opt_velocity) > radius * radius:
        result = _normalize(opt_velocity) * radius
    else:
        result = np.array(opt_velocity, dtype=float)

    for i, line in enumerate(lines):
        if _det(line.direction, line.point - result) > 0.0:
            temp = result
            ok, result = _linear_program1(lines, i, radius, opt_velocity, direction_opt, result)
            if not ok:
                return i, temp
    return len(lines), result


def _linear_program3(
    lines: list[Line],
    num_obst_lines: int,
    begin_line: int,
    radius: float,
    result: np.ndarray,
) -> np.ndarray:
    """Densest-feasible fallback when the program is infeasible."""
    distance = 0.0
    for i in range(begin_line, len(lines)):
        if _det(lines[i].direction, lines[i].point - result) <= distance:
            continue
        proj_lines = list(lines[:num_obst_lines])
        for j in range(num_obst_lines, i):
            determinant = _det(lines[i].direction, lines[j].direction)
            if abs(determinant) <= EPS:
                if float(lines[i].direction @ lines[j].direction) > 0.0:
                    continue
                point = 0.5 * (lines[i].point + lines[j].point)
            else:
                offset = _det(lines[j].direction, lines[i].point - lines[j].point) / determinant
                point = lines[i].point + offset * lines[i].direction
            direction = _normalize(lines[j].direction - lines[i].direction)
            proj_lines.append(Line(point, direction))

        temp = result
        new_dir = np.array([-lines[i].direction[1], lines[i].direction[0]], dtype=float)
        fail, result = _linear_program2(proj_lines, radius, new_dir, True)
        if fail < len(proj_lines):
            result = temp
        distance = _det(lines[i].direction, lines[i].point - result)
    return result


def orca_line(
    pos_a: np.ndarray,
    vel_a: np.ndarray,
    pos_b: np.ndarray,
    vel_b: np.ndarray,
    combined_radius: float,
    time_horizon: float,
    dt: float,
    responsibility: float = 0.5,
) -> Line:
    """Half-plane of velocities for A that avoids B for `time_horizon` seconds."""
    rel_pos = np.asarray(pos_b, dtype=float) - np.asarray(pos_a, dtype=float)
    rel_vel = np.asarray(vel_a, dtype=float) - np.asarray(vel_b, dtype=float)
    dist_sq = float(rel_pos @ rel_pos)
    radius_sq = combined_radius * combined_radius

    if dist_sq > radius_sq:
        w = rel_vel - rel_pos / time_horizon
        w_len_sq = float(w @ w)
        dot1 = float(w @ rel_pos)
        if dot1 < 0.0 and dot1 * dot1 > radius_sq * w_len_sq:
            # Projection on the cut-off circle.
            w_len = float(np.sqrt(w_len_sq))
            unit_w = w / max(w_len, 1e-12)
            direction = np.array([unit_w[1], -unit_w[0]], dtype=float)
            u = (combined_radius / time_horizon - w_len) * unit_w
        else:
            # Projection on one of the legs of the velocity obstacle.
            leg = float(np.sqrt(max(dist_sq - radius_sq, 0.0)))
            if _det(rel_pos, w) > 0.0:
                direction = (
                    np.array(
                        [
                            rel_pos[0] * leg - rel_pos[1] * combined_radius,
                            rel_pos[0] * combined_radius + rel_pos[1] * leg,
                        ]
                    )
                    / dist_sq
                )
            else:
                direction = (
                    -np.array(
                        [
                            rel_pos[0] * leg + rel_pos[1] * combined_radius,
                            -rel_pos[0] * combined_radius + rel_pos[1] * leg,
                        ]
                    )
                    / dist_sq
                )
            dot2 = float(rel_vel @ direction)
            u = dot2 * direction - rel_vel
    else:
        # Already overlapping: escape within one timestep.
        inv_dt = 1.0 / max(dt, 1e-6)
        w = rel_vel - rel_pos * inv_dt
        w_len = float(np.linalg.norm(w))
        unit_w = w / max(w_len, 1e-12)
        direction = np.array([unit_w[1], -unit_w[0]], dtype=float)
        u = (combined_radius * inv_dt - w_len) * unit_w

    return Line(np.asarray(vel_a, dtype=float) + responsibility * u, direction)


class ORCAController(BaseController):
    """Reciprocal velocity-obstacle collision avoidance with corridor walls."""

    name = "orca"

    def __init__(
        self,
        time_horizon: float = 4.0,
        time_horizon_obstacle: float = 2.0,
        neighbor_dist: float = 50.0,
        max_neighbors: int = 10,
        radius: float | None = None,
        safety_margin: float = 0.0,
        lookahead: float = 15.0,
        name: str | None = None,
    ):
        self.time_horizon = float(time_horizon)
        self.time_horizon_obstacle = float(time_horizon_obstacle)
        self.neighbor_dist = float(neighbor_dist)
        self.max_neighbors = int(max_neighbors)
        self.radius = radius
        self.safety_margin = float(safety_margin)
        self.lookahead = float(lookahead)
        if name:
            self.name = name
        self._radius = 2.4
        self._dest_s: np.ndarray = np.array([])

    def reset(self, scenario: "Scenario") -> None:
        if self.radius is not None:
            self._radius = float(self.radius)
        else:
            # Circumscribed disc of the vehicle box: disc-disjoint => box-disjoint.
            length = scenario.vehicle_length
            width = scenario.vehicle_width
            self._radius = 0.5 * float(np.hypot(length, width))
        self._radius += self.safety_margin
        self._dest_s = np.array([a.dest_s for a in scenario.agents], dtype=float)

    def _wall_lines(self, agent: TrafficAgent, scenario: "Scenario") -> list[Line]:
        """Static half-planes keeping the agent inside the measured corridor."""
        _, _, _, seg_i, t = scenario.corridor.project(agent.pos)
        lower, upper = scenario.corridor.edge_points_at(seg_i, t)
        chord = upper - lower
        chord_len = float(np.linalg.norm(chord))
        if chord_len < 1e-6:
            return []
        unit = chord / chord_len  # points from the lower edge toward the upper edge
        mid = 0.5 * (lower + upper)
        signed = float((np.asarray(agent.pos, dtype=float) - mid) @ unit)
        c_lo = 0.5 * chord_len + signed
        c_hi = 0.5 * chord_len - signed
        tau = max(self.time_horizon_obstacle, 1e-3)

        lines: list[Line] = []
        for clearance, normal in ((c_lo, unit), (c_hi, -unit)):
            slack = clearance - self._radius
            # Feasible set: n . v >= -slack / tau
            c = -slack / tau
            point = c * normal
            direction = np.array([normal[1], -normal[0]], dtype=float)
            lines.append(Line(point, direction))
        return lines

    def compute_controls(
        self,
        agents: list[TrafficAgent],
        scenario: "Scenario",
        step: int,
    ) -> list[tuple[float, float]]:
        max_speed = float(scenario.sim_config.get("max_agent_speed", 16.0))
        dt = float(scenario.dt)
        controls: list[tuple[float, float]] = []

        for i, agent in enumerate(agents):
            if agent.reached_destination:
                controls.append((0.0, 0.0))
                continue

            v_pref = preferred_velocity(agent, scenario, float(self._dest_s[i]), self.lookahead)
            if float(np.linalg.norm(v_pref)) > max_speed:
                v_pref = _normalize(v_pref) * max_speed

            wall_lines = self._wall_lines(agent, scenario)
            num_obst = len(wall_lines)

            ranked: list[tuple[float, int]] = []
            for j, other in enumerate(agents):
                if j == i or other.reached_destination:
                    continue
                d = float(np.linalg.norm(other.pos - agent.pos))
                if d <= self.neighbor_dist:
                    ranked.append((d, j))
            ranked.sort(key=lambda x: x[0])

            lines = list(wall_lines)
            for _, j in ranked[: self.max_neighbors]:
                lines.append(
                    orca_line(
                        agent.pos,
                        agent.vel,
                        agents[j].pos,
                        agents[j].vel,
                        2.0 * self._radius,
                        self.time_horizon,
                        dt,
                    )
                )

            fail, v_new = _linear_program2(lines, max_speed, v_pref, False)
            if fail < len(lines):
                v_new = _linear_program3(lines, num_obst, fail, max_speed, v_new)

            controls.append(velocity_to_control(agent, v_new, scenario))
        return controls
