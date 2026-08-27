"""Self-driven particle / social-force baseline.

Reference: Helbing & Molnar, "Social force model for pedestrian dynamics",
Phys. Rev. E 51, 1995; Helbing & Vicsek, self-driven many-particle systems.

Each agent is a self-driven particle: a relaxation force toward its desired
velocity, anisotropic exponential repulsion from neighbours, and exponential
repulsion from the two corridor edges. The net force gives a desired velocity
that is mapped onto the same bicycle model used by every other model.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

import Baselines._paths  # noqa: F401
from Baselines.controllers import BaseController
from Baselines.dynamics import preferred_velocity, velocity_to_control
from utility_model import TrafficAgent

if TYPE_CHECKING:  # pragma: no cover
    from Baselines.scenario import Scenario


class SocialForceController(BaseController):
    """v_des = v + dt * (relaxation + neighbour repulsion + wall repulsion)."""

    name = "social_force"

    def __init__(
        self,
        tau: float = 0.6,
        a_social: float = 8.0,
        b_social: float = 2.5,
        anisotropy: float = 0.35,
        a_wall: float = 12.0,
        b_wall: float = 1.0,
        interaction_range: float = 40.0,
        lookahead: float = 15.0,
        name: str | None = None,
    ):
        self.tau = float(tau)
        self.a_social = float(a_social)
        self.b_social = float(b_social)
        self.anisotropy = float(anisotropy)
        self.a_wall = float(a_wall)
        self.b_wall = float(b_wall)
        self.interaction_range = float(interaction_range)
        self.lookahead = float(lookahead)
        if name:
            self.name = name
        self._radius = 2.4
        self._dest_s: np.ndarray = np.array([])

    def reset(self, scenario: "Scenario") -> None:
        self._radius = 0.5 * float(np.hypot(scenario.vehicle_length, scenario.vehicle_width))
        self._dest_s = np.array([a.dest_s for a in scenario.agents], dtype=float)

    def _neighbour_force(
        self,
        agent: TrafficAgent,
        agents: list[TrafficAgent],
        idx: int,
    ) -> np.ndarray:
        force = np.zeros(2, dtype=float)
        heading_vec = np.array([np.cos(agent.heading), np.sin(agent.heading)], dtype=float)
        for j, other in enumerate(agents):
            if j == idx or other.reached_destination:
                continue
            diff = agent.pos - other.pos
            d = float(np.linalg.norm(diff))
            if d > self.interaction_range or d < 1e-6:
                continue
            n_ij = diff / d
            magnitude = self.a_social * np.exp((2.0 * self._radius - d) / self.b_social)
            # Anisotropy: interactions ahead of the agent matter more.
            cos_phi = float(-n_ij @ heading_vec)
            weight = self.anisotropy + (1.0 - self.anisotropy) * 0.5 * (1.0 + cos_phi)
            force += magnitude * weight * n_ij
        return force

    def _wall_force(self, agent: TrafficAgent, scenario: "Scenario") -> np.ndarray:
        _, _, _, seg_i, t = scenario.corridor.project(agent.pos)
        lower, upper = scenario.corridor.edge_points_at(seg_i, t)
        chord = upper - lower
        chord_len = float(np.linalg.norm(chord))
        if chord_len < 1e-6:
            return np.zeros(2, dtype=float)
        unit = chord / chord_len
        mid = 0.5 * (lower + upper)
        signed = float((np.asarray(agent.pos, dtype=float) - mid) @ unit)
        c_lo = 0.5 * chord_len + signed
        c_hi = 0.5 * chord_len - signed
        force = np.zeros(2, dtype=float)
        for clearance, normal in ((c_lo, unit), (c_hi, -unit)):
            magnitude = self.a_wall * np.exp((self._radius - clearance) / self.b_wall)
            force += magnitude * normal
        return force

    def compute_controls(
        self,
        agents: list[TrafficAgent],
        scenario: "Scenario",
        step: int,
    ) -> list[tuple[float, float]]:
        dt = float(scenario.dt)
        max_speed = float(scenario.sim_config.get("max_agent_speed", 16.0))
        controls: list[tuple[float, float]] = []

        for i, agent in enumerate(agents):
            if agent.reached_destination:
                controls.append((0.0, 0.0))
                continue

            v_desired = preferred_velocity(agent, scenario, float(self._dest_s[i]), self.lookahead)
            force = (v_desired - agent.vel) / max(self.tau, 1e-6)
            force = force + self._neighbour_force(agent, agents, i)
            force = force + self._wall_force(agent, scenario)

            v_new = agent.vel + force * dt
            speed = float(np.linalg.norm(v_new))
            if speed > max_speed:
                v_new = v_new * (max_speed / speed)
            controls.append(velocity_to_control(agent, v_new, scenario))
        return controls
