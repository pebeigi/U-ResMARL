"""Vectorised local corridor frame for sampling-based planners.

The planners evaluate thousands of candidate trajectory points per agent per
step. Calling `HighwayCorridor.project` on each of them is far too slow, so a
short window of the corridor around the agent is extracted once and all points
are projected against that window in a single batched operation.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

import Baselines._paths  # noqa: F401


@dataclass
class LocalFrame:
    """A short slice of the corridor with batched projection."""

    vertices: np.ndarray  # (m, 2) centreline
    station: np.ndarray  # (m,) cumulative s at each vertex
    lat_lower: np.ndarray  # (m,) signed offset of the lower edge along the normal
    lat_upper: np.ndarray  # (m,) signed offset of the upper edge along the normal

    def project_many(self, points: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Project points onto the window -> (station, lateral, signed clearance).

        The clearance is the distance to the nearer corridor edge; negative
        means the point is outside the corridor.
        """
        points = np.atleast_2d(np.asarray(points, dtype=float))
        a = self.vertices[:-1]  # (m-1, 2)
        b = self.vertices[1:]
        ab = b - a
        seg_len_sq = np.maximum(np.sum(ab * ab, axis=1), 1e-12)  # (m-1,)

        delta = points[:, None, :] - a[None, :, :]  # (N, m-1, 2)
        t = np.clip(np.sum(delta * ab[None, :, :], axis=2) / seg_len_sq[None, :], 0.0, 1.0)
        closest = a[None, :, :] + t[:, :, None] * ab[None, :, :]
        dist_sq = np.sum((points[:, None, :] - closest) ** 2, axis=2)  # (N, m-1)
        best = np.argmin(dist_sq, axis=1)  # (N,)

        rows = np.arange(points.shape[0])
        seg_len = np.sqrt(seg_len_sq)
        tangent = ab / seg_len[:, None]
        normal = np.stack([-tangent[:, 1], tangent[:, 0]], axis=1)  # (m-1, 2)

        t_best = t[rows, best]
        station = self.station[best] + t_best * seg_len[best]
        lateral = np.sum((points - closest[rows, best]) * normal[best], axis=1)

        # Interpolate the edge offsets along the chosen segment.
        low = self.lat_lower[best] + t_best * (self.lat_lower[best + 1] - self.lat_lower[best])
        high = self.lat_upper[best] + t_best * (self.lat_upper[best + 1] - self.lat_upper[best])
        inner = np.minimum(low, high)
        outer = np.maximum(low, high)
        clearance = np.minimum(lateral - inner, outer - lateral)
        return station, lateral, clearance

    def point_at(self, station: np.ndarray, lateral: np.ndarray) -> np.ndarray:
        """Inverse map (s, lateral) -> world xy, for trajectory generation."""
        station = np.atleast_1d(np.asarray(station, dtype=float))
        lateral = np.atleast_1d(np.asarray(lateral, dtype=float))
        idx = np.clip(
            np.searchsorted(self.station, station, side="right") - 1,
            0,
            len(self.vertices) - 2,
        )
        a = self.vertices[idx]
        b = self.vertices[idx + 1]
        ab = b - a
        seg_len = np.maximum(np.linalg.norm(ab, axis=1), 1e-9)
        t = np.clip((station - self.station[idx]) / seg_len, 0.0, 1.0)
        tangent = ab / seg_len[:, None]
        normal = np.stack([-tangent[:, 1], tangent[:, 0]], axis=1)
        return a + t[:, None] * ab + lateral[:, None] * normal

    def half_width_at(self, station: np.ndarray) -> np.ndarray:
        idx = np.clip(
            np.searchsorted(self.station, np.atleast_1d(station), side="right") - 1,
            0,
            len(self.vertices) - 1,
        )
        return 0.5 * np.abs(self.lat_upper[idx] - self.lat_lower[idx])


def build_local_frame(corridor, station: float, back: float = 15.0, ahead: float = 80.0) -> LocalFrame:
    """Extract the corridor window covering [station - back, station + ahead]."""
    lo = float(station) - float(back)
    hi = float(station) + float(ahead)
    i0 = int(np.searchsorted(corridor.cumulative_s, lo, side="right") - 1)
    i1 = int(np.searchsorted(corridor.cumulative_s, hi, side="left") + 1)
    i0 = max(0, i0)
    i1 = min(len(corridor.center) - 1, max(i1, i0 + 2))

    center = corridor.center[i0 : i1 + 1]
    lower = corridor.lower[i0 : i1 + 1]
    upper = corridor.upper[i0 : i1 + 1]
    station_axis = corridor.cumulative_s[i0 : i1 + 1]

    # Vertex normals from the local tangent, matching HighwayCorridor.project.
    seg = np.diff(center, axis=0)
    seg_tangent = seg / np.maximum(np.linalg.norm(seg, axis=1)[:, None], 1e-12)
    vertex_tangent = np.vstack([seg_tangent[:1], seg_tangent])
    vertex_normal = np.stack([-vertex_tangent[:, 1], vertex_tangent[:, 0]], axis=1)

    lat_lower = np.sum((lower - center) * vertex_normal, axis=1)
    lat_upper = np.sum((upper - center) * vertex_normal, axis=1)
    return LocalFrame(
        vertices=center,
        station=station_axis,
        lat_lower=lat_lower,
        lat_upper=lat_upper,
    )


def frenet_conflict(
    ego_station: np.ndarray,  # (k, H+1)
    ego_lateral: np.ndarray,  # (k, H+1)
    neighbour_station: np.ndarray,  # (n, H+1)
    neighbour_lateral: np.ndarray,  # (n, H+1)
    length: float,
    width: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Footprint conflict between candidate trajectories and predicted neighbours.

    A circumscribed disc is far too conservative on a corridor where vehicles
    routinely pass side by side, so the footprint is treated as a box in
    corridor coordinates: a conflict needs overlap both along and across the
    corridor. Returns (separation, overlap), each (k, n, H); separation > 0
    means the footprints are clear and overlap > 0 measures the intrusion.
    The current step is excluded because it is not actionable.
    """
    if neighbour_station.shape[0] == 0:
        k = ego_station.shape[0]
        horizon = max(ego_station.shape[1] - 1, 1)
        return np.full((k, 1, horizon), np.inf), np.zeros((k, 1, horizon))

    delta_s = np.abs(ego_station[:, None, 1:] - neighbour_station[None, :, 1:])
    delta_d = np.abs(ego_lateral[:, None, 1:] - neighbour_lateral[None, :, 1:])
    longitudinal_gap = delta_s - length
    lateral_gap = delta_d - width
    separation = np.maximum(longitudinal_gap, lateral_gap)
    overlap = np.maximum(0.0, -longitudinal_gap) * np.maximum(0.0, -lateral_gap)
    return separation, overlap


def predict_neighbours(
    agents,
    ego_idx: int,
    horizon_steps: int,
    dt: float,
    radius: float = 60.0,
    max_neighbours: int = 8,
) -> np.ndarray:
    """Constant-velocity prediction of nearby agents: (k, horizon_steps+1, 2)."""
    ego = agents[ego_idx]
    ranked = []
    for j, other in enumerate(agents):
        if j == ego_idx or other.reached_destination:
            continue
        d = float(np.linalg.norm(other.pos - ego.pos))
        if d <= radius:
            ranked.append((d, j))
    ranked.sort(key=lambda x: x[0])
    selected = [j for _, j in ranked[:max_neighbours]]
    if not selected:
        return np.zeros((0, horizon_steps + 1, 2))

    times = np.arange(horizon_steps + 1)[None, :, None] * dt
    positions = np.array([agents[j].pos for j in selected])[:, None, :]
    velocities = np.array([agents[j].vel for j in selected])[:, None, :]
    return positions + velocities * times
