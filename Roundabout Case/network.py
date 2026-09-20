"""Jounieh roundabout roadway: extracted curb polygon, not PCA lanes."""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np
from shapely.geometry import Point
from shapely.prepared import prep

from config import (
    ROUTING_CLEARANCE,
    SPAWN_CLEARANCE,
    STREET_BOUNDARIES,
    VEHICLE_LENGTH,
    VEHICLE_WIDTH,
)

_CALIB = None


def _calib():
    global _CALIB
    if _CALIB is None:
        from Calibration.calibrate_utility_from_data import (
            OnRoadDestField,
            load_site_roadway,
            polygon_path_error,
        )

        _CALIB = (load_site_roadway, OnRoadDestField, polygon_path_error)
    return _CALIB


def site_vehicle_size() -> tuple[float, float]:
    """Closed-loop box in the extracted-curb metre frame (long side along heading)."""
    return float(VEHICLE_LENGTH), float(VEHICLE_WIDTH)


class SiteRoute:
    """Fixed route for one explicit start/goal, with physical polygon clearance."""

    def __init__(self, site, start, dest):
        from RL.corridor import HighwayCorridor

        self.site, self.roadway = site, site.roadway
        self.dest = np.asarray(dest, float).copy()
        points = site.route_points(start, dest)
        segment = np.diff(points, axis=0)
        distance = np.linalg.norm(segment, axis=1)
        stations = np.r_[0.0, np.cumsum(distance)]
        tangent = segment / distance[:, None]
        normals = np.c_[-tangent[:, 1], tangent[:, 0]]
        normals = np.vstack([normals, normals[-1]])
        width = np.array([max(site.clearances(p)[0], 0.1) for p in points])
        self.geometry = HighwayCorridor(
            0,
            0,
            points,
            points - normals * width[:, None],
            points + normals * width[:, None],
            stations,
            tangent,
        )
        self.center, self.lower, self.upper = self.geometry.center, self.geometry.lower, self.geometry.upper
        self.cumulative_s = stations - stations[-1]
        self.tangents = tangent
        self.run_id = self.lane_kf = 0

    @property
    def length(self):
        return self.geometry.length

    def project(self, point):
        s, n, tangent, i, t = self.geometry.project(point)
        return s - self.length, n, tangent, i, t

    def xy_from_frenet(self, s, lateral):
        return self.geometry.xy_from_frenet(float(s) + self.length, lateral)

    def clearances(self, point):
        return self.site.clearances(point)

    def inside(self, point, margin=0.0):
        return self.site.inside(point, margin)

    def edge_points_at(self, i, t):
        return self.geometry.edge_points_at(i, t)

    def wall_contacts(self, point):
        return self.site.wall_contacts(point)

    def make_local_frame(self, *_args, **_kwargs):
        from Baselines.local_frame import LocalFrame
        import shapely

        site = self.site

        class PolygonFrame(LocalFrame):
            def project_many(self, points):
                station, lateral, _ = super().project_many(points)
                geom = shapely.points(np.atleast_2d(points))
                clearance = shapely.distance(geom, site.roadway.boundary)
                clearance = np.where(shapely.covers(site.roadway, geom), clearance, -clearance)
                return station, lateral, clearance

        width = np.array([max(site.clearances(p)[0], 0.1) for p in self.center])
        return PolygonFrame(self.center, self.cumulative_s, -width, width)


class SiteNetworkCorridor:
    """Immutable shared curb geometry. Goals belong to agents, never this object."""

    run_id = 0
    lane_kf = 0

    def __init__(self, roadway):
        import shapely
        from scipy.sparse.csgraph import shortest_path

        _, OnRoadDestField, self._polygon_path_error = _calib()
        self.roadway = roadway
        self.on_road = OnRoadDestField(roadway)
        self.map_boundaries = [np.asarray(roadway.exterior.coords)] + [
            np.asarray(r.coords) for r in roadway.interiors
        ]
        self.center = self.lower = self.map_boundaries[0]
        self.upper = self.map_boundaries[-1]
        minx, miny, maxx, maxy = roadway.bounds
        self._length = float(np.hypot(maxx - minx, maxy - miny))
        shrunk = roadway.buffer(-float(ROUTING_CLEARANCE)).simplify(0.08, preserve_topology=True)
        if shrunk.is_empty:
            raise ValueError("Roundabout route clearance emptied the curb polygon")
        if shrunk.geom_type == "MultiPolygon":
            shrunk = max(shrunk.geoms, key=lambda g: g.area)
        if shrunk.geom_type != "Polygon":
            raise ValueError("Roundabout route clearance disconnected the network")
        self.routing_region = shrunk.buffer(-0.02, join_style=2)
        if self.routing_region.is_empty or self.routing_region.geom_type != "Polygon":
            self.routing_region = shrunk
        rings = [self.routing_region.exterior, *self.routing_region.interiors]
        self.nodes = np.concatenate([np.asarray(r.coords)[:-1] for r in rings])
        delta = self.nodes[:, None] - self.nodes[None]
        weights = np.linalg.norm(delta, axis=-1)
        pairs = np.stack(np.broadcast_arrays(self.nodes[:, None], self.nodes[None]), axis=-2)
        visible = shapely.covers(self.routing_region.buffer(1e-7), shapely.linestrings(pairs))
        weights[~visible] = np.inf
        np.fill_diagonal(weights, 0.0)
        self.distances, self.predecessors = shortest_path(weights, directed=False, return_predecessors=True)
        self._goal_cache = {}
        self._interior_pool = self._build_interior_pool(clearance=float(SPAWN_CLEARANCE))

    @property
    def length(self):
        return self._length

    def project(self, _point):
        raise TypeError("Network projection requires an explicit agent route")

    def for_agent(self, agent):
        route = getattr(agent, "_route_geometry", None)
        if route is None or getattr(route, "site", None) is not self or not np.array_equal(route.dest, agent.dest):
            route = SiteRoute(self, agent.pos, agent.dest)
            agent._route_geometry = route
        return route

    def _connections(self, point, region=None):
        import shapely

        point = np.asarray(point, float)
        segments = shapely.linestrings(np.stack([np.broadcast_to(point, self.nodes.shape), self.nodes], axis=1))
        visible = shapely.covers(self.roadway if region is None else region, segments)
        return np.where(visible, np.linalg.norm(self.nodes - point, axis=1), np.inf)

    def _goal_distances(self, dest):
        key = tuple(np.asarray(dest, float))
        if key not in self._goal_cache:
            costs = self._connections(dest)
            total = self.distances + costs[None]
            last = np.argmin(total, axis=1)
            if len(self._goal_cache) > 512:
                self._goal_cache.clear()
            self._goal_cache[key] = (total[np.arange(len(last)), last], last)
        return self._goal_cache[key]

    def remaining_to_goal(self, point, dest):
        from shapely.geometry import LineString
        from shapely.ops import nearest_points

        if not self.roadway.covers(Point(*np.asarray(point, float))):
            nearest = np.asarray(nearest_points(Point(*point), self.roadway)[1].coords[0])
            return float(np.linalg.norm(np.asarray(point) - nearest)) + self.remaining_to_goal(nearest, dest)
        if self.roadway.covers(LineString([point, dest])):
            return float(np.linalg.norm(np.asarray(point) - dest))
        distances, _ = self._goal_distances(dest)
        result = float(np.min(self._connections(point) + distances))
        if not np.isfinite(result):
            raise ValueError("Point or goal is not connected through the roadway")
        return result

    def route_points(self, start, dest):
        from shapely.geometry import LineString

        start, dest = np.asarray(start, float), np.asarray(dest, float)
        if np.linalg.norm(start - dest) < 1e-6:
            raise ValueError("Cannot create a route with coincident start and goal")
        region = self.routing_region.buffer(1e-7)
        if region.covers(LineString([start, dest])):
            corners = np.array([start, dest])
        else:
            a, b = self._connections(start, region), self._connections(dest, region)
            if not np.isfinite(a).any():
                a = self._connections(start)
            if not np.isfinite(b).any():
                b = self._connections(dest)
            costs = a[:, None] + self.distances + b[None]
            first, last = np.unravel_index(np.argmin(costs), costs.shape)
            if not np.isfinite(costs[first, last]):
                raise ValueError("No road-contained route")
            ids = [last]
            while ids[-1] != first:
                previous = int(self.predecessors[first, ids[-1]])
                if previous < 0:
                    raise ValueError("Disconnected route graph")
                ids.append(previous)
            corners = np.vstack([start, self.nodes[ids[::-1]], dest])
        points = [corners[0]]
        for a, b in zip(corners[:-1], corners[1:]):
            length = np.linalg.norm(b - a)
            if length < 1e-7:
                continue
            points.extend(a + (b - a) * t for t in np.linspace(0.0, 1.0, max(1, int(np.ceil(length / 2.0))) + 1)[1:])
        return np.array(points)

    def _edge_state(self, point):
        geom = Point(*np.asarray(point, float))
        return float(self.roadway.boundary.distance(geom)), bool(self.roadway.covers(geom))

    def clearances(self, point):
        d, inside = self._edge_state(point)
        return (d if inside else -d), (d if inside else -d), 0.0

    def inside(self, point, margin=0.0):
        d, inside = self._edge_state(point)
        return inside and d >= margin

    def wall_contacts(self, point):
        from shapely.ops import nearest_points

        p = np.asarray(point, float)
        for ring in [self.roadway.exterior, *self.roadway.interiors]:
            q = np.asarray(nearest_points(Point(*p), ring)[1].coords[0])
            d = float(np.linalg.norm(p - q))
            if d > 1e-9:
                yield d, (p - q) / d

    def path_error(self, point, boundary_buffer=1.5):
        return float(self._polygon_path_error(np.atleast_2d(point), self.roadway, boundary_buffer)[0])

    def _build_interior_pool(self, n: int = 2500, clearance: float = 0.7) -> np.ndarray:
        minx, miny, maxx, maxy = self.roadway.bounds
        rng = np.random.default_rng(0)
        pts = []
        for _ in range(n * 60):
            xy = np.array([rng.uniform(minx, maxx), rng.uniform(miny, maxy)], dtype=float)
            if self.inside(xy, margin=clearance):
                pts.append(xy)
                if len(pts) >= n:
                    break
        if len(pts) < 32:
            raise RuntimeError("Roundabout curb polygon has too little interior for spawn")
        return np.asarray(pts, dtype=float)

    def sample_interior(self, rng: np.random.Generator, clearance: float, tries: int = 800):
        if len(self._interior_pool):
            for _ in range(int(tries)):
                xy = self._interior_pool[int(rng.integers(0, len(self._interior_pool)))]
                if self.inside(xy, margin=float(clearance)):
                    return np.asarray(xy, dtype=float)
        minx, miny, maxx, maxy = self.roadway.bounds
        for _ in range(int(tries)):
            xy = np.array([rng.uniform(minx, maxx), rng.uniform(miny, maxy)], dtype=float)
            if self.inside(xy, margin=float(clearance)):
                return xy
        raise RuntimeError("No interior sample inside the roundabout curb polygon")


@lru_cache(maxsize=1)
def load_site_corridor(csv_path: str | None = None) -> SiteNetworkCorridor:
    load_site_roadway, _, _ = _calib()
    path = Path(csv_path) if csv_path else STREET_BOUNDARIES
    roadway = load_site_roadway(path)
    if roadway is None or roadway.is_empty:
        raise FileNotFoundError(f"Roundabout curb polygon missing or empty: {path}")
    return SiteNetworkCorridor(roadway)


_ROADS = {}


def road_region_from_site(corridor, margin: float = 0.0):
    roadway = getattr(corridor, "roadway", None)
    if roadway is None:
        raise TypeError("Roundabout road_region requires a site corridor")
    if not np.isfinite(margin) or margin < 0:
        raise ValueError("Boundary margin must be finite and nonnegative")
    key = (id(roadway), float(margin))
    cached = _ROADS.get(key)
    if cached is None:
        region = roadway.buffer(-float(margin)) if margin else roadway
        if region.is_empty:
            raise ValueError("Boundary margin leaves no drivable road")
        cached = (region, prep(region))
        _ROADS[key] = cached
    return cached
