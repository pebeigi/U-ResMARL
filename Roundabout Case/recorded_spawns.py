"""Recorded initial traffic cohorts in the supplied Jounieh coordinate frame.

The observed endpoint is an explicit, shared navigation goal. Future positions
are never replayed as controls or supplied as the planned route. This is a
goal-conditioned driving experiment, not endpoint-free trajectory forecasting.
"""
from __future__ import annotations

from functools import lru_cache
import numpy as np
import pandas as pd

from config import TRAJECTORIES_CSV, BASE_DESIRED_SPEED, MIN_RECORDED_GOAL_DISTANCE


def split_for_seed(seed):
    # The site launchers reserve these disjoint evaluation seed blocks.
    if seed is not None and 910000 <= seed < 920000:
        return "validation"
    if seed is not None and 810000 <= seed < 820000:
        return "test"
    return "train"


@lru_cache(maxsize=1)
def catalogue():
    from Baselines.data_evaluation import temporal_partition

    columns = ["id", "time", "xloc_kf", "yloc_kf", "keep_ego"]
    frame = pd.read_csv(TRAJECTORIES_CSV, usecols=columns).sort_values(["id", "time"])
    intervals, membership = temporal_partition(frame)
    # Use a causal half-second displacement, exactly as offline initialization.
    previous = frame[["id", "time", "xloc_kf", "yloc_kf"]].copy()
    previous["tick"] = np.rint((previous.time + .5) * 10).astype(int)
    previous = previous.rename(columns={"xloc_kf": "past_x", "yloc_kf": "past_y"})
    frame["tick"] = np.rint(frame.time * 10).astype(int)
    joined = frame.merge(previous[["id", "tick", "past_x", "past_y"]], on=["id", "tick"])
    joined["vx"] = (joined.xloc_kf - joined.past_x) / .5
    joined["vy"] = (joined.yloc_kf - joined.past_y) / .5
    joined["speed"] = np.hypot(joined.vx, joined.vy)
    joined["heading"] = np.arctan2(joined.vy, joined.vx)
    joined["split"] = joined.id.map(membership)
    # Every cohort belongs to exactly one partition; crossing tracks are purged.
    joined = joined[joined.keep_ego & joined.split.notna() & (joined.tick % 10 == 0)
                    & (joined.speed > .15)]
    tracks = {int(i): g for i, g in frame.groupby("id")}
    groups = {name: [g for _, g in joined[joined.split == name].groupby("tick")]
              for name in intervals}
    return groups, tracks, intervals


class RecordedSpawnPool:
    def __init__(self, corridor, sim):
        from RL.boundary import footprint_clearance

        self.corridor = corridor
        self.groups, tracks, self.intervals = catalogue()
        self.goals = {}
        self._poses = {}
        length, width = sim["vehicle_length"], sim["vehicle_width"]
        margin = sim.get("boundary_margin", .1)
        # An endpoint at the image edge often leaves half a car outside. Use the
        # last observed pose whose complete footprint is contained, not a random goal.
        for vehicle_id, track in tracks.items():
            xy = track[["xloc_kf", "yloc_kf"]].to_numpy()
            for index in range(len(xy) - 1, 0, -2):
                delta = xy[index] - xy[max(0, index - 5)]
                if np.linalg.norm(delta) < .05:
                    continue
                heading = float(np.arctan2(delta[1], delta[0]))
                if footprint_clearance(corridor, xy[index], heading, length, width) >= margin:
                    self.goals[vehicle_id] = xy[index].copy()
                    break

    def sample(self, env):
        from RL.boundary import footprint_clearance
        from RL.corridor import boxes_overlap
        from RL.spawn_safety import has_straight_braking_backup
        from RL.traffic_env import SpawnPackingError
        from utility_model import TrafficAgent

        sim, rng = env.config.sim_config, env.rng
        split = env._recorded_split
        groups = self.groups[split]
        n = env.config.num_agents
        eligible = [g for g in groups if len(g) >= n]
        if not eligible:
            raise SpawnPackingError(f"No {split} recorded cohort contains {n} moving vehicles")
        length, width = sim["vehicle_length"], sim["vehicle_width"]
        for _ in range(200):
            group = eligible[int(rng.integers(len(eligible)))]
            agents = []
            for row in group.iloc[rng.permutation(len(group))].itertuples():
                pos = np.array([row.xloc_kf, row.yloc_kf])
                goal = self.goals.get(int(row.id))
                if goal is None or np.linalg.norm(goal - pos) < MIN_RECORDED_GOAL_DISTANCE or row.speed > sim["max_agent_speed"]:
                    continue
                key = (int(row.id), int(row.tick))
                if key not in self._poses:
                    self._poses[key] = footprint_clearance(self.corridor, pos, row.heading, length, width) >= sim.get("boundary_margin", .1)
                if not self._poses[key]:
                    continue
                if any(boxes_overlap(pos, row.heading, a.pos, a.heading, length, width) for a in agents):
                    continue
                agent = TrafficAgent(agent_id=int(row.id), pos=pos,
                    vel=np.array([row.vx, row.vy]), dest=goal.copy(),
                    desired_speed=BASE_DESIRED_SPEED, nominal_y=float(pos[1]),
                    run_id=0, lane_kf=0, heading_angle=float(row.heading))
                # Reject incompatible samples; never alter recorded speeds to make them fit.
                if not has_straight_braking_backup(agents + [agent], sim, self.corridor):
                    continue
                agents.append(agent)
                if len(agents) == n:
                    env._dest_s = [0.] * n
                    sim["recorded_initialization"] = dict(split=split, time_s=float(row.time),
                        agent_ids=[int(a.agent_id) for a in agents],
                        goal="last footprint-contained recorded endpoint, supplied as navigation intent")
                    return agents
        raise SpawnPackingError(f"No feasible {n}-vehicle recorded {split} cohort; reduce agent count")
