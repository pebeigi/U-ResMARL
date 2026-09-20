"""Explicit per-agent geometry; highway callers keep their original corridor."""
import numpy as np


def agent_route(corridor, agent):
    return corridor.for_agent(agent) if hasattr(corridor, 'for_agent') else corridor


def agent_station(corridor, agent, point=None):
    point = agent.pos if point is None else point
    if hasattr(corridor, 'remaining_to_goal'):
        return -float(corridor.remaining_to_goal(point, agent.dest))
    return float(corridor.project(point)[0])


def arrival_reached(corridor, agent, dest_s, tolerance):
    if hasattr(corridor, 'remaining_to_goal'):
        # Grid distance and route projection cannot certify a physical arrival.
        return bool(np.linalg.norm(agent.pos-agent.dest) <= tolerance)
    return bool(agent.reached_destination or corridor.project(agent.pos)[0] >= dest_s-tolerance)
