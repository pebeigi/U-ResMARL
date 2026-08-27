"""Compatibility shim — implementation lives in RL.gym_env."""

from RL.gym_env import *  # noqa: F401,F403
from RL.gym_env import TrafficMARLEnv, agent_id, agent_index

__all__ = ["TrafficMARLEnv", "agent_id", "agent_index"]
