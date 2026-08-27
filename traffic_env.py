"""Compatibility shim — implementation lives in RL.traffic_env.

Kept so calibration / visualization scripts can keep importing `traffic_env`.
"""

from RL.traffic_env import *  # noqa: F401,F403
from RL.traffic_env import EnvConfig, MultiAgentTrafficEnv

__all__ = ["EnvConfig", "MultiAgentTrafficEnv"]
