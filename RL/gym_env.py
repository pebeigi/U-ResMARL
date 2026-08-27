"""Gymnasium / RLlib multi-agent wrapper for the traffic simulator."""

from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

import RL._paths  # noqa: F401
from RL.calibration_io import (
    DEFAULT_RESIDUAL_SCALES,
    RESIDUAL_PARAM_KEYS,
    residual_vector_to_dict,
)
from RL.traffic_env import EnvConfig, MultiAgentTrafficEnv

try:
    from ray.rllib.env.multi_agent_env import MultiAgentEnv
except ImportError:  # pragma: no cover - RLlib optional
    MultiAgentEnv = gym.Env  # type: ignore[misc,assignment]


def agent_id(index: int) -> str:
    return f"agent_{index}"


def agent_index(agent_key: str) -> int:
    return int(agent_key.split("_")[1])


try:
    from gymnasium.envs.registration import register

    register(
        id="TrafficMAR-v0",
        entry_point="RL.gym_env:TrafficMARLEnv",
        max_episode_steps=500,
    )
except Exception:
    # Registration can fail if called multiple times in the same interpreter.
    pass


class TrafficMARLEnv(MultiAgentEnv):
    """
    RLlib-compatible multi-agent environment.

    Each agent outputs a residual vector ΔΘ over the modulated utility
    parameters (weights + collision-kernel scales). The environment
    applies utility maximization and kinematic updates internally.
    """

    metadata = {"render_modes": []}

    def __init__(self, env_config: dict[str, Any] | None = None):
        env_config = env_config or {}
        scales = env_config.get("residual_scales", DEFAULT_RESIDUAL_SCALES)
        self.residual_scales = np.asarray(
            [float(scales[k]) for k in RESIDUAL_PARAM_KEYS],
            dtype=np.float32,
        )
        # Scalar fallback used only for logging / legacy configs.
        self.residual_scale = float(env_config.get("residual_scale", float(np.mean(self.residual_scales))))

        self._cfg = EnvConfig(
            dt=float(env_config.get("dt", 0.5)),
            max_steps=int(env_config.get("max_steps", 240)),
            num_agents=int(env_config.get("num_agents", 10)),
            base_desired_speed=float(env_config.get("base_desired_speed", 8.0)),
            min_initial_spacing=float(env_config.get("min_initial_spacing", 8.0)),
            run_id=int(env_config.get("run_id", 2)),
            lane_kf=int(env_config.get("lane_kf", 1)),
            base_params=env_config.get("base_params"),
            reward_weights=env_config.get("reward_weights"),
            sim_config=env_config.get("sim_config"),
        )
        self._env = MultiAgentTrafficEnv(self._cfg, seed=env_config.get("seed"))
        self._agent_ids = {agent_id(i) for i in range(self._cfg.num_agents)}

        obs_dim = self._env.obs_dim
        act_dim = len(RESIDUAL_PARAM_KEYS)
        high_obs = np.full(obs_dim, np.inf, dtype=np.float32)
        low_obs = -high_obs

        agent_list = sorted(self._agent_ids)
        self.observation_spaces = {
            aid: spaces.Box(low=low_obs, high=high_obs, dtype=np.float32) for aid in agent_list
        }
        self.action_spaces = {
            aid: spaces.Box(
                low=-self.residual_scales,
                high=self.residual_scales,
                dtype=np.float32,
            )
            for aid in agent_list
        }
        self.observation_space = self.observation_spaces[agent_list[0]]
        self.action_space = self.action_spaces[agent_list[0]]
        super().__init__()

    @property
    def agents(self):
        """RLlib / demo helper: live agent id set."""
        return set(self._agent_ids)

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        if seed is not None:
            self._env.rng = np.random.default_rng(seed)
        obs_list = self._env.reset()
        agent_list = sorted(self._agent_ids)
        observations = {agent_list[i]: obs_list[i] for i in range(len(obs_list))}
        infos = {aid: {} for aid in agent_list}
        return observations, infos

    def step(self, action_dict: dict[str, np.ndarray]):
        residual_actions: list[dict[str, float]] = []
        agent_list = sorted(self._agent_ids)
        for i in range(self._cfg.num_agents):
            aid = agent_list[i]
            if aid in action_dict and action_dict[aid] is not None:
                vec = np.clip(np.asarray(action_dict[aid], dtype=np.float32), -self.residual_scales, self.residual_scales)
                residual_actions.append(residual_vector_to_dict(vec))
            else:
                residual_actions.append({})

        obs_list, rewards_list, done, info = self._env.step(residual_actions)
        observations = {agent_list[i]: obs_list[i] for i in range(len(obs_list))}
        rewards = {agent_list[i]: float(rewards_list[i]) for i in range(len(rewards_list))}
        terminateds = {aid: False for aid in agent_list}
        truncateds = {aid: False for aid in agent_list}
        terminateds["__all__"] = done
        truncateds["__all__"] = done
        infos = {agent_list[i]: dict(info) for i in range(len(agent_list))}
        return observations, rewards, terminateds, truncateds, infos

    def rollout_metric(self) -> float:
        return self._env.rollout_metric()
