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
        super().__init__()
        env_config = env_config or {}
        mode = env_config.get("residual_mode", "candidate_logits")
        self._cfg = EnvConfig(
            dt=float(env_config.get("dt", 0.5)),
            max_steps=int(env_config.get("max_steps", 240)),
            num_agents=int(env_config.get("num_agents", 10)),
            base_desired_speed=float(env_config.get("base_desired_speed", 8.0)),
            min_initial_spacing=float(env_config.get("min_initial_spacing", 8.0)),
            run_id=int(env_config.get("run_id", 2)), lane_kf=int(env_config.get("lane_kf", 1)),
            base_params=env_config.get("base_params"), reward_weights=env_config.get("reward_weights"),
            sim_config=env_config.get("sim_config"), residual_mode=mode,
            obb_safety_filter=bool(env_config.get("obb_safety_filter", False)),
            collision_penalty=float(env_config.get("collision_penalty", 8.0)),
            leftover_coef=float(env_config.get("leftover_coef", 0.05)),
            arrival_bonus=float(env_config.get("arrival_bonus", 5.0)),
        )
        self._env = MultiAgentTrafficEnv(self._cfg, seed=env_config.get("seed"))
        if mode == "candidate_logits":
            self.residual_scales = np.full(self._env.residual_dim,
                float(env_config.get("candidate_logit_scale", 2.0)), dtype=np.float32)
        else:
            scales = env_config.get("residual_scales", DEFAULT_RESIDUAL_SCALES)
            self.residual_scales = np.array([scales[k] for k in RESIDUAL_PARAM_KEYS], dtype=np.float32)
        self.residual_scale = float(self.residual_scales.mean())
        self.possible_agents = [agent_id(i) for i in range(self._cfg.num_agents)]
        self.agents = list(self.possible_agents)
        self._agent_ids = set(self.possible_agents)
        self.observation_spaces = {aid: spaces.Box(-np.inf, np.inf, shape=(self._env.obs_dim,), dtype=np.float32)
                                   for aid in self.possible_agents}
        self.action_spaces = {aid: spaces.Box(-self.residual_scales, self.residual_scales, dtype=np.float32)
                              for aid in self.possible_agents}
        self.observation_space = self.observation_spaces[self.agents[0]]
        self.action_space = self.action_spaces[self.agents[0]]

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        if seed is not None:
            self._env.rng = np.random.default_rng(seed)
        obs = self._env.reset()
        self.agents = list(self.possible_agents)
        return {aid: obs[agent_index(aid)] for aid in self.agents}, {aid: {} for aid in self.agents}

    def step(self, action_dict: dict[str, np.ndarray]):
        active = list(self.agents)
        residuals = [None] * self._cfg.num_agents
        for aid in active:
            if aid not in action_dict:
                continue
            vector = np.asarray(action_dict[aid], dtype=np.float32)
            if vector.shape != self.residual_scales.shape:
                raise ValueError(f"{aid}: expected residual shape {self.residual_scales.shape}, got {vector.shape}")
            vector = np.clip(vector, -self.residual_scales, self.residual_scales)
            residuals[agent_index(aid)] = (vector if self._cfg.residual_mode == "candidate_logits"
                                          else residual_vector_to_dict(vector))
        obs, rewards, done, info = self._env.step(residuals)
        terminated = {aid: bool(self._env.agents[agent_index(aid)].reached_destination) for aid in active}
        truncated = {aid: bool(info["truncated"] and not terminated[aid]) for aid in active}
        terminated["__all__"] = all(a.reached_destination for a in self._env.agents)
        truncated["__all__"] = bool(info["truncated"])
        self.agents = [] if done else [aid for aid in active if not terminated[aid]]
        return ({aid: obs[agent_index(aid)] for aid in active},
                {aid: float(rewards[agent_index(aid)]) for aid in active},
                terminated, truncated, {aid: dict(info) for aid in active})

    def rollout_metric(self) -> float:
        return self._env.rollout_metric()
