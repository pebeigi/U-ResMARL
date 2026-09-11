"""Regression tests for the residual PPO trainer's credit assignment and actions."""

from __future__ import annotations

import unittest

import numpy as np
import torch

from RL.calibration_io import DEFAULT_RESIDUAL_SCALES, RESIDUAL_PARAM_KEYS
from RL.obs import EGO_OBS_DIM, observation_dim
from RL.param_gauge import AMPLITUDE_LOGIT_KEYS, amplitude_proportions, apply_gauge
from RL.train_ppo import (
    RESIDUAL_MODE_CANDIDATE,
    RESIDUAL_MODE_PARAM,
    SQUASHED_ACTION_SPACE,
    TorchResidualPolicy,
    _SQUASH_EPS,
    _persist_curve,
    compute_gae,
    compute_gae_by_trajectory,
)
from utility_model import DEFAULT_SIM_CONFIG, n_candidate_actions

GAMMA = 0.99
LAMBDA = 0.95


class ObservationLayoutTest(unittest.TestCase):
    def test_ego_dim_includes_remaining_station(self) -> None:
        self.assertEqual(EGO_OBS_DIM, 8)
        self.assertEqual(observation_dim(6), 8 + 4 * 6)


class TrajectoryGAETest(unittest.TestCase):
    def test_matches_per_trajectory_gae_on_interleaved_buffer(self) -> None:
        rng = np.random.default_rng(0)
        n_agents, n_steps = 3, 5
        rewards = rng.normal(size=n_agents * n_steps).astype(np.float32)
        values = rng.normal(size=n_agents * n_steps).astype(np.float32)
        traj_ids = np.array([a for _ in range(n_steps) for a in range(n_agents)])

        adv, ret = compute_gae_by_trajectory(rewards, values, traj_ids, GAMMA, LAMBDA)

        for agent in range(n_agents):
            idx = np.where(traj_ids == agent)[0]
            expected_adv, expected_ret = compute_gae(
                rewards[idx], values[idx], np.zeros(len(idx), dtype=np.float32), GAMMA, LAMBDA
            )
            np.testing.assert_allclose(adv[idx], expected_adv, rtol=1e-6)
            np.testing.assert_allclose(ret[idx], expected_ret, rtol=1e-6)

    def test_truncation_bootstraps_instead_of_zeroing(self) -> None:
        rewards = np.array([1.0, 1.0], dtype=np.float32)
        values = np.array([0.0, 10.0], dtype=np.float32)
        dones = np.array([0.0, 1.0], dtype=np.float32)
        timeouts = np.array([0.0, 1.0], dtype=np.float32)
        boot = np.array([0.0, 5.0], dtype=np.float32)

        truncated, _ = compute_gae(
            rewards, values, dones, GAMMA, LAMBDA, timeouts=timeouts, bootstrap_values=boot
        )
        terminal, _ = compute_gae(rewards, values, dones, GAMMA, LAMBDA)

        self.assertFalse(np.allclose(truncated, terminal))
        # Last advantage under truncation uses boot=5 rather than 0.
        self.assertGreater(float(truncated[-1]), float(terminal[-1]))


class CandidateLogitActionTest(unittest.TestCase):
    def setUp(self) -> None:
        self.obs_dim = observation_dim(4)
        self.n_cand = n_candidate_actions(DEFAULT_SIM_CONFIG)
        self.policy = TorchResidualPolicy(
            self.obs_dim,
            hidden_dim=16,
            action_space=SQUASHED_ACTION_SPACE,
            residual_mode=RESIDUAL_MODE_CANDIDATE,
            action_dim=self.n_cand,
        )

    def test_starts_on_the_prior(self) -> None:
        obs = np.linspace(-1.0, 1.0, self.obs_dim, dtype=np.float32)
        residual, _ = self.policy.act(obs, 0.0)
        np.testing.assert_allclose(residual, np.zeros(self.n_cand), atol=1e-6)

    def test_sampled_action_is_executed_logits(self) -> None:
        obs = np.linspace(-1.0, 1.0, self.obs_dim, dtype=np.float32)
        for _ in range(10):
            env_action, action, _, _ = self.policy.sample_action(obs)
            expected = np.tanh(action) * float(self.policy.residual_scales[0])
            np.testing.assert_allclose(env_action, expected, rtol=1e-5)

    def test_log_prob_includes_tanh_jacobian(self) -> None:
        obs = torch.linspace(-1.0, 1.0, self.obs_dim)
        pre_squash = torch.zeros(self.n_cand)
        log_prob, _, _ = self.policy.log_prob(obs, pre_squash)
        mean, _ = self.policy.evaluate(obs)
        std = self.policy._stddev(mean)
        dist = torch.distributions.Normal(mean, std)
        expected = dist.log_prob(pre_squash).sum() - torch.log(
            1.0 - torch.tanh(pre_squash).pow(2) + _SQUASH_EPS
        ).sum()
        self.assertAlmostEqual(float(log_prob), float(expected), places=5)


class ParamDeltaActionTest(unittest.TestCase):
    def setUp(self) -> None:
        self.obs_dim = observation_dim(4)
        self.policy = TorchResidualPolicy(
            self.obs_dim,
            hidden_dim=16,
            action_space=SQUASHED_ACTION_SPACE,
            residual_mode=RESIDUAL_MODE_PARAM,
        )

    def test_starts_on_the_prior(self) -> None:
        obs = np.linspace(-1.0, 1.0, self.obs_dim, dtype=np.float32)
        delta, _ = self.policy.act(obs, 0.0)
        for key in RESIDUAL_PARAM_KEYS:
            self.assertAlmostEqual(delta[key], 0.0, places=6)

    def test_residual_respects_scales(self) -> None:
        obs = np.linspace(-1.0, 1.0, self.obs_dim, dtype=np.float32)
        for _ in range(20):
            delta, _, _, _ = self.policy.sample_action(obs)
            for key in RESIDUAL_PARAM_KEYS:
                self.assertLessEqual(abs(delta[key]), DEFAULT_RESIDUAL_SCALES[key] + 1e-6)


class GaugeCenteringTest(unittest.TestCase):
    def test_common_logit_shift_has_no_effect(self) -> None:
        base = {
            "w_speed": 1.0,
            "w_prox": 1.0,
            "w_dir": 1.0,
            "w_path": 1.0,
            "w_coll": 1.0,
            "xi_i": 1.0,
            "gamma": 1.0,
            "sigma_long": 4.0,
            "sigma_lat": 1.5,
        }
        shifted = {key: 0.7 for key in AMPLITUDE_LOGIT_KEYS}
        unshifted = amplitude_proportions(apply_gauge(base, None, lambda p: p))
        with_shift = amplitude_proportions(apply_gauge(base, shifted, lambda p: p))
        for key, value in unshifted.items():
            self.assertAlmostEqual(with_shift[key], value, places=9)


class LearningCurveTest(unittest.TestCase):
    def test_writes_csv_and_png(self) -> None:
        import tempfile
        from pathlib import Path

        rows = [
            {
                "update": 0.0,
                "train_metric": float("nan"),
                "val_metric": 30.0,
                "val_arrival_rate": 0.85,
                "val_collisions": 0.0,
                "residual_usage": 0.0,
                "control_flip_rate": 0.0,
                "prior_val_metric": 30.0,
                "prior_val_arrival_rate": 0.85,
                "explained_variance": float("nan"),
                "approx_kl": float("nan"),
                "clip_frac": float("nan"),
                "entropy": float("nan"),
                "value_loss": float("nan"),
            },
            {
                "update": 10.0,
                "train_metric": 20.0,
                "val_metric": 25.0,
                "val_arrival_rate": 0.88,
                "val_collisions": 0.0,
                "residual_usage": 0.12,
                "control_flip_rate": 0.2,
                "prior_val_metric": 30.0,
                "prior_val_arrival_rate": 0.85,
                "explained_variance": 0.4,
                "approx_kl": 0.01,
                "clip_frac": 0.05,
                "entropy": 2.0,
                "value_loss": 1.0,
            },
        ]
        with tempfile.TemporaryDirectory() as tmp:
            csv_path = Path(tmp) / "curve.csv"
            _persist_curve(csv_path, rows)
            self.assertTrue(csv_path.exists())
            self.assertTrue(csv_path.with_suffix(".png").exists())


class CandidateSelectionTest(unittest.TestCase):
    def test_nonzero_logit_residual_can_flip_argmax(self) -> None:
        from utility_model import (
            DEFAULT_BASE_PARAMS,
            TrafficAgent,
            select_candidate_with_logit_residual,
        )

        agent = TrafficAgent(
            agent_id=0,
            pos=np.array([0.0, 0.0]),
            vel=np.array([5.0, 0.0]),
            dest=np.array([50.0, 0.0]),
            desired_speed=8.0,
            nominal_y=0.0,
        )
        sim = dict(DEFAULT_SIM_CONFIG)
        sim["obb_safety_filter"] = False
        sim["utility_frame"] = "destination"
        sim["path_mode"] = "boundary"
        n = n_candidate_actions(sim)
        residual = np.zeros(n)
        residual[-1] = 100.0  # force last candidate
        chosen, idx, prior_idx = select_candidate_with_logit_residual(
            0, agent, [agent], DEFAULT_BASE_PARAMS, sim, logit_residual=residual
        )
        self.assertEqual(idx, n - 1)
        self.assertNotEqual(idx, prior_idx)
        self.assertIn("accel_longitudinal", chosen)


if __name__ == "__main__":
    unittest.main()
