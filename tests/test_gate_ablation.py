"""Gate vs residual-learning inference ablations."""
import unittest

import numpy as np

from Baselines.ablation_models import GATE_ABLATION_MODELS
from Baselines.registry import (
    LEARNED_CHECKPOINTS,
    RANDOM_INIT_RESIDUAL_MODELS,
    build_controller,
    is_learned,
    resolve_train_seeds,
)
from Baselines.residual_marl import build_random_residual_policy


class GateAblationRegistryTests(unittest.TestCase):
    def test_package_lists_prior_residual_gate_and_direct_rl(self):
        self.assertEqual(
            GATE_ABLATION_MODELS,
            [
                "utility_pt",
                "residual_no_gate",
                "residual_marl",
                "residual_random_gate",
                "direct_discrete_rl",
            ],
        )

    def test_no_gate_reuses_trained_residual_checkpoints(self):
        self.assertEqual(
            LEARNED_CHECKPOINTS["residual_no_gate"], LEARNED_CHECKPOINTS["residual_marl"]
        )
        ctrl = build_controller("residual_no_gate")
        self.assertFalse(ctrl.accept_if_better)
        self.assertTrue(build_controller("residual_marl").accept_if_better)

    def test_random_gate_is_seeded_without_weight_files(self):
        self.assertIn("residual_random_gate", RANDOM_INIT_RESIDUAL_MODELS)
        self.assertTrue(is_learned("residual_random_gate"))
        pairs = resolve_train_seeds("residual_random_gate", [0, 1, 2])
        self.assertEqual(pairs, [(0, None), (1, None), (2, None)])
        ctrl = build_controller("residual_random_gate", random_seed=7)
        self.assertTrue(ctrl.random_init)
        self.assertTrue(ctrl.accept_if_better)
        self.assertIsNone(ctrl.checkpoint)


class RandomResidualPolicyTests(unittest.TestCase):
    def test_random_actor_is_not_the_zero_head_prior(self):
        policy = build_random_residual_policy(32, seed=0, template=None)
        obs = np.linspace(-1.0, 1.0, 32, dtype=np.float32)
        residual, _ = policy.act(obs)
        self.assertEqual(residual.shape[-1], 63)
        self.assertGreater(float(np.max(np.abs(residual))), 1e-4)
        other = build_random_residual_policy(32, seed=1, template=None)
        first, _ = policy.act(obs)
        second, _ = other.act(obs)
        self.assertFalse(np.allclose(first, second))
