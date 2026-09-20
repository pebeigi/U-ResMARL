"""Valid initial conditions and non-regressing residual checkpoint selection."""
import unittest
from itertools import combinations
from unittest.mock import patch
from types import SimpleNamespace
from pathlib import Path
import tempfile
import contextlib
import io

import numpy as np
import torch

from RL.boundary import footprint_clearance
from RL.corridor import boxes_overlap
from RL.traffic_env import EnvConfig, MultiAgentTrafficEnv, SpawnPackingError
from RL.train_ppo import _validation_improves, train, TorchResidualPolicy


class SpawnTests(unittest.TestCase):
    def test_previous_overlap_seeds_are_valid_and_reproducible(self):
        for seed in (0, 4, 900005, 700002, 700011):
            first = MultiAgentTrafficEnv(EnvConfig(), seed=seed)
            second = MultiAgentTrafficEnv(EnvConfig(), seed=seed)
            first.reset()
            second.reset()
            np.testing.assert_array_equal([a.pos for a in first.agents],
                                          [a.pos for a in second.agents])
            for a, b in combinations(first.agents, 2):
                self.assertGreaterEqual(np.linalg.norm(a.pos - b.pos), 8.0)
                self.assertFalse(boxes_overlap(a.pos, a.heading, b.pos, b.heading, 4.5, 1.8))
            for a in first.agents:
                self.assertGreaterEqual(footprint_clearance(first.corridor, a.pos, a.heading), .1)

    def test_dense_layout_keeps_requested_spacing(self):
        env = MultiAgentTrafficEnv(EnvConfig(num_agents=16, spawn_s_range=(20., 80.),
                                  spawn_lateral_frac=.55, min_initial_spacing=5.), seed=0)
        env.reset()
        self.assertTrue(all(np.linalg.norm(a.pos-b.pos) >= 5.
                            for a,b in combinations(env.agents, 2)))

    def test_failed_packing_never_uses_overlapping_fallback(self):
        env = MultiAgentTrafficEnv(EnvConfig(), seed=0)
        with patch.object(env, "_sample_start_pose", side_effect=SpawnPackingError("full")):
            with self.assertRaises(SpawnPackingError):
                env.reset()


class SelectionTests(unittest.TestCase):
    def setUp(self):
        self.prior = dict(collision_events=1., collisions=2., metric=100., arrival_rate=.9,
                          goal_progress=.9, offroad_rate=0.)

    def test_arrival_improvement_cannot_buy_more_contacts(self):
        candidate = dict(collision_events=2., collisions=2., metric=0., arrival_rate=1.,
                         goal_progress=1., offroad_rate=0.)
        self.assertFalse(_validation_improves(candidate, self.prior, self.prior))

    def test_fewer_events_cannot_buy_more_persistent_overlap(self):
        candidate = dict(collision_events=0., collisions=3., metric=0., arrival_rate=1.,
                         goal_progress=1., offroad_rate=0.)
        self.assertFalse(_validation_improves(candidate, self.prior, self.prior))

    def test_safer_policy_is_selected_even_if_leftover_is_worse(self):
        candidate = dict(collision_events=0., collisions=0., metric=120., arrival_rate=.9,
                         goal_progress=.9, offroad_rate=0.)
        self.assertTrue(_validation_improves(candidate, self.prior, self.prior))

    def test_in_band_comfort_does_not_veto_a_safer_policy(self):
        prior = {**self.prior, "rms_jerk": 7.0, "mean_abs_accel": 1.5, "mean_abs_steering": 0.04,
                 "unsafe_ttc_rate": 0.006}
        candidate = {**prior, "collision_events": 0., "collisions": 0., "metric": 30.,
                     "arrival_rate": 0.95, "goal_progress": 0.95, "rms_jerk": 8.0,
                     "mean_abs_accel": 1.9, "mean_abs_steering": 0.05, "unsafe_ttc_rate": 0.007}
        self.assertTrue(_validation_improves(candidate, prior, prior))

    def test_out_of_band_comfort_loses_when_safety_is_tied(self):
        prior = dict(collision_events=0., collisions=0., metric=100., arrival_rate=.9,
                     goal_progress=.9, offroad_rate=0., rms_jerk=2., mean_abs_accel=1.,
                     mean_abs_steering=0.04)
        candidate = {**prior, "rms_jerk": 20.}
        self.assertFalse(_validation_improves(candidate, prior, prior))

    def test_collision_free_can_win_with_slightly_lower_arrival(self):
        candidate = dict(collision_events=0., collisions=0., metric=0., arrival_rate=.8,
                         goal_progress=.8, offroad_rate=0.)
        self.assertTrue(_validation_improves(candidate, self.prior, self.prior))

    def test_tie_keeps_prior_but_same_safety_better_pdms_can_win(self):
        tied = dict(collision_events=0., collisions=0., metric=100., arrival_rate=.9,
                    goal_progress=.9, offroad_rate=0.)
        self.assertFalse(_validation_improves(tied, tied, tied))
        candidate = {**tied, "goal_progress": 0.95, "arrival_rate": 0.95, "metric": 90.}
        self.assertTrue(_validation_improves(candidate, tied, tied))

    def test_invalid_exploration_scales_are_rejected(self):
        for scale in (0., -1., float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                TorchResidualPolicy(32, candidate_logit_scale=scale)


class CollisionEventTests(unittest.TestCase):
    def test_persistent_contact_and_recontact_match_benchmark_definition(self):
        env = MultiAgentTrafficEnv(EnvConfig(num_agents=2, max_steps=5), seed=0)
        env.reset()
        def transition(pairs):
            return SimpleNamespace(controls=[(0., 0.)] * 2, rewards=[0., 0.],
                                   collision_pairs=pairs, colliding_agents={i for p in pairs for i in p},
                                   candidates=[])
        sequence = [{(0, 1)}, {(0, 1)}, set(), {(0, 1)}]
        with patch("RL.traffic_env.advance_agents", side_effect=[transition(p) for p in sequence]):
            for _ in sequence:
                _, _, _, info = env.step()
        self.assertEqual(info["collision_count"], 3)
        self.assertEqual(info["collision_events"], 2)
        env.reset()
        self.assertEqual(env.collision_events, 0)
        self.assertEqual(env._collision_pairs, set())


class SavedFallbackTests(unittest.TestCase):
    def test_training_restores_and_labels_prior_if_update_regresses(self):
        args = SimpleNamespace(seed=2, hidden_dim=8, lr=.001, calibration=None,
                               prefer_params="robust", collision_penalty=8.,
                               behavior_coef=0., behavior_shaping_coef=0.,
                               val_episodes=1, test_episodes=1,
                               curve=None, updates=1, anneal_lr=False, val_every=1,
                               episodes_per_update=1, gamma=.99, gae_lambda=.95,
                               clip_coef=.2, value_coef=.5, entropy_coef=0.,
                               ppo_epochs=1, minibatch_size=8, target_kl=.02,
                               log_every=1, num_agents=2, max_steps=1)
        env = MultiAgentTrafficEnv(EnvConfig(num_agents=2, max_steps=1), seed=0)
        calls = []
        def evaluate(args, policy, seeds, **kwargs):
            calls.append(list(seeds))
            changed = policy is not None and bool(torch.any(policy.actor[4].bias != 0))
            return dict(metric=0., collisions=float(changed), collision_events=float(changed),
                        arrival_rate=1., residual_usage=0., control_flip_rate=0.)
        def update(policy, *args, **kwargs):
            with torch.no_grad():
                policy.actor[4].bias.fill_(1.)
            return dict(explained_variance=0., approx_kl=0., epochs_run=1., entropy=0.)
        with tempfile.TemporaryDirectory() as tmp:
            args.save = Path(tmp) / "policy.pt"
            with patch("RL.train_ppo.make_env", return_value=env), \
                 patch("RL.train_ppo.evaluate_deterministic", side_effect=evaluate), \
                 patch("RL.train_ppo.collect_rollouts", return_value=(None, [0.], [0], [float("nan")], {"control_flip_rate":0., "environment_steps":1, "active_agent_transitions":2})), \
                 patch("RL.train_ppo.ppo_update", side_effect=update), \
                 patch("RL.train_ppo._persist_curve"), contextlib.redirect_stdout(io.StringIO()):
                train(args)
            saved = torch.load(args.save, map_location="cpu", weights_only=False)
        self.assertEqual(saved["selected_update"], 0)
        self.assertEqual(saved["selection_status"], "utility_fallback")
        self.assertEqual(saved["spawn_protocol_version"], 3)
        self.assertEqual(saved["test_collision_events"], 0.)
        self.assertTrue(torch.all(saved["state_dict"]["actor.4.bias"] == 0))
        self.assertEqual(calls, [[910000], [910000], [810000], [810000]])

if __name__ == "__main__":
    unittest.main()
