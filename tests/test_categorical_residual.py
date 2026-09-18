import unittest
from pathlib import Path
import tempfile
import numpy as np
import torch

from RL.candidate_policy import CandidateIndex, candidate_context
from RL.train_ppo import TorchResidualPolicy, CATEGORICAL_ACTION_SPACE, collect_rollouts, ppo_update
from RL.traffic_env import EnvConfig, MultiAgentTrafficEnv
from RL.spawn_safety import swept_fixed_heading_pairs, has_straight_braking_backup
from RL.obs import contact_safety_reward, footprint_surface_gap
from utility_model import (select_candidate_with_logit_residual, TrafficAgent,
                           candidate_obb_conflict, build_step_context)


class CategoricalTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.policy = TorchResidualPolicy(32, hidden_dim=8, action_dim=3,
                                         action_space=CATEGORICAL_ACTION_SPACE,
                                         candidate_logit_scale=.5, candidate_temperature=.1)
        self.obs = torch.zeros(32)

    def test_probabilities_match_utility_softmax_and_exclude_masked_actions(self):
        utility = torch.tensor([0., .1, 100.])
        mask = torch.tensor([True, True, False])
        dist, _ = self.policy.categorical_distribution(self.obs, utility, mask)
        np.testing.assert_allclose(dist.probs.detach().numpy(),
                                   [1/(1+np.e), np.e/(1+np.e), 0.], rtol=1e-6)
        for _ in range(30):
            action, saved, logp, _ = self.policy.sample_action(self.obs.numpy(), utility, mask)
            self.assertIsInstance(action, CandidateIndex)
            self.assertIn(action.index, (0, 1))
            recomputed, _, _ = self.policy.log_prob(self.obs, torch.as_tensor(saved), utility, mask)
            self.assertAlmostEqual(logp, float(recomputed), places=6)

    def test_gradient_rewards_selected_action_and_ignores_masked_coordinate(self):
        loss, _, _ = self.policy.log_prob(self.obs, torch.tensor(0),
                                         torch.zeros(3), torch.tensor([True, True, False]))
        (-loss).backward()
        gradient = self.policy.actor[4].bias.grad
        self.assertLess(float(gradient[0]), 0.)
        self.assertGreater(float(gradient[1]), 0.)
        self.assertEqual(float(gradient[2]), 0.)

    def test_deterministic_residual_ranking_matches_distribution_mode(self):
        with torch.no_grad():
            self.policy.actor[4].bias.copy_(torch.tensor([.4, -.2, .9]))
        u = np.array([0., .1, -.2], dtype=np.float32)
        mask = np.array([True, True, False])
        delta, _ = self.policy.act(self.obs.numpy())
        dist, _ = self.policy.categorical_distribution(self.obs, u, mask)
        self.assertEqual(int(dist.probs.argmax()), int(np.argmax(np.where(mask, u+delta, -np.inf))))

    def test_empty_mask_is_rejected(self):
        with self.assertRaises(ValueError):
            self.policy.categorical_distribution(self.obs, np.zeros(3), np.zeros(3, dtype=bool))

    def test_nonzero_categorical_checkpoint_reloads_exactly(self):
        from Baselines.residual_marl import load_residual_policy
        with torch.no_grad():
            self.policy.actor[4].bias.copy_(torch.tensor([.2, -.3, .5]))
            self.policy.candidate_temperature.fill_(.13)
        blob = dict(protocol_version=3, obs_dim=32, hidden_dim=8,
                    action_space=CATEGORICAL_ACTION_SPACE, residual_mode="candidate_logits",
                    action_dim=3, state_dict=self.policy.state_dict(),
                    residual_scales={"c0": .5, "c1": .5, "c2": .5})
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "policy.pt"
            torch.save(blob, path)
            restored = load_residual_policy(path, 32)
        np.testing.assert_array_equal(self.policy.act(self.obs.numpy())[0], restored.act(self.obs.numpy())[0])
        before, _ = self.policy.categorical_distribution(self.obs, torch.zeros(3), torch.ones(3,dtype=torch.bool))
        after, _ = restored.categorical_distribution(self.obs, torch.zeros(3), torch.ones(3,dtype=torch.bool))
        np.testing.assert_array_equal(before.probs.detach().numpy(), after.probs.detach().numpy())

    def test_context_matches_existing_utility_selector_with_and_without_car_filter(self):
        for enabled in (False, True):
            env = MultiAgentTrafficEnv(EnvConfig(num_agents=3, obb_safety_filter=enabled), seed=42)
            env.reset()
            for i, a in enumerate(env.agents):
                context = candidate_context(i, env.agents, env.config.base_params, env.config.sim_config)
                for delta in (None, np.random.default_rng(i).normal(0, .1, 63)):
                    _, idx, prior = select_candidate_with_logit_residual(i, a, env.agents,
                                      env.config.base_params, env.config.sim_config, delta)
                    values = context.utilities if delta is None else context.utilities + delta
                    self.assertEqual(prior, context.prior_index)
                    self.assertEqual(idx, int(np.argmax(np.where(context.mask, values, -np.inf))))

    def test_rollout_probability_ratios_and_ppo_update_are_finite(self):
        env = MultiAgentTrafficEnv(EnvConfig(num_agents=2, max_steps=3), seed=42)
        policy = TorchResidualPolicy(env.obs_dim, hidden_dim=8, action_dim=env.residual_dim,
                                     action_space=CATEGORICAL_ACTION_SPACE)
        memory, *_ = collect_rollouts(env, policy, 1)
        obs = torch.tensor(np.array(memory.observations))
        actions = torch.tensor(np.array(memory.actions))
        logp, _, _ = policy.log_prob(obs, actions, np.array(memory.utilities), np.array(memory.action_masks))
        np.testing.assert_allclose(logp.detach().numpy(), memory.log_probs, atol=2e-6)
        stats = ppo_update(policy, torch.optim.Adam(policy.parameters(), lr=.001), memory,
                           gamma=.99, gae_lambda=.95, clip_coef=.2, value_coef=.5,
                           entropy_coef=0., epochs=2, minibatch_size=16, target_kl=.02)
        self.assertTrue(all(np.isfinite(v) for v in stats.values()))

    def test_distribution_mode_and_evaluation_execute_same_interacting_trajectory(self):
        from RL.calibration_io import load_base_params
        config = EnvConfig(num_agents=16, max_steps=6, spawn_s_range=(20., 80.),
                           spawn_lateral_frac=.55, min_initial_spacing=5.,
                           base_params=load_base_params())
        evaluation = MultiAgentTrafficEnv(config, seed=910101)
        categorical = MultiAgentTrafficEnv(config, seed=910101)
        obs = evaluation.reset()
        categorical.reset()
        policy = TorchResidualPolicy(32, hidden_dim=8, action_dim=63,
                                     action_space=CATEGORICAL_ACTION_SPACE,
                                     candidate_temperature=.005, candidate_logit_scale=.5)
        with torch.no_grad():
            policy.actor[4].bias.copy_(torch.linspace(-.03, .03, 63))
            for _ in range(6):
                deltas = [policy.act(o)[0] for o in obs]
                indices = []
                for i in range(16):
                    context = categorical.candidate_context(i)
                    dist, _ = policy.categorical_distribution(torch.tensor(obs[i]),
                                                               context.utilities, context.mask)
                    indices.append(CandidateIndex(int(dist.logits.argmax())))
                obs, rewards_eval, _, info_eval = evaluation.step(deltas)
                _, rewards_cat, _, info_cat = categorical.step(indices)
                self.assertEqual(info_eval["selected_controls"], info_cat["selected_controls"])
                np.testing.assert_allclose(rewards_eval, rewards_cat, atol=1e-10)
                np.testing.assert_allclose([a.pos for a in evaluation.agents],
                                           [a.pos for a in categorical.agents], atol=1e-10)


class BrakingSpawnTests(unittest.TestCase):
    def test_collision_filter_always_checks_execution_endpoint(self):
        # The crossing vehicle passes between .375 and .75 lookahead samples.
        ego = TrafficAgent(0, np.array([0., 0.]), np.array([0., 0.]), np.array([100.,0.]), 8., 0.)
        other = TrafficAgent(1, np.array([0., 20.]), np.array([0., -40.]), np.array([0.,-100.]), 40., 20.)
        sim = {"dt": .5, "conflict_horizon": 1.5, "conflict_substeps": 4,
               "vehicle_length": 4.5, "vehicle_width": 1.8, "utility_frame": "destination"}
        candidate = {"pos": ego.pos.copy(), "vel": ego.vel.copy(), "heading": ego.heading,
                     "time_to_reach": .5}
        ctx = build_step_context(0, ego, [ego, other], sim)
        self.assertNotIn(.5, [s[0] for s in ctx.conflict_samples])
        self.assertTrue(candidate_obb_conflict(candidate, 0, [ego, other], sim, context=ctx))

    def test_sweep_detects_between_endpoint_contact(self):
        start = np.array([[0., 0.], [10., 0.]])
        end = np.array([[10., 0.], [0., 0.]])
        self.assertEqual(swept_fixed_heading_pairs(start, end, np.zeros(2), 4.5, 1.8), {(0, 1)})

    def test_parallel_side_by_side_braking_is_safe(self):
        start = np.array([[0., 0.], [0., 2.1]])
        self.assertEqual(swept_fixed_heading_pairs(start, start+[10.,0.], np.zeros(2), 4.5, 1.8), set())

    def test_nonoverlap_is_insufficient_when_follower_cannot_stop(self):
        env = MultiAgentTrafficEnv(EnvConfig(num_agents=2), seed=42)
        env.reset()
        for a, x, speed in zip(env.agents, (0., 6.), (10., 2.)):
            a.pos = np.array([x, 0.]); a.vel = np.array([speed, 0.]); a.heading_angle = 0.
        self.assertFalse(has_straight_braking_backup(env.agents, env.config.sim_config))

    def test_side_by_side_clearance_is_not_penalized_as_a_crash(self):
        env = MultiAgentTrafficEnv(EnvConfig(num_agents=2), seed=42)
        env.reset()
        a, b = env.agents
        a.pos = np.array([0.,0.]); a.heading_angle = b.heading_angle = 0.
        b.pos = np.array([0.,2.5])
        side_gap = footprint_surface_gap(a,b)
        b.pos = np.array([5.2,0.])
        front_gap = footprint_surface_gap(a,b)
        self.assertAlmostEqual(side_gap, .7)
        self.assertAlmostEqual(front_gap, .7)
        self.assertGreater(contact_safety_reward(side_gap), -1.)
        self.assertAlmostEqual(contact_safety_reward(side_gap), contact_safety_reward(front_gap))


if __name__ == "__main__":
    unittest.main()
