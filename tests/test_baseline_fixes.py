import unittest
from unittest.mock import patch
import numpy as np
import torch
from scipy.special import ndtr
from scipy.stats import truncnorm

from Baselines.scenario import build_scenario
from Baselines.mppi import MPPIController
from Baselines.frenet_planner import integrated_squared_jerk, cartesian_feasible
from Baselines.train_marl import Episode, episode_gae


class BaselineFixTests(unittest.TestCase):
    def test_frenet_ignores_collisions_after_candidate_horizon(self):
        from Baselines.frenet_planner import trajectory_collision_free
        times = np.array([[0., 1., 2., 2., 2.], [0., 1., 2., 3., 4.]])
        path = np.zeros_like(times)
        predictions = np.stack((6. - 2. * np.arange(5), np.zeros(5)), axis=-1)[None, :, :]
        class Frame:
            def project_many(self, points):
                return points[:, 0], points[:, 1], np.ones(len(points)) * 10
        actual = trajectory_collision_free(path, path, times, predictions, Frame(), 1., 1., 1.)
        np.testing.assert_array_equal(actual, [True, False])

    def test_hatrpo_rejects_steps_above_configured_kl_limit(self):
        from Baselines import train_marl
        from Baselines.marl import MARLPolicy
        from types import SimpleNamespace
        torch.manual_seed(0)
        policy = MARLPolicy(32, 2, algo="hatrpo", hidden_dim=8)
        episode = train_marl.collect_episode(build_scenario(0, num_agents=2, max_steps=3), policy)
        batch = train_marl.build_batch([episode], .99, .95)
        before = [p.detach().clone() for p in policy.actors.parameters()]
        original = train_marl._gaussian_kl
        trials = []
        def kl(*args):
            if torch.is_grad_enabled():
                return original(*args)
            trials.append(True)
            return torch.tensor(.012)  # Above .01, below the old 1.5x relaxation.
        args = SimpleNamespace(cg_iterations=3, cg_damping=.1, max_kl=.01,
                               line_search_decay=.5, line_search_steps=3, accept_ratio=0.)
        with patch.object(train_marl, "_gaussian_kl", side_effect=kl):
            train_marl.update_actors_hatrpo(policy, batch, args)
        self.assertTrue(trials)
        for old, new in zip(before, policy.actors.parameters()):
            torch.testing.assert_close(old, new)

    def test_mppi_update_matches_bounded_density_importance_weights(self):
        scenario = build_scenario(0, num_agents=1, max_steps=1)
        agent = scenario.spawn_agents()[0]
        controller = MPPIController(horizon=1, samples=2, accel_std=1., steering_std=1.,
                                    w_control=0., w_steering=0.)
        controller.reset(scenario)
        controller._nominal[agent.agent_id][:] = [3.9, 0.]
        samples = np.array([3.99, 2.9])
        class Quantiles:
            def uniform(self, *args, **kwargs):
                return np.stack((ndtr(samples - 3.9), np.full(2, .5)), axis=-1)[:, None, :]
        controller.rng = Quantiles()
        weights = truncnorm.pdf(samples, -4., 4.) / truncnorm.pdf(samples, -7.9, .1, loc=3.9)
        expected = float(weights @ samples / weights.sum())
        with patch.object(controller, "_rollout_cost", return_value=np.zeros(2)), \
             patch("Baselines.mppi.sanitize_control", side_effect=lambda i, a, agents, c, s: c):
            actual = controller.compute_controls([agent], scenario, 0)[0][0]
        self.assertAlmostEqual(actual, expected, places=8)

    def test_mppi_reset_reproduces_same_scenario(self):
        scenario = build_scenario(5, num_agents=1, max_steps=1)
        controller = MPPIController()
        controller.reset(scenario)
        first = controller.rng.uniform(size=20)
        controller.reset(build_scenario(6, num_agents=1, max_steps=1))
        controller.rng.uniform(size=100)
        controller.reset(scenario)
        np.testing.assert_array_equal(first, controller.rng.uniform(size=20))

    def test_jerk_integral_uses_only_own_horizon(self):
        # d(t)=t^3 has constant jerk 6, so the integral is exactly 36*T.
        coeff = np.tile([0., 0., 0., 1., 0., 0.], (2, 1))
        np.testing.assert_allclose(integrated_squared_jerk(coeff, np.array([2., 4.])), [72., 144.])

    def test_cartesian_limits_reject_acceleration_and_curvature(self):
        times = np.linspace(0., 2., 21)[None, :]
        t = times[0]
        accelerating = np.stack((t, 3. * t**2), axis=-1)[None, :, :]
        self.assertFalse(cartesian_feasible(accelerating, times, 100., 4., 100.)[0])
        circle = np.stack((np.cos(t), np.sin(t)), axis=-1)[None, :, :]
        self.assertFalse(cartesian_feasible(circle, times, 100., 100., .2)[0])
        straight = np.stack((t, np.zeros_like(t)), axis=-1)[None, :, :]
        self.assertTrue(cartesian_feasible(straight, times, 2., 4., .2)[0])

    def test_team_returns_include_rewards_after_individual_arrival(self):
        rewards = np.array([[1., 1.], [2., 2.], [3., 3.]], dtype=np.float32)
        masks = np.array([[1., 1.], [0., 1.], [0., 1.]], dtype=np.float32)
        empty = np.zeros((3, 2, 1), dtype=np.float32)
        episode = Episode(empty, empty, empty, rewards*0, rewards*0, rewards, masks,
                          cooperative=True)
        _, returns = episode_gae(episode, 1., 1.)
        np.testing.assert_allclose(returns[0], [6., 6.])
        from Baselines.train_marl import build_batch
        batch = build_batch([episode], 1., 1.)
        self.assertEqual(batch.masks[1, 0], 0.)
        self.assertEqual(batch.critic_masks[1, 0], 1.)

    def test_team_reward_collection_is_shared(self):
        from Baselines.marl import MARLPolicy
        from Baselines.train_marl import collect_episode
        scenario = build_scenario(0, num_agents=2, max_steps=2)
        policy = MARLPolicy(32, 2, algo="happo", hidden_dim=8)
        episode = collect_episode(scenario, policy)
        self.assertTrue(episode.cooperative)
        np.testing.assert_array_equal(episode.rewards[:, 0], episode.rewards[:, 1])

    def test_dwa_detects_contact_during_braking(self):
        from Baselines.dwa import braking_admissible
        from test_boundary import straight_road
        from types import SimpleNamespace
        from utility_model import TrafficAgent
        sim = dict(build_scenario(0, num_agents=1, max_steps=1).sim_config)
        sim["boundary_safety_filter"] = False
        scenario = SimpleNamespace(dt=.5, sim_config=sim, corridor=straight_road(),
                                   vehicle_length=4.5, vehicle_width=1.8)
        ego = TrafficAgent(0, np.array([20., 0.]), np.array([10., 0.]), np.array([90., 0.]), 10., 0.)
        other = TrafficAgent(1, np.array([33., 0.]), np.zeros(2), np.array([90., 0.]), 10., 0.)
        # First step at x=25 is clear. Its subsequent stopping position x=35 is not.
        mask = braking_admissible(ego, [ego, other], 0, np.array([0.]), np.array([0.]), scenario)
        self.assertFalse(mask[0])
        self.assertTrue(braking_admissible(ego, [ego], 0, np.array([0.]), np.array([0.]), scenario)[0])


if __name__ == "__main__":
    unittest.main()
