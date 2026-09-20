"""Learning, reward alignment and interruption regressions for the primary PPO."""
import contextlib
import io
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

from RL.train_ppo import (TorchResidualPolicy, CATEGORICAL_ACTION_SPACE, PPOMemory,
                          ppo_update, experiment_seeds, train, categorical_kl_from_logits,
                          collect_rollouts, evaluate_deterministic)
from RL.traffic_env import EnvConfig, MultiAgentTrafficEnv
from RL.transition import advance_agents, driving_reward

torch.set_num_threads(1)


def settings(**overrides):
    result = dict(seed=0, hidden_dim=8, lr=.001, calibration=None,
                  prefer_params="robust", collision_penalty=8., behavior_coef=0.,
                  behavior_shaping_coef=0., val_episodes=1,
                  test_episodes=1, curve=None, updates=2, anneal_lr=False, val_every=1,
                  episodes_per_update=1, gamma=.99, gae_lambda=.95, clip_coef=.2,
                  value_coef=.5, entropy_coef=0., ppo_epochs=1, minibatch_size=8,
                  target_kl=.02, log_every=1, num_agents=2, max_steps=2,
                  skip_test=True, save=None)
    result.update(overrides)
    return SimpleNamespace(**result)


class LearningTests(unittest.TestCase):
    def test_kl_does_not_treat_float32_underflow_as_lost_action_support(self):
        mask = torch.tensor([[True, True, False]])
        old = torch.tensor([[0., -100., -torch.inf]])
        new = torch.tensor([[0., -105., -torch.inf]])
        # The rare action has finite logits but its new float32 probability is zero.
        self.assertEqual(float(torch.softmax(new, -1)[0, 1]), 0.)
        value = categorical_kl_from_logits(old, new, mask)
        self.assertTrue(torch.isfinite(value).all())
        self.assertGreater(float(value), 0.)
        self.assertLess(float(value), 1e-40)
        self.assertEqual(float(categorical_kl_from_logits(old, old, mask)), 0.)
        # Real loss of support is still rejected, rather than hidden as underflow.
        with self.assertRaises(FloatingPointError):
            categorical_kl_from_logits(old, torch.tensor([[0., -torch.inf, -torch.inf]]), mask)

    def test_normalization_preserves_predictions(self):
        torch.manual_seed(3)
        policy = TorchResidualPolicy(32, hidden_dim=8)
        obs = torch.randn(12, 32)
        original = policy.forward(obs)[1].detach().clone()
        for returns in (torch.linspace(-200, 500, 12), torch.linspace(50, 90, 12)):
            policy.update_value_stats(returns)
            torch.testing.assert_close(policy.forward(obs)[1], original, atol=3e-5, rtol=1e-4)

    @staticmethod
    def memory(policy, count=256):
        obs = torch.zeros(count, 32)
        utility = torch.tensor([0., -.01, -.02]).expand(count, -1)
        mask = torch.ones(count, 3, dtype=torch.bool)
        with torch.no_grad():
            dist, value = policy.categorical_distribution(obs, utility, mask)
            actions = dist.sample()
        return PPOMemory(list(obs.numpy()), list(actions.numpy()), list(dist.log_prob(actions).numpy()),
                         list(policy.denormalize_value(value).numpy()),
                         list(torch.where(actions == 1, 1., -1.).numpy()),
                         [1.] * count, [0.] * count, [0.] * count, list(range(count)),
                         list(utility.numpy()), list(mask.numpy()))

    def test_ppo_learns_better_action_than_frozen_prior_on_controlled_task(self):
        torch.manual_seed(42)
        np.random.seed(42)
        policy = TorchResidualPolicy(32, hidden_dim=8, action_dim=3,
                                     action_space=CATEGORICAL_ACTION_SPACE,
                                     candidate_logit_scale=.5, candidate_temperature=.1)
        optimizer = torch.optim.Adam(policy.parameters(), lr=.01)
        for _ in range(12):
            ppo_update(policy, optimizer, self.memory(policy), gamma=0., gae_lambda=0.,
                       clip_coef=.2, value_coef=.5, entropy_coef=0., epochs=4,
                       minibatch_size=64, target_kl=.03)
        delta, _ = policy.act(np.zeros(32, dtype=np.float32))
        self.assertEqual(int(np.argmax(np.array([0., -.01, -.02]) + delta)), 1)
        dist, _ = policy.categorical_distribution(torch.zeros(32), torch.tensor([0., -.01, -.02]),
                                                  torch.ones(3, dtype=torch.bool))
        self.assertGreater(float(dist.probs[1]), .9)

    def test_kl_stops_before_another_optimizer_step(self):
        policy = TorchResidualPolicy(32, hidden_dim=8, action_dim=3,
                                     action_space=CATEGORICAL_ACTION_SPACE)
        memory = self.memory(policy, 64)
        optimizer = torch.optim.Adam(policy.parameters(), lr=.001)
        def jump():
            with torch.no_grad():
                policy.actor[4].bias.copy_(torch.tensor([10., -10., -10.]))
        with patch.object(optimizer, "step", side_effect=jump) as step:
            stats = ppo_update(policy, optimizer, memory, gamma=0., gae_lambda=0.,
                               clip_coef=.2, value_coef=.5, entropy_coef=0., epochs=4,
                               minibatch_size=16, target_kl=.02)
        self.assertEqual(step.call_count, 1)
        self.assertEqual(stats["early_stop"], 1.)


class RewardTests(unittest.TestCase):
    def test_waiting_cannot_earn_positive_reward(self):
        env = MultiAgentTrafficEnv(EnvConfig(num_agents=1), seed=42)
        env.reset()
        agent = env.agents[0]
        station = env.corridor.project(agent.pos)[0]
        agent.vel[:] = 0.
        reward = driving_reward(env.agents, 0, env.corridor, env.config.sim_config,
                                env._dest_s[0], (0., 0.), previous_station=station)
        self.assertLessEqual(reward, 0.)

    def test_proximity_cost_does_not_multiply_with_neighbor_count(self):
        env = MultiAgentTrafficEnv(EnvConfig(num_agents=2), seed=42)
        env.reset()
        weights = dict(progress=1., time=0., safety=1., smooth=0.)
        station = env.corridor.project(env.agents[0].pos)[0]
        def reward(agents):
            return driving_reward(agents, 0, env.corridor, env.config.sim_config,
                                  env._dest_s[0], (0., 0.), weights, leftover_coef=0.,
                                  previous_station=station - 4.)
        with patch("RL.transition.footprint_surface_gap", return_value=1.):
            self.assertEqual(reward(env.agents), reward([env.agents[0]] + [env.agents[1]] * 6))

    def test_action_replaced_by_filter_is_not_reported_as_executed_intervention(self):
        from RL.candidate_policy import CandidateIndex
        env = MultiAgentTrafficEnv(EnvConfig(num_agents=1), seed=42)
        env.reset()
        context = env.candidate_context(0)
        alternative = next(int(i) for i in np.flatnonzero(context.mask) if i != context.prior_index)
        with patch("RL.traffic_env.sanitize_control_command", return_value=(0., 0.)), \
             patch("RL.transition.sanitize_control_command", return_value=(0., 0.)):
            _, _, _, info = env.step([CandidateIndex(alternative)])
        self.assertEqual(info["candidate_flips"], 1)
        self.assertEqual(info["control_flips"], 0)

    def test_contact_on_arrival_is_negative_even_with_arrival_bonus(self):
        env = MultiAgentTrafficEnv(EnvConfig(num_agents=2), seed=0)
        env.reset()
        for agent in env.agents:
            agent.pos = env.agents[0].pos.copy()
            agent.vel[:] = 0.
        station = env.corridor.project(env.agents[0].pos)[0]
        result = advance_agents(env.agents, [(0., 0.)]*2, env.corridor, env.config.sim_config,
                                [station]*2, collision_penalty=8.)
        self.assertEqual(result.arrived, {0, 1})
        self.assertTrue(all(reward <= 0 for reward in result.rewards))

    def test_collision_event_penalty_charges_onset_not_continuation(self):
        env = MultiAgentTrafficEnv(EnvConfig(num_agents=2), seed=0)
        env.reset()
        for agent in env.agents:
            agent.pos = env.agents[0].pos.copy()
            agent.vel[:] = 0.
            agent.reached_destination = False
        station = env.corridor.project(env.agents[0].pos)[0] + 50.
        first = advance_agents(
            env.agents, [(0., 0.)] * 2, env.corridor, env.config.sim_config, [station] * 2,
            leftover_coef=0., arrival_bonus=0., collision_penalty=1., collision_event_penalty=10.,
        )
        self.assertEqual(first.collision_pairs, {(0, 1)})
        onset = list(first.rewards)
        second = advance_agents(
            env.agents, [(0., 0.)] * 2, env.corridor, env.config.sim_config, [station] * 2,
            leftover_coef=0., arrival_bonus=0., collision_penalty=1., collision_event_penalty=10.,
            previous_collision_pairs=first.collision_pairs,
        )
        self.assertEqual(second.collision_pairs, {(0, 1)})
        for reward in second.rewards:
            self.assertEqual(reward, 0.)

    def test_progress_uses_result_of_action(self):
        results = []
        for accel in (-3., 3.):
            env = MultiAgentTrafficEnv(EnvConfig(num_agents=1), seed=42)
            env.reset()
            result = advance_agents(env.agents, [(accel, 0.)], env.corridor, env.config.sim_config,
                                    env._dest_s, reward_weights={"progress":1., "safety":0., "smooth":0.},
                                    leftover_coef=0.)
            results.append(result.rewards[0])
        self.assertGreater(results[1], results[0])
        self.assertLessEqual(results[1], 1.)


class ExperimentTests(unittest.TestCase):
    def test_reported_returns_use_episode_time_and_include_arrival_reward(self):
        args = settings(num_agents=2, max_steps=2, gamma=.5)
        env = MultiAgentTrafficEnv(EnvConfig(num_agents=2, max_steps=2), seed=42)
        policy = TorchResidualPolicy(32, hidden_dim=8, action_dim=63,
                                     action_space=CATEGORICAL_ACTION_SPACE)
        counter = 0
        def step(agents, *args, **kwargs):
            nonlocal counter
            result = advance_agents(agents, *args, **kwargs)
            counter += 1
            agents[0].reached_destination = True
            result.rewards = [3., 5.] if counter == 1 else [0., 7.]
            return result
        with patch("RL.traffic_env.advance_agents", side_effect=step):
            *_, stats = collect_rollouts(env, policy, 1, gamma=.5)
        self.assertEqual(stats["mean_return"], 7.5)
        self.assertEqual(stats["discounted_return"], 5.75)
        counter = 0
        with patch("RL.train_ppo.make_env", return_value=env), \
             patch("RL.traffic_env.advance_agents", side_effect=step):
            evaluated = evaluate_deterministic(args, None, [42])
        self.assertEqual(evaluated["mean_return"], 7.5)
        self.assertEqual(evaluated["discounted_return"], 5.75)

    def test_training_evaluation_uses_identical_benchmark_metrics(self):
        from Baselines.metrics import rollout_metrics
        from Baselines.runner import rollout
        from Baselines.scenario import build_scenario
        from Baselines.utility_prior import UtilityPriorController
        from RL.calibration_io import DEFAULT_CALIBRATION_PATH
        from RL.train_ppo import evaluate_deterministic
        args = settings(num_agents=3, max_steps=3, calibration=DEFAULT_CALIBRATION_PATH)
        measured = evaluate_deterministic(args, None, [42])["episode_results"][0]
        benchmark = rollout_metrics(rollout(build_scenario(42, num_agents=3, max_steps=3),
                                            UtilityPriorController()))
        for key, value in benchmark.items():
            if key.startswith("wall_time") or key in ("model", "selection_status", "training_revision"):
                continue
            with self.subTest(metric=key):
                np.testing.assert_allclose(measured[key], value, atol=1e-10)

    def test_all_episode_seeds_are_disjoint(self):
        training, validation, test = experiment_seeds(settings(episodes_per_update=3))
        self.assertEqual(training, [[10001000, 10001001, 10001002], [10002000, 10002001, 10002002]])
        self.assertFalse(set(validation) & set(test))
        experiment_seeds(settings(updates=2000))  # larger budgets stay outside evaluation blocks
        for config in (settings(validation_seed_start=10001001, episodes_per_update=2),
                       settings(test_seed_start=10002000),
                       settings(validation_seed_start=810000)):
            with self.assertRaises(ValueError):
                experiment_seeds(config)

    def test_resume_replays_pending_validation_without_retraining_and_matches_full_run(self):
        stats = dict(metric=0., collisions=0., collision_events=0., arrival_rate=1.,
                     residual_usage=0., control_flip_rate=0.)
        with tempfile.TemporaryDirectory() as tmp:
            args = settings(save=Path(tmp)/"interrupted.pt")
            with patch("RL.train_ppo._persist_curve"), contextlib.redirect_stdout(io.StringIO()):
                with patch("RL.train_ppo.evaluate_deterministic", side_effect=[stats, RuntimeError("interrupted")]):
                    with self.assertRaisesRegex(RuntimeError, "interrupted"):
                        train(args)
                resume_path = args.save.with_suffix(".resume.pt")
                state = torch.load(resume_path, weights_only=False)
                self.assertTrue(state["pending_evaluation"])
                self.assertEqual(state["update"], 1)
                self.assertTrue(args.save.exists())
                args.resume = resume_path
                with patch("RL.train_ppo.evaluate_deterministic", return_value=stats) as evaluate:
                    train(args)
                    self.assertEqual(evaluate.call_count, 2)
                recovered = torch.load(resume_path, weights_only=False)
                del args.resume
                args.save = Path(tmp)/"uninterrupted.pt"
                with patch("RL.train_ppo.evaluate_deterministic", return_value=stats):
                    train(args)
                full = torch.load(args.save.with_suffix(".resume.pt"), weights_only=False)
        self.assertTrue(recovered["complete"])
        self.assertEqual(recovered["update"], 2)
        for key, value in full["state_dict"].items():
            torch.testing.assert_close(value, recovered["state_dict"][key], rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
