"""Regression coverage for the September RL / benchmark review findings."""

import argparse
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from Baselines.discrete_action import feasible_action_mask
from Baselines.registry import build_controller, resolve_train_seeds, LEARNED_CHECKPOINTS
from Baselines.runner import rollout
from Baselines.scenario import build_scenario
from Baselines.utility_prior import UtilityPriorController
from RL.calibration_io import load_base_params
from RL.protocol import validate_checkpoint
from RL.traffic_env import EnvConfig, MultiAgentTrafficEnv
from RL.transition import advance_agents
from utility_model import kinematic_bicycle_rollout, sanitize_control_command

torch.set_num_threads(1)


class Coast:
    name = "coast"

    def reset(self, scenario):
        pass

    def compute_controls(self, agents, scenario, step):
        return [(0.0, 0.0)] * len(agents)


class SharedTransitionTest(unittest.TestCase):
    def test_environment_and_benchmark_match_in_interacting_scenario(self):
        cfg = EnvConfig(num_agents=10, max_steps=3, base_params=load_base_params())
        env = MultiAgentTrafficEnv(cfg, seed=1)
        env.reset()
        positions = [np.array([a.pos.copy() for a in env.agents])]
        for _ in range(3):
            env.step()
            positions.append(np.array([a.pos.copy() for a in env.agents]))
        result = rollout(build_scenario(1, num_agents=10, max_steps=3), UtilityPriorController())
        np.testing.assert_allclose(result.positions, positions, atol=1e-10)

    def test_arrival_flag_from_integrator_is_recorded(self):
        scenario = build_scenario(0, num_agents=1, max_steps=1)
        agent = scenario.spawn_agents()[0]
        move = kinematic_bicycle_rollout(agent.pos, agent.heading, agent.speed, 0, 0,
                                         scenario.dt, scenario.sim_config)
        scenario.agents[0].dest = move["pos"]
        scenario.agents[0].dest_s = scenario.corridor.project(move["pos"])[0]
        result = rollout(scenario, Coast())
        self.assertEqual(result.arrival_step.tolist(), [1])
        self.assertFalse(result.active[-1, 0])
        from Baselines.metrics import rollout_metrics
        metrics = rollout_metrics(result)
        self.assertGreater(metrics["mean_speed_mps"], 0.)
        self.assertEqual(metrics["mean_capped_travel_time_s"], scenario.dt)
        self.assertTrue(np.isfinite(metrics["rms_jerk"]))

    def test_contacts_on_arrival_step_are_not_hidden(self):
        scenario = build_scenario(0, num_agents=2, max_steps=1, obb_safety_filter=False)
        agents = scenario.spawn_agents()
        for a in agents:
            a.pos = agents[0].pos.copy()
            a.vel[:] = 0.0
            a.dest = a.pos.copy()
        station = scenario.corridor.project(agents[0].pos)[0]
        result = advance_agents(agents, [(0.0, 0.0)] * 2, scenario.corridor,
                                scenario.sim_config, [station, station], collision_penalty=8.0)
        self.assertEqual(result.arrived, {0, 1})
        self.assertEqual(result.collision_pairs, {(0, 1)})

    def test_baseline_and_residual_reward_match(self):
        cfg = EnvConfig(num_agents=3, max_steps=1, base_params=load_base_params(), collision_penalty=8.0)
        env = MultiAgentTrafficEnv(cfg, seed=1)
        env.reset()
        scenario = build_scenario(1, num_agents=3, max_steps=1)
        agents = scenario.spawn_agents()
        controller = UtilityPriorController()
        controller.reset(scenario)
        transition = advance_agents(agents, controller.compute_controls(agents, scenario, 0),
                                    scenario.corridor, scenario.sim_config,
                                    [a.dest_s for a in scenario.agents], collision_penalty=8.0)
        _, rewards, _, _ = env.step()
        np.testing.assert_allclose(rewards, transition.rewards)


class SafetyAndPlannerTest(unittest.TestCase):
    def test_all_three_planners_finish_from_two_metres_short(self):
        for name in ("dwa", "mppi", "frenet"):
            with self.subTest(model=name):
                scenario = build_scenario(0, num_agents=1, max_steps=6)
                scenario.agents[0].vel[:] = 0.0
                scenario.agents[0].dest_s = scenario.agents[0].start_s + 2.0
                scenario.agents[0].dest = scenario.corridor.xy_from_frenet(scenario.agents[0].dest_s, 0)[0]
                result = rollout(scenario, build_controller(name))
                self.assertGreaterEqual(result.arrival_step[0], 1)

    def test_disabled_filter_does_not_mask_direct_actions(self):
        scenario = build_scenario(0, num_agents=2, max_steps=1, obb_safety_filter=False)
        # This test isolates the car-to-car filter. Road containment has its own mask.
        scenario.sim_config["boundary_safety_filter"] = False
        agents = scenario.spawn_agents()
        for i, agent in enumerate(agents):
            agent.pos, tangent = scenario.corridor.xy_from_frenet(40 + 5 * i, 0)
            agent.vel = tangent * (8 if i == 0 else 2)
            agent.heading_angle = float(np.arctan2(tangent[1], tangent[0]))
        self.assertTrue(feasible_action_mask(0, agents[0], agents, scenario).all())
        scenario.sim_config["obb_safety_filter"] = True
        self.assertFalse(feasible_action_mask(0, agents[0], agents, scenario).all())

    def test_infeasible_filter_keeps_braking(self):
        scenario = build_scenario(0, num_agents=1, max_steps=1)
        agents = scenario.spawn_agents()
        with patch("utility_model.control_command_obb_conflict", return_value=True):
            control = sanitize_control_command(0, agents[0], agents, (1, 0), scenario.sim_config)
        self.assertLess(control[0], 0)

    def test_existing_overlap_is_zero_ttc_even_without_relative_motion(self):
        from Baselines.metrics import _pairwise_ttc
        _, ttc, unsafe, pairs = _pairwise_ttc(np.zeros((2, 2, 2)), np.zeros((2, 2)),
                                             np.ones(2, dtype=bool), 1.0)
        self.assertEqual((ttc, unsafe, pairs), (0.0, 1, 1))


class TrainingBufferTest(unittest.TestCase):
    def test_marl_distinguishes_arrivals_from_horizon_truncations(self):
        from Baselines import train_marl
        from Baselines.marl import MARLPolicy

        for horizon, arrivals in [(3, [True, True]), (1, [True, True]), (1, [True, False])]:
            with self.subTest(horizon=horizon, arrivals=arrivals):
                scenario = build_scenario(0, num_agents=2, max_steps=horizon)
                policy = MARLPolicy(32, 2, algo="mappo", hidden_dim=8)
                advance = train_marl.advance_agents

                def step(agents, *args, **kwargs):
                    result = advance(agents, *args, **kwargs)
                    for agent, arrived in zip(agents, arrivals):
                        agent.reached_destination = arrived
                    return result

                with patch.object(train_marl, "advance_agents", side_effect=step):
                    episode = train_marl.collect_episode(scenario, policy)
                expected = np.logical_not(arrivals)
                np.testing.assert_array_equal(episode.truncated, expected)
                episode.rewards[:] = 1.0
                episode.values[:] = 0.0
                # Stale nonzero values must never bootstrap a true terminal.
                episode.bootstrap_values[:] = 5.0
                with patch.object(train_marl, "compute_gae", wraps=train_marl.compute_gae) as gae:
                    _, returns = train_marl.episode_gae(episode, .9, .95)
                np.testing.assert_allclose(returns[-1], 1.0 + .9 * 5.0 * expected)
                for call, truncated in zip(gae.call_args_list, expected):
                    self.assertEqual(bool(call.kwargs["timeouts"][-1]), truncated)
                # A true timeout remains a timeout when its predicted value is zero.
                episode.bootstrap_values[:] = 0.0
                with patch.object(train_marl, "compute_gae", wraps=train_marl.compute_gae) as gae:
                    train_marl.episode_gae(episode, .9, .95)
                for call, truncated in zip(gae.call_args_list, expected):
                    self.assertEqual(bool(call.kwargs["timeouts"][-1]), truncated)

    def test_baselines_store_separate_trajectories_and_bootstrap_timeouts(self):
        from Baselines import train_direct_discrete_rl, train_pure_rl
        from Baselines.direct_discrete_rl import DirectDiscretePolicy
        from Baselines.pure_rl import PureRLPolicy
        from RL.train_ppo import compute_gae_by_trajectory
        scenario = build_scenario(0, num_agents=2, max_steps=2, obb_safety_filter=False)
        args = argparse.Namespace(gamma=.99, gae_lambda=.95, ppo_epochs=1, minibatch_size=8,
                                  clip_coef=.2, value_coef=.5, entropy_coef=0.)
        for module, policy in [(train_direct_discrete_rl, DirectDiscretePolicy(32, 63, hidden_dim=8)),
                               (train_pure_rl, PureRLPolicy(32, hidden_dim=8))]:
            with self.subTest(trainer=module.__name__):
                memory = module.PPOMemory()
                for _ in range(2):
                    module.run_episode(scenario, policy, memory, 8.0)
                self.assertEqual(memory.traj_ids, [0, 1, 0, 1, 2, 3, 2, 3])
                self.assertEqual(memory.timeouts, [0, 0, 1, 1, 0, 0, 1, 1])
                self.assertFalse(any(memory.dones))
                self.assertTrue(np.isfinite(memory.bootstrap_values).all())
                with patch.object(module, "compute_gae_by_trajectory", wraps=compute_gae_by_trajectory) as gae:
                    stats = module.ppo_update(policy, torch.optim.Adam(policy.parameters()), memory, args)
                self.assertTrue(gae.called)
                np.testing.assert_array_equal(gae.call_args[0][2], memory.traj_ids)
                self.assertTrue(np.isfinite(list(stats.values())).all())

    def test_hatrpo_old_distribution_does_not_alias_live_parameters(self):
        from Baselines.marl import MARLPolicy
        from Baselines import train_marl
        policy = MARLPolicy(32, 2, algo="hatrpo", hidden_dim=8)
        scenario = build_scenario(0, num_agents=2, max_steps=3, obb_safety_filter=False)
        episode = train_marl.collect_episode(scenario, policy, 8.0)
        batch = train_marl.build_batch([episode], .99, .95)
        original = train_marl._gaussian_kl
        calls = []

        def checked(actor, obs, old_mean, old_log_std, mask):
            self.assertNotEqual(old_log_std.data_ptr(), actor.log_std.data_ptr())
            calls.append(True)
            return original(actor, obs, old_mean, old_log_std, mask)

        args = argparse.Namespace(cg_damping=.1, cg_iterations=2, max_kl=torch.tensor(.01),
                                  line_search_decay=.5, line_search_steps=3, accept_ratio=.1)
        with patch.object(train_marl, "_gaussian_kl", side_effect=checked):
            train_marl.update_actors_hatrpo(policy, batch, args)
        self.assertTrue(calls)


class CheckpointAndAblationTest(unittest.TestCase):
    def test_sequential_policy_rejects_agent_count_reuse(self):
        from Baselines.marl import MARLController, MARLPolicy
        policy = MARLPolicy(obs_dim=32, num_agents=2, algo="happo", hidden_dim=8)
        controller = MARLController(algo="happo", policy=policy)
        with self.assertRaisesRegex(ValueError, "agent-specific"):
            controller.reset(build_scenario(0, num_agents=3, max_steps=1))

    def test_bootstrap_table_preserves_asymmetric_endpoints(self):
        from Baselines.stats import MetricSummary
        summary = MetricSummary("model", "metric", 2., 1., .5, 4., 3, 30, "train_seed")
        self.assertEqual(summary.format(2), "2.00 [0.50, 4.00]")

    def test_old_protocol_and_observations_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "predates"):
            validate_checkpoint({"obs_dim": 32}, 32, "old.pt")
        with self.assertRaisesRegex(ValueError, "observation"):
            validate_checkpoint({"protocol_version": 3, "obs_dim": 31}, 32, "old_obs.pt")

    def test_partial_seed_sets_are_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp) / "model.pt"
            (Path(tmp) / "model_seed0.pt").touch()
            with self.assertRaisesRegex(FileNotFoundError, "seed1"):
                resolve_train_seeds("residual_marl", [0, 1], base)

    def test_ablation_and_nominal_paths_are_distinct(self):
        self.assertEqual(LEARNED_CHECKPOINTS["residual_param"], LEARNED_CHECKPOINTS["residual_weights_only"])
        self.assertEqual(LEARNED_CHECKPOINTS["residual_marl"], LEARNED_CHECKPOINTS["residual_no_gate"])
        self.assertNotEqual(LEARNED_CHECKPOINTS["residual_marl"], LEARNED_CHECKPOINTS["residual_param"])
        self.assertNotEqual(LEARNED_CHECKPOINTS["residual_marl"], LEARNED_CHECKPOINTS["residual_nominal"])

    def test_parameter_ablation_rejects_candidate_checkpoint(self):
        from RL.train_ppo import TorchResidualPolicy
        policy = TorchResidualPolicy(32, hidden_dim=8)
        blob = {"protocol_version": 3, "state_dict": policy.state_dict(), "obs_dim": 32,
                "hidden_dim": 8, "residual_mode": "candidate_logits", "action_dim": 63,
                "action_space": "squashed_tanh", "param_gauge": "logit_simplex"}
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "candidate.pt"
            torch.save(blob, path)
            controller = build_controller("residual_weights_only", checkpoint=path)
            with self.assertRaisesRegex(ValueError, "param_delta"):
                controller.reset(build_scenario(0, num_agents=1, max_steps=1))


class DiagnosticAndWrapperTest(unittest.TestCase):
    def test_histogram_of_nearly_constant_controls_has_finite_density(self):
        from Baselines.plots import histogram_bins
        values = np.array([4.0, np.nextafter(4.0, 5.0)])
        edges = histogram_bins(values)
        self.assertTrue((np.diff(edges) > 0).all())
        density, _ = np.histogram(values, bins=edges, density=True)
        self.assertTrue(np.isfinite(density).all())

    def test_metric_plot_handles_entirely_missing_ttc(self):
        import pandas as pd
        from Baselines.plots import plot_metric_bars
        frame = pd.DataFrame({"model": ["utility_pt", "mappo"],
                              "min_ttc_s": [np.nan, np.nan],
                              "arrival_rate": [0.0, 1.0]})
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "metrics.png"
            plot_metric_bars(frame, path)
            self.assertGreater(path.stat().st_size, 0)


    def test_candidate_heatmap_uses_emitted_coordinates(self):
        from RL.visualize import residual_series_matrix
        keys, values = residual_series_matrix([{}, {"c0": .5, "c10": -.7, "c2": .1}])
        self.assertEqual(keys, ["c0", "c2", "c10"])
        np.testing.assert_allclose(values[:, 0], [.5, .1, .7])


if __name__ == "__main__":
    unittest.main()
