"""Fairness regressions: information, moving obstacles, selection and real CLI budgets."""
import copy
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

from Baselines.scenario import build_scenario
from RL.decision import neighbor_indices, trajectory_obb_free, select_best_candidate
from RL.experiment_protocol import TrainingBudget, selection_key, SELECTION_RULE

ROOT = Path(__file__).resolve().parents[1]


class DecisionInformationTests(unittest.TestCase):
    def test_hidden_agent_cannot_change_utility_mask_or_predictor(self):
        from Baselines.local_frame import predict_neighbours
        from Baselines.discrete_action import feasible_action_mask
        from Baselines.utility_prior import UtilityPriorController
        from RL.candidate_policy import candidate_context
        scenario = build_scenario(8, num_agents=8, max_steps=2)
        agents = scenario.spawn_agents()
        sim = scenario.sim_config
        sim['perception_radius'] = 1000.
        selected = neighbor_indices(agents, 0, sim)
        hidden = next(i for i in range(1, len(agents)) if i not in selected)
        changed = copy.deepcopy(agents)
        changed[hidden].vel = np.array([100., -50.])
        changed[hidden].heading_angle += .8
        prior = UtilityPriorController()
        a = candidate_context(0, agents, prior.params, sim)
        b = candidate_context(0, changed, prior.params, sim)
        np.testing.assert_array_equal(a.mask, b.mask)
        np.testing.assert_array_equal(a.utilities, b.utilities)
        np.testing.assert_array_equal(feasible_action_mask(0, agents[0], agents, scenario), a.mask)
        ca = select_best_candidate(0, agents[0], agents, prior.params, sim)
        cb = select_best_candidate(0, changed[0], changed, prior.params, sim)
        self.assertEqual(ca['accel_longitudinal'], cb['accel_longitudinal'])
        self.assertEqual(ca['steering_angle'], cb['steering_angle'])
        predictions = predict_neighbours(agents, 0, 4, .5, sim_config=sim)
        np.testing.assert_allclose(predictions[:, 0], [agents[i].pos for i in selected])
        np.testing.assert_array_equal(predictions, predict_neighbours(changed, 0, 4, .5, sim_config=sim))
        agents[selected[0]].reached_destination = True
        self.assertNotIn(selected[0], neighbor_indices(agents, 0, sim))

    def test_crossing_between_endpoints_uses_velocity_and_rotated_footprints(self):
        ego = SimpleNamespace(pos=np.array([0., 0.]), vel=np.array([10., 0.]), heading=0., reached_destination=False)
        obstacle = SimpleNamespace(pos=np.array([5., -5.]), vel=np.array([0., 10.]), heading=np.pi/2, reached_destination=False)
        sim = dict(dt=1., conflict_horizon=1.5, conflict_substeps=6,
                   vehicle_length=2., vehicle_width=1., max_neighbors=6, perception_radius=60.)
        positions = np.array([[[0., 0.], [10., 0.]]])
        headings = np.zeros((1, 2))
        # Clear at t=0 and t=1, but crossing at t=.5 must be rejected.
        self.assertFalse(trajectory_obb_free(positions, headings, [0., 1.], [ego, obstacle], 0, sim)[0])
        obstacle.vel[:] = 0.
        self.assertTrue(trajectory_obb_free(positions, headings, [0., 1.], [ego, obstacle], 0, sim)[0])
        obstacle.vel[:] = [0., 10.]
        sim['perception_radius'] = 1.
        self.assertTrue(trajectory_obb_free(positions, headings, [0., 1.], [ego, obstacle], 0, sim)[0])

    def test_mppi_rejects_unsafe_average_of_feasible_samples(self):
        from Baselines.mppi import MPPIController
        from Baselines.dynamics import simulate_bicycle_batch
        scenario = build_scenario(4, num_agents=1, max_steps=1)
        controller = MPPIController(samples=8, horizon=2)
        controller.reset(scenario)
        sampled = []
        def capture(*args, **kwargs):
            sampled.append(np.stack([args[3], args[4]], axis=-1))
            return simulate_bicycle_batch(*args, **kwargs)
        # Model a nonconvex free set: samples safe, their weighted mean unsafe.
        with patch('Baselines.mppi.simulate_bicycle_batch', side_effect=capture), \
             patch('RL.decision.control_feasible', return_value=True), \
             patch('RL.decision.trajectory_obb_free', side_effect=[np.ones(8, bool), np.zeros(1, bool)]):
            result = controller.compute_controls(scenario.spawn_agents(), scenario, 0)
        self.assertEqual(len(sampled), 2)
        self.assertTrue(any(np.allclose(result[0], control[0]) for control in sampled[0]))

    def test_interventions_include_execution_clipping(self):
        from Baselines.runner import rollout
        class InvalidCommand:
            name = 'probe'
            def reset(self, scenario): pass
            def compute_controls(self, agents, scenario, step): return [(999., 0.)]*len(agents)
        result = rollout(build_scenario(2, num_agents=2, max_steps=1), InvalidCommand())
        self.assertEqual(result.extra['shield_decisions'], 2)
        self.assertEqual(result.extra['shield_interventions'], 2)
        self.assertEqual(result.extra['shield_intervention_rate'], 1.)
        self.assertEqual(result.extra['proposed_controls'][0][0][0], 999.)


class SelectionAndBudgetTests(unittest.TestCase):
    def test_safety_before_pdms_and_honest_failed_selection(self):
        prior = dict(collision_events=0., collisions=0., offroad_rate=0., arrival_rate=.8,
                     metric=10., closed_loop_score=.8)
        safe = dict(prior, closed_loop_score=.7)
        fast = dict(prior, arrival_rate=1., metric=0., closed_loop_score=1., collision_events=1., collisions=1.)
        offroad = dict(prior, offroad_rate=.1, closed_loop_score=1.)
        self.assertLess(selection_key(safe, prior), selection_key(fast, prior))
        self.assertLess(selection_key(safe, prior), selection_key(offroad, prior))
        self.assertLess(selection_key(fast, prior), selection_key(dict(fast, collision_events=2.), prior))
        self.assertEqual(selection_key(fast, prior)[0], 1)
        self.assertEqual(selection_key(dict(fast, collisions=float('nan')), prior)[0], 1)

    def test_step_validation_at_final_update_and_resumed_budget(self):
        args = SimpleNamespace(total_env_steps=5, val_every_env_steps=20, val_every=1, updates=2)
        budget = TrainingBudget(args)
        budget.add(3, 6)
        self.assertEqual(budget.remaining(100), 2)
        self.assertFalse(budget.validation_due(args, 1))
        self.assertTrue(budget.validation_due(args, 2))
        resumed = TrainingBudget(args, budget.state())
        resumed.add(2, 3)
        self.assertTrue(resumed.exhausted)
        self.assertEqual(resumed.remaining(2), 0)
        self.assertEqual(resumed.state()['active_agent_transitions'], 9)

    def test_strict_benchmark_rejects_legacy_and_mismatched_validation(self):
        from Baselines.benchmark import experiment_manifest
        scenario = build_scenario(0, num_agents=2, max_steps=1)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            args = SimpleNamespace(models=['mappo', 'happo'], train_seeds=None,
                checkpoint_dir=root, residual_checkpoint=None, pure_rl_checkpoint=None,
                no_obb_safety_filter=False, require_matched_protocol=True)
            common = dict(selection_rule=SELECTION_RULE, decision_protocol_version=1,
                          val_seeds=[910000], validation_config={'num_agents': 2})
            torch.save({}, root/'mappo_policy.pt')
            with self.assertRaisesRegex(ValueError, 'legacy'):
                experiment_manifest(args, [scenario])
            torch.save(common, root/'mappo_policy.pt')
            torch.save(dict(common, validation_config={'num_agents': 3}), root/'happo_policy.pt')
            with self.assertRaisesRegex(ValueError, 'different validation conditions'):
                experiment_manifest(args, [scenario])
            torch.save(common, root/'happo_policy.pt')
            self.assertTrue(all(row['matched_protocol'] for row in experiment_manifest(args, [scenario])['checkpoints']))

    def test_all_online_training_clis_honor_exact_interaction_cap(self):
        modules = [('RL.train_ppo', []), ('Baselines.train_direct_discrete_rl', []),
                   ('Baselines.train_pure_rl', []),
                   *[('Baselines.train_marl', ['--algo', algo, '--critic-epochs', '1'])
                     for algo in ('mappo', 'happo', 'hatrpo')]]
        contracts = []
        with tempfile.TemporaryDirectory() as tmp:
            for i, (module, extra) in enumerate(modules):
                with self.subTest(module=module, extra=extra):
                    output = Path(tmp)/str(i)/'policy.pt'
                    cmd = [sys.executable, '-c',
                           'import sys,runpy,torch; torch.set_num_threads(1); m=sys.argv.pop(1); runpy.run_module(m,run_name="__main__")',
                           module, '--updates', '3', '--episodes-per-update', '2', '--num-agents', '2',
                           '--max-steps', '3', '--total-env-steps', '5', '--val-every-env-steps', '4',
                           '--val-episodes', '1', '--test-episodes', '1', '--skip-test',
                           '--hidden-dim', '8', '--ppo-epochs', '1', '--minibatch-size', '8',
                           '--save', str(output), *extra]
                    result = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, timeout=60)
                    self.assertEqual(result.returncode, 0, result.stdout+'\n'+result.stderr)
                    report = json.loads(output.with_suffix('.summary.json').read_text())
                    self.assertEqual(report['budget']['environment_steps'], 5)
                    self.assertGreater(report['budget']['optimizer_steps'], 0)
                    self.assertEqual(report['stopping_reason'], 'environment_budget')
                    self.assertEqual(report['selection_rule'], SELECTION_RULE)
                    contracts.append(report['validation_config'])
            self.assertTrue(all(contract == contracts[0] for contract in contracts))


if __name__ == '__main__':
    unittest.main()
