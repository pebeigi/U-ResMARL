"""Bounded site regressions. Run through run_module.py for isolated activation."""
import copy
import unittest
import numpy as np
import torch
from shapely.geometry import LineString

from config import CALIBRATION
from RL.routing import agent_route, agent_station, arrival_reached
from RL.calibration_io import load_base_params
from RL.traffic_env import EnvConfig, MultiAgentTrafficEnv
from Baselines.scenario import build_scenario
from Baselines.runner import rollout
from Baselines.utility_prior import UtilityPriorController


class TGSIMTests(unittest.TestCase):
    def test_recorded_sampling_covers_configured_evaluation_seeds(self):
        from config import VAL_EPISODES, TEST_EPISODES, NUM_AGENTS
        seeds = [0, 1, 2, 42] + list(range(910000, 910000 + VAL_EPISODES)) + list(range(810000, 810000 + TEST_EPISODES))
        for seed in seeds:
            with self.subTest(seed=seed):
                scene = build_scenario(seed, num_agents=NUM_AGENTS, max_steps=2)
                self.assertEqual(len(scene.agents), NUM_AGENTS)
                for agent in scene.agents:
                    self.assertTrue(np.isfinite(agent.pos).all())
                    self.assertGreater(np.linalg.norm(agent.dest - agent.pos), 4.99)

    def test_recorded_spawns_preserve_poses_goals_and_disjoint_splits(self):
        from recorded_spawns import catalogue
        from RL.boundary import footprint_clearance
        from RL.spawn_safety import has_straight_braking_backup
        groups, tracks, _ = catalogue()
        seen = []
        for seed, split in ((3, 'train'), (910000, 'validation'), (810000, 'test')):
            with self.subTest(split=split):
                scene = build_scenario(seed, num_agents=6, max_steps=2)
                repeated = build_scenario(seed, num_agents=6, max_steps=2)
                meta = scene.sim_config['recorded_initialization']
                self.assertEqual(meta['split'], split)
                group = next(g for g in groups[split] if np.isclose(g.time.iloc[0], meta['time_s']))
                by_id = group.set_index('id')
                ids = {a.agent_id for a in scene.agents}
                self.assertTrue(all(not (ids & previous) for previous in seen))
                seen.append(ids)
                for a, b in zip(scene.agents, repeated.agents):
                    row = by_id.loc[a.agent_id]
                    self.assertEqual(a.agent_id, b.agent_id)
                    np.testing.assert_array_equal(a.pos, b.pos)
                    np.testing.assert_array_equal(a.dest, b.dest)
                    np.testing.assert_array_equal(a.pos, [row.xloc_kf, row.yloc_kf])
                    np.testing.assert_array_equal(a.vel, [row.vx, row.vy])
                    self.assertEqual(a.heading, row.heading)
                    future = tracks[a.agent_id].query('time > @row.time')
                    self.assertTrue(np.isclose(future[['xloc_kf', 'yloc_kf']].to_numpy(), a.dest).all(axis=1).any())
                    self.assertGreaterEqual(footprint_clearance(scene.corridor, a.pos, a.heading,
                        scene.vehicle_length, scene.vehicle_width), scene.sim_config['boundary_margin'])
                self.assertTrue(has_straight_braking_backup(scene.spawn_agents(), scene.sim_config, scene.corridor))

    def test_new_baseline_dynamics_match_site(self):
        from tools.verify_site_learning import check_site_dynamics
        check_site_dynamics()

    def test_scene_order_does_not_change_observations_or_goal_distances(self):
        from Baselines.dynamics import observation
        first = build_scenario(3, num_agents=3, max_steps=2)
        agents = first.spawn_agents()
        before = [observation(agents, i, first) for i in range(3)]
        stations = [agent_station(first.corridor, a) for a in agents]
        other = build_scenario(7, num_agents=6, max_steps=2)
        for a in other.spawn_agents(): agent_route(other.corridor, a)
        np.testing.assert_array_equal(stations, [agent_station(first.corridor, a) for a in agents])
        for i in range(3): np.testing.assert_array_equal(before[i], observation(agents, i, first))
        with self.assertRaisesRegex(TypeError, 'explicit agent'):
            first.corridor.project(agents[0].pos)

    def test_routes_have_real_inverse_and_avoid_block(self):
        scenario = build_scenario(3, num_agents=6, max_steps=2)
        for agent in scenario.spawn_agents():
            route = agent_route(scenario.corridor, agent)
            np.testing.assert_allclose(route.xy_from_frenet(0., 0.)[0], agent.dest)
            self.assertTrue(scenario.corridor.roadway.covers(LineString(route.center)))
            # Points away from vertices have a well-defined segment inverse.
            for j in range(0, len(route.center)-1, 7):
                s = .5*(route.cumulative_s[j]+route.cumulative_s[j+1])
                point, _ = route.xy_from_frenet(s, 0.)
                self.assertAlmostEqual(route.project(point)[0], s, places=6)

    def test_another_agents_goal_cannot_trigger_arrival(self):
        from RL.transition import advance_agents
        scenario = build_scenario(4, num_agents=2, max_steps=2)
        agents = scenario.spawn_agents()
        agents[0].pos = agents[1].dest.copy()
        agents[0].vel[:] = 0.
        self.assertGreater(np.linalg.norm(agents[0].pos-agents[0].dest), 10.)
        self.assertFalse(arrival_reached(scenario.corridor, agents[0], 0., 2.))
        advance_agents(agents, [(0., 0.), (-3., 0.)], scenario.corridor, scenario.sim_config, [0., 0.])
        self.assertFalse(agents[0].reached_destination)

    def test_utility_grid_and_zero_residual_agree(self):
        from RL.candidate_policy import candidate_context
        from RL.decision import select_candidate_with_logit_residual
        scenario = build_scenario(3, num_agents=3, max_steps=2)
        agents = scenario.spawn_agents()
        params = load_base_params(CALIBRATION)
        for i, agent in enumerate(agents):
            context = candidate_context(i, agents, params, scenario.sim_config)
            selected, index, prior_index = select_candidate_with_logit_residual(
                i, agent, agents, params, scenario.sim_config, np.zeros(63))
            self.assertEqual(context.prior_index, index)
            self.assertEqual(index, prior_index)
            np.testing.assert_allclose(selected['pos'], context.candidates[index]['pos'])

    def test_environment_and_benchmark_share_utility_transitions(self):
        from Baselines.scenario import scenario_from_env
        env = MultiAgentTrafficEnv(EnvConfig(num_agents=3, max_steps=8,
                                   base_params=load_base_params(CALIBRATION)), seed=3)
        env.reset()
        scenario = scenario_from_env(env, 3)
        result = rollout(scenario, UtilityPriorController(calibration=CALIBRATION))
        while env.step_count < scenario.max_steps:
            _, _, done, _ = env.step(None)
            if done: break
        np.testing.assert_allclose(result.positions[-1], [a.pos for a in env.agents], atol=1e-9)

    def test_arrivals_and_progress_use_each_own_goal(self):
        from Baselines.metrics import rollout_metrics
        scenarios = [build_scenario(s, num_agents=6, max_steps=80) for s in (3, 7)]
        scenario = scenarios[0]
        result = rollout(scenario, UtilityPriorController(calibration=CALIBRATION))
        for i, agent in enumerate(scenario.agents):
            if result.arrival_step[i] >= 0:
                self.assertLessEqual(np.linalg.norm(result.positions[-1, i]-agent.dest), 2.000001)
        remaining = np.array([scenario.corridor.remaining_to_goal(result.positions[-1, i], a.dest)
                              for i, a in enumerate(scenario.agents)])
        expected = np.clip((-result.start_s-remaining)/(-result.start_s), 0., 1.).mean()
        self.assertAlmostEqual(rollout_metrics(result)['goal_progress'], expected)
        self.assertEqual(result.offroad_steps, 0)

    def test_all_geometric_controllers_and_batched_dynamics(self):
        from Baselines.registry import build_controller
        from Baselines.dynamics import simulate_bicycle_batch
        from utility_model import kinematic_bicycle_rollout
        scenario = build_scenario(2, num_agents=2, max_steps=2)
        for name in ('orca', 'social_force', 'dwa', 'mppi', 'frenet'):
            result = rollout(scenario, build_controller(name))
            self.assertEqual(result.steps, 2)
            self.assertTrue(np.isfinite(result.positions).all())
        agent = scenario.spawn_agents()[0]
        position, _, heading = simulate_bicycle_batch(agent.pos, agent.heading, 0., np.array([[2.]]), np.array([[.2]]), scenario)
        scalar = kinematic_bicycle_rollout(agent.pos, agent.heading, 0., 2., .2, scenario.dt, scenario.sim_config)
        np.testing.assert_allclose(position[0, -1], scalar['pos'])
        self.assertAlmostEqual(heading[0, -1], scalar['heading'])

    def test_network_maps_and_polygon_guidance(self):
        from new_baselines.data import map_polylines, tensor_tree
        from new_baselines.models import make_guidance
        from prepare_new_baselines import generation_config
        from shapely.geometry import Point
        scenario = build_scenario(0, num_agents=1, max_steps=2)
        cfg = generation_config()
        _, types = map_polylines(scenario.corridor, cfg)
        self.assertTrue((types[:, 0] == 0).all())
        self.assertTrue((types[:, 1] == 1).all())
        # A stationary footprint inside the block is penalized more than one
        # inside the drivable road, and derivatives remain finite.
        road = scenario.agents[0].pos
        hole = np.asarray(Point(scenario.corridor.roadway.interiors[0].coords[0]).coords[0])
        from shapely.geometry import Polygon
        hole = np.asarray(Polygon(scenario.corridor.roadway.interiors[0]).representative_point().coords[0])
        costs = []
        for pos in (road, hole):
            state = torch.tensor([[np.r_[pos, 0., 0., 0., cfg.length, cfg.width, 1.]]], dtype=torch.float32)
            goals = torch.tensor([[np.r_[pos, 0., 0., 0.]]], dtype=torch.float32)
            guide = make_guidance(state, goals, scenario.corridor, cfg)
            sample = torch.zeros(1, 1, cfg.horizon, 7, requires_grad=True)
            loss = guide(sample); loss.backward()
            self.assertTrue(torch.isfinite(sample.grad).all())
            costs.append(float(loss))
        self.assertGreater(costs[1], costs[0]+1.)


if __name__ == '__main__':
    unittest.main()
