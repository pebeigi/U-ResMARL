import unittest
from unittest.mock import patch
import numpy as np

from RL.boundary import (BoundaryInfeasibleError, candidate_boundary_safe,
                         footprint_clearance, filter_boundary_control, _sweep_inside)
from RL.corridor import HighwayCorridor
from RL.transition import advance_agents
from Baselines.scenario import build_scenario
from Baselines.discrete_action import feasible_action_mask, grid_candidate
from utility_model import kinematic_bicycle_rollout, select_candidate_with_logit_residual, DEFAULT_BASE_PARAMS


def straight_road():
    center = np.array([[0., 0.], [100., 0.]])
    return HighwayCorridor(0, 0, center, center + [0., -3.], center + [0., 3.], np.array([0., 100.]), np.array([[1., 0.]]))


class BoundaryTests(unittest.TestCase):
    def setUp(self):
        self.scenario = build_scenario(0, num_agents=1, max_steps=3, obb_safety_filter=False)
        self.agent = self.scenario.spawn_agents()[0]
        self.road = straight_road()
        self.sim = dict(self.scenario.sim_config)
        self.agent.pos = np.array([20., 0.])
        self.agent.heading_angle = 0.
        self.agent.vel = np.array([8., 0.])

    def test_footprint_overhang_is_outside_even_with_inside_center(self):
        self.assertTrue(self.road.inside(np.array([20., 2.5])))
        self.assertLess(footprint_clearance(self.road, [20., 2.5], 0.), 0.)

    def test_sweep_rejects_rotation_with_safe_endpoints(self):
        self.agent.pos = np.array([20., 1.5])
        candidate = {"pos": self.agent.pos.copy(), "heading": np.pi, "speed": 0.}
        self.assertGreater(footprint_clearance(self.road, self.agent.pos, 0.), 0.)
        self.assertGreater(footprint_clearance(self.road, candidate["pos"], np.pi), 0.)
        self.assertFalse(_sweep_inside(self.agent.pos, 0., candidate, self.road, self.sim))

    def test_next_pose_inside_is_not_enough_without_braking_room(self):
        self.agent.heading_angle = .1
        self.agent.pos = np.array([20., .8])
        self.agent.vel = 8. * np.array([np.cos(.1), np.sin(.1)])
        candidate = kinematic_bicycle_rollout(self.agent.pos, .1, 8., 0., 0., .5, self.sim)
        self.assertTrue(_sweep_inside(self.agent.pos, .1, candidate, self.road, self.sim))
        self.assertFalse(candidate_boundary_safe(self.agent, candidate, self.sim, self.road))

    def test_continuous_filter_replaces_offroad_command(self):
        command = filter_boundary_control(self.agent, (3., .45), self.sim, self.road)
        self.assertNotEqual(command, (3., .45))
        candidate = kinematic_bicycle_rollout(self.agent.pos, 0., 8., *command, .5, self.sim)
        self.assertTrue(candidate_boundary_safe(self.agent, candidate, self.sim, self.road))

    def test_infeasible_start_does_not_move_any_agent(self):
        self.agent.pos[1] = 4.
        before = self.agent.pos.copy()
        with patch("RL.boundary.load_corridor", return_value=self.road):
            with self.assertRaises(BoundaryInfeasibleError):
                advance_agents([self.agent], [(0., 0.)], self.road, self.sim, [90.])
        np.testing.assert_array_equal(before, self.agent.pos)

    def test_braking_backup_remains_available_until_stopped(self):
        command = filter_boundary_control(self.agent, (3., .45), self.sim, self.road)
        candidate = kinematic_bicycle_rollout(self.agent.pos, self.agent.heading,
                    self.agent.speed, *command, .5, self.sim)
        self.agent.update_state_from_candidate(candidate, .5, 1.)
        for _ in range(10):
            backup = kinematic_bicycle_rollout(self.agent.pos, self.agent.heading,
                        self.agent.speed, -3., 0., .5, self.sim)
            self.assertTrue(candidate_boundary_safe(self.agent, backup, self.sim, self.road))
            self.agent.update_state_from_candidate(backup, .5, 1.)
        self.assertEqual(self.agent.speed, 0.)

    def test_old_checkpoint_cannot_enter_boundary_protocol(self):
        from RL.protocol import validate_checkpoint
        with self.assertRaisesRegex(ValueError, "retrain"):
            validate_checkpoint({"protocol_version": 2, "obs_dim": 32}, 32, "v2.pt")

    def test_masks_still_enforce_road_when_collision_filter_is_off(self):
        agents = self.scenario.spawn_agents()
        mask = feasible_action_mask(0, agents[0], agents, self.scenario)
        expected = [candidate_boundary_safe(agents[0], grid_candidate(agents[0], k, .5,
                    self.scenario.sim_config), self.scenario.sim_config, self.scenario.corridor)
                    for k in range(len(mask))]
        np.testing.assert_array_equal(mask, expected)
        self.assertTrue(mask.any())
        self.assertFalse(mask.all())
        residual = np.where(mask, -1e6, 1e6)
        candidate, chosen, _ = select_candidate_with_logit_residual(
            0, agents[0], agents, DEFAULT_BASE_PARAMS, self.scenario.sim_config, residual)
        self.assertTrue(mask[chosen])


if __name__ == "__main__":
    unittest.main()
