"""Independent numerical checks of baseline algorithm components.

These establish selected mathematical properties, not complete paper fidelity.
"""
import unittest
import numpy as np
import torch
from scipy.optimize import minimize

from Baselines.orca import Line, _linear_program2, _det, orca_line
from Baselines.frenet_planner import _quintic_coefficients, _quartic_coefficients, _polyval
from Baselines.dynamics import simulate_bicycle_batch
from Baselines.scenario import build_scenario
from utility_model import kinematic_bicycle_rollout


class BaselineAlgorithmTests(unittest.TestCase):
    def test_orca_solver_matches_independent_constrained_optimizer(self):
        rng = np.random.default_rng(21)
        for _ in range(100):
            angles = rng.uniform(-np.pi, np.pi, 6)
            normals = np.column_stack((np.cos(angles), np.sin(angles)))
            bounds = rng.uniform(.05, 1.5, 6)
            lines = [Line(-b * n, np.array([n[1], -n[0]]))
                     for n, b in zip(normals, bounds)]
            desired = rng.normal(size=2) * 3
            fail, actual = _linear_program2(lines, 2., desired, False)
            reference = minimize(lambda v: .5 * np.sum((v - desired)**2), np.zeros(2),
                jac=lambda v: v - desired, method="SLSQP",
                constraints=[{"type": "ineq", "fun": lambda v: normals @ v + bounds},
                             {"type": "ineq", "fun": lambda v: 4. - v @ v}],
                options={"ftol": 1e-8, "maxiter": 200})
            self.assertTrue(reference.success)
            self.assertEqual(fail, len(lines))
            np.testing.assert_allclose(actual, reference.x, atol=2e-5)

    def test_orca_head_on_constraints_are_reciprocal_and_exclude_collision_velocity(self):
        a, b = np.array([-3., 0.]), np.array([3., 0.])
        va, vb = np.array([1., 0.]), np.array([-1., 0.])
        line_a = orca_line(a, va, b, vb, 2., 4., .5)
        line_b = orca_line(b, vb, a, va, 2., 4., .5)
        np.testing.assert_allclose(line_a.point, -line_b.point)
        np.testing.assert_allclose(line_a.direction, -line_b.direction)
        self.assertGreater(_det(line_a.direction, line_a.point - va), 0.)

    def test_frenet_polynomials_satisfy_all_endpoint_conditions(self):
        rng = np.random.default_rng(42)
        start = rng.normal(size=(30, 3))
        target = rng.normal(size=30)
        horizon = rng.uniform(1., 5., 30)
        zero = np.zeros((30, 1))
        end = horizon[:, None]
        for coefficients, degree in [(_quintic_coefficients(start, target, horizon), 5),
                                     (_quartic_coefficients(start, target, horizon), 4)]:
            for derivative in range(3):
                np.testing.assert_allclose(_polyval(coefficients, zero, derivative)[:, 0],
                                           start[:, derivative], atol=1e-9)
            np.testing.assert_allclose(_polyval(coefficients, end, 0 if degree == 5 else 1)[:, 0],
                                       target, atol=1e-9)
            np.testing.assert_allclose(_polyval(coefficients, end, 2), 0., atol=1e-9)
            if degree == 5:
                np.testing.assert_allclose(_polyval(coefficients, end, 1), 0., atol=1e-9)

    def test_batched_planner_dynamics_match_scalar_execution_at_default_settings(self):
        scenario = build_scenario(3, num_agents=1, max_steps=1)
        agent = scenario.spawn_agents()[0]
        rng = np.random.default_rng(9)
        accels = rng.uniform(-4., 4., (20, 8))
        steering = rng.uniform(-.45, .45, (20, 8))
        positions, speeds, headings = simulate_bicycle_batch(
            agent.pos, agent.heading, agent.speed, accels, steering, scenario)
        for k in range(20):
            pos, heading, speed = agent.pos, agent.heading, agent.speed
            for t in range(8):
                result = kinematic_bicycle_rollout(pos, heading, speed, accels[k, t],
                                                   steering[k, t], scenario.dt, scenario.sim_config)
                pos, heading, speed = result["pos"], result["heading"], result["speed"]
                np.testing.assert_allclose(positions[k, t + 1], pos, atol=1e-10)
                self.assertAlmostEqual(speeds[k, t + 1], speed)
                self.assertAlmostEqual(headings[k, t + 1], heading)

    def test_hatrpo_kl_matches_torch_distribution_kl(self):
        from Baselines.marl import Actor
        from Baselines.train_marl import _gaussian_kl
        actor = Actor(3, 8)
        obs = torch.randn(7, 3)
        old_mean, old_log_std = torch.randn(7, 2), torch.randn(7, 2) * .1
        mask = torch.tensor([1., 1., 0., 1., 0., 1., 1.])
        expected = torch.distributions.kl_divergence(
            torch.distributions.Normal(old_mean, old_log_std.exp()),
            actor.distribution(obs)).sum(-1)
        expected = (expected * mask).sum() / mask.sum()
        torch.testing.assert_close(_gaussian_kl(actor, obs, old_mean, old_log_std, mask), expected)


if __name__ == "__main__":
    unittest.main()
