"""NAVSIM-style episode score and PDM residual-accept gate."""
import unittest

import numpy as np

from RL.closed_loop_score import closed_loop_score, COMFORT_RMS_JERK
from RL.candidate_policy import CandidateIndex
from RL.traffic_env import EnvConfig, MultiAgentTrafficEnv


class EpisodeScoreTests(unittest.TestCase):
    def test_collision_or_offroad_zeros_the_score(self):
        base = dict(collision_events=0., offroad_rate=0., goal_progress=1.,
                    arrival_rate=1., unsafe_ttc_rate=0., mean_abs_accel=1.,
                    rms_jerk=1., mean_abs_steering=0.02)
        self.assertGreater(closed_loop_score(base), 0.9)
        self.assertEqual(closed_loop_score({**base, "collision_events": 1.}), 0.)
        self.assertEqual(closed_loop_score({**base, "offroad_rate": 0.1}), 0.)

    def test_in_band_comfort_does_not_zero_the_score(self):
        stats = dict(collision_events=0., offroad_rate=0., goal_progress=1.,
                     mean_abs_accel=1.9, rms_jerk=COMFORT_RMS_JERK, mean_abs_steering=0.09)
        self.assertEqual(closed_loop_score(stats), closed_loop_score(
            {**stats, "mean_abs_accel": 0.5, "rms_jerk": 1., "mean_abs_steering": 0.01}))

    def test_out_of_band_comfort_lowers_but_does_not_zero_a_safe_run(self):
        safe = dict(collision_events=0., offroad_rate=0., goal_progress=1.,
                    mean_abs_accel=1., rms_jerk=1., mean_abs_steering=0.02)
        harsh = {**safe, "rms_jerk": 20.}
        self.assertGreater(closed_loop_score(safe), closed_loop_score(harsh))
        self.assertGreater(closed_loop_score(harsh), 0.)


class ResidualAcceptTests(unittest.TestCase):
    def test_eval_gate_keeps_utility_when_residual_candidate_does_not_score_better(self):
        from RL.closed_loop_score import proposal_closed_loop_score
        env = MultiAgentTrafficEnv(EnvConfig(num_agents=1, accept_residual_if_better=True), seed=42)
        env.reset()
        context = env.candidate_context(0)
        scored = [
            (proposal_closed_loop_score(
                env.agents[0], context.candidates[int(i)], env.agents, 0,
                env.config.sim_config, env.corridor), int(i))
            for i in np.flatnonzero(context.mask)
        ]
        worst = min(scored)[1]
        if worst == context.prior_index:
            self.skipTest("utility already has the worst feasible proposal on this seed")
        _, _, _, info = env.step([CandidateIndex(worst)])
        self.assertEqual(info["candidate_flips"], 0)
        self.assertEqual(info["control_flips"], 0)

    def test_training_still_executes_the_sampled_residual(self):
        env = MultiAgentTrafficEnv(EnvConfig(num_agents=1, accept_residual_if_better=False), seed=42)
        env.reset()
        context = env.candidate_context(0)
        alternative = next(int(i) for i in np.flatnonzero(context.mask) if i != context.prior_index)
        _, _, _, info = env.step([CandidateIndex(alternative)])
        self.assertEqual(info["candidate_flips"], 1)
