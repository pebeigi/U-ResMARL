"""Known-answer metrics and leakage/coverage regression tests."""

import copy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from Baselines.data_evaluation import (
    RecordedScene, aligned_motion, build_recorded_scenes, horizon_steps,
    make_recorded_scene, resample_track, temporal_partition, trajectory_metrics,
)
from Baselines.realism import (
    TTC_CAP_S, _js_divergence, interaction_features, paired_realism_metrics,
)
from Baselines.runner import RolloutResult
from Baselines.scenario import AgentInit, Scenario
from RL.corridor import HighwayCorridor, load_corridor
from RL.traffic_env import EnvConfig


def fixture(speed=2., dt=.5, steps=10):
    center = np.array([[0., 0.], [200., 0.]])
    corridor = HighwayCorridor(2, 1, center, center + [0., -10.], center + [0., 10.],
                               np.array([0., 200.]), np.array([[1., 0.]]))
    times = np.arange(-2, steps + 1) * dt
    positions = np.zeros((len(times), 2, 2))
    positions[:, :, 0] = np.array([50., 70.]) + speed * times[:, None]
    agents = [AgentInit(i, positions[2, i].copy(), np.array([speed, 0.]), 0.,
                        np.array([195., 0.]), 195., positions[2, i, 0], 8.)
              for i in range(2)]
    scenario = Scenario(0, 2, 1, dt, steps, {}, agents, corridor)
    scene = RecordedScene(scenario, positions.copy(), 2, {})
    future = positions[2:].copy()
    result = RolloutResult(
        model="test", seed=0, run_id=2, lane_kf=1, dt=dt, vehicle_length=4.5,
        vehicle_width=1.8, steps=steps, num_agents=2, positions=future,
        headings=np.zeros((steps + 1, 2)), speeds=np.full((steps + 1, 2), speed),
        accels=np.zeros((steps, 2)), steerings=np.zeros((steps, 2)),
        active=np.ones((steps + 1, 2), dtype=bool), lateral=future[:, :, 1],
        clearance=np.ones((steps + 1, 2)), station=future[:, :, 0],
        arrival_step=np.array([-1, -1]), start_s=np.array([50., 70.]), dest_s=np.array([195., 195.]))
    return result, scene


class TrajectoryMetricsTest(unittest.TestCase):
    def test_perfect_match_has_zero_error_and_realism_distance(self):
        result, scene = fixture()
        errors = trajectory_metrics(result, scene)
        for key, value in errors.items():
            if key.startswith(("ade_", "fde_", "speed_rmse_", "heading_mae_")):
                self.assertAlmostEqual(value, 0.)
        scores = paired_realism_metrics(result, scene)
        for key, value in scores.items():
            if key.startswith(("w1_", "js_")) or key == "realism_score":
                self.assertAlmostEqual(value, 0.)
        self.assertEqual(scores["realism_samples_speed"], 20)

    def test_known_position_and_speed_error_at_each_horizon(self):
        result, scene = fixture()
        result.positions[1:, :, 0] += np.arange(1, 11)[:, None] * .5
        errors = trajectory_metrics(result, scene)
        self.assertAlmostEqual(errors["ade_1s_m"], .75)
        self.assertAlmostEqual(errors["fde_1s_m"], 1.)
        self.assertAlmostEqual(errors["ade_3s_m"], 1.75)
        self.assertAlmostEqual(errors["fde_5s_m"], 5.)
        self.assertAlmostEqual(errors["speed_rmse_5s_mps"], 1.)

    def test_constant_xy_offset_is_euclidean_displacement(self):
        result, scene = fixture()
        result.positions[1:] += [3., 4.]
        errors = trajectory_metrics(result, scene)
        self.assertAlmostEqual(errors["ade_5s_m"], 5.)
        self.assertAlmostEqual(errors["fde_5s_m"], 5.)

    def test_missing_future_has_horizon_specific_coverage(self):
        result, scene = fixture()
        scene.reference_positions[2 + 5:, 1] = np.nan
        errors = trajectory_metrics(result, scene)
        self.assertEqual(errors["trajectory_agents_1s"], 2)
        self.assertEqual(errors["trajectory_agents_3s"], 1)
        self.assertEqual(errors["trajectory_coverage_5s"], .5)

    def test_arrived_or_failed_predictions_are_not_removed(self):
        result, scene = fixture()
        result.active[1:, 0] = False
        result.positions[1:, 0] = result.positions[0, 0]
        errors = trajectory_metrics(result, scene)
        self.assertEqual(errors["trajectory_agents_5s"], 2)
        self.assertAlmostEqual(errors["fde_5s_m"], 5.)
        self.assertEqual(paired_realism_metrics(result, scene)["realism_samples_speed"], 20)
        result.positions[-1, 0, 0] = np.nan
        with self.assertRaisesRegex(ValueError, "Non-finite"):
            trajectory_metrics(result, scene)

    def test_heading_wraps_across_pi(self):
        result, scene = fixture()
        for positions, angle, zero in (
                (scene.reference_positions, -np.pi + .01, 2),
                (result.positions, np.pi - .01, 0)):
            for t in range(zero + 1, len(positions)):
                positions[t] = positions[zero] + (t - zero) * np.array([np.cos(angle), np.sin(angle)])
        errors = trajectory_metrics(result, scene)
        self.assertAlmostEqual(errors["heading_mae_5s_rad"], .02)

    def test_stationary_reference_has_no_heading_score(self):
        result, scene = fixture(speed=0.)
        errors = trajectory_metrics(result, scene)
        self.assertTrue(np.isnan(errors["heading_mae_5s_rad"]))
        self.assertEqual(errors["heading_samples_5s"], 0)

    def test_incomplete_or_mismatched_rollouts_fail(self):
        result, scene = fixture()
        result.steps -= 1
        with self.assertRaisesRegex(ValueError, "entire fixed horizon"):
            aligned_motion(result, scene)
        result.steps += 1
        result.positions[0] = result.positions[0, ::-1]
        with self.assertRaisesRegex(ValueError, "agent order"):
            aligned_motion(result, scene)
        with self.assertRaises(ValueError):
            horizon_steps([1.1], .5)


class RealismMetricsTest(unittest.TestCase):
    def test_identical_and_out_of_reference_histograms(self):
        reference = np.ones(10)
        self.assertEqual(_js_divergence(reference, reference), 0.)
        self.assertAlmostEqual(_js_divergence(reference + 100., reference), np.log(2))

    def test_closing_pair_has_finite_ttc_and_equal_speed_pair_is_safe(self):
        p = np.array([[0., 0.], [12., 0.]])
        v = np.array([[4., 0.], [2., 0.]])
        gap, ttc = interaction_features(p, v, np.zeros(2), np.ones(2, bool), 4.5, 1.8)
        self.assertTrue((gap > 0).all())
        np.testing.assert_allclose(ttc, gap / 2.)
        _, safe = interaction_features(p, np.ones((2, 2)), np.zeros(2), np.ones(2, bool), 4.5, 1.8)
        np.testing.assert_allclose(safe, TTC_CAP_S)

    def test_no_neighbor_preserves_safe_ttc_mass(self):
        _, ttc = interaction_features(np.zeros((1, 2)), np.zeros((1, 2)),
                                      np.zeros(1), np.ones(1, bool), 4.5, 1.8)
        self.assertEqual(ttc[0], TTC_CAP_S)

    def test_behavior_scores_use_motion_not_command_acceleration(self):
        result, scene = fixture()
        result.accels[:] = 100.
        result.speeds[:] = 100.
        self.assertEqual(paired_realism_metrics(result, scene)["w1_accel"], 0.)


class RecordedDataProtocolTest(unittest.TestCase):
    def test_whole_tracks_crossing_time_cuts_are_purged(self):
        frame = pd.DataFrame({"id": [1, 1, 2, 2, 3, 3, 4, 4],
                              "time": [0., 20., 61., 75., 81., 100., 55., 65.]})
        intervals, groups = temporal_partition(frame)
        self.assertEqual(intervals["test"], (80., 100.))
        self.assertEqual(groups, {1: "train", 2: "validation", 3: "test"})

    def test_no_future_interpolation_no_extrapolation_no_gap_bridge(self):
        track = pd.DataFrame({"time": [0., .1, 1.], "xloc_kf": [0., 1., 100.],
                              "yloc_kf": [0., 0., 0.], "lane_kf": [1, 1, 1]})
        values = resample_track(track, np.array([-.1, .05, .5, 1.1]), .15)
        self.assertTrue(np.isnan(values[[0, 2, 3]]).all())
        self.assertAlmostEqual(values[1, 0], .5)
        self.assertTrue(np.isnan(resample_track(track, np.array([.05]), .15, cutoff=.05)).all())

    def test_future_changes_do_not_change_controller_inputs(self):
        cfg = EnvConfig(max_steps=2)
        corridor = load_corridor()
        times = np.arange(9., 12.01, .1)
        tracks = {}
        for i in range(2):
            xy = np.array([corridor.xy_from_frenet(100 + i * 20 + 2 * (t - 9), 0.)[0] for t in times])
            tracks[i] = pd.DataFrame(dict(time=times, xloc_kf=xy[:, 0], yloc_kf=xy[:, 1], lane_kf=1))
        first = make_recorded_scene(tracks, 10., cfg, seed=0, history_steps=2, max_gap=.15)
        modified = copy.deepcopy(tracks)
        for track in modified.values():
            track.loc[track.time > 10. + 1e-8, ["xloc_kf", "yloc_kf"]] += 100.
        second = make_recorded_scene(modified, 10., cfg, seed=0, history_steps=2, max_gap=.15)
        self.assertIsNotNone(first)
        self.assertIsNotNone(second)
        for a, b in zip(first.scenario.agents, second.scenario.agents):
            np.testing.assert_array_equal(a.pos, b.pos)
            np.testing.assert_array_equal(a.vel, b.vel)
            np.testing.assert_array_equal(a.dest, b.dest)
            self.assertEqual(a.desired_speed, b.desired_speed)
        self.assertFalse(np.allclose(first.reference_positions, second.reference_positions))

    def test_builder_is_reproducible_and_does_not_certify_old_calibration(self):
        corridor = load_corridor()
        rows = [dict(id=99, time=t, run_id=2, lane_kf=2, xloc_kf=0., yloc_kf=0.)
                for t in [0., 100.]]
        for i in range(2):
            for t in np.arange(83., 100.01, .1):
                p, _ = corridor.xy_from_frenet(100 + 20 * i + 2 * (t - 83), 0.)
                rows.append(dict(id=i, time=t, run_id=2, lane_kf=1, xloc_kf=p[0], yloc_kf=p[1]))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "data.csv"
            pd.DataFrame(rows).to_csv(path, index=False)
            kwargs = dict(count=1, seed=0, run_id=2, lane_kf=1)
            scenes, manifest = build_recorded_scenes(path, **kwargs)
            again, second = build_recorded_scenes(path, **kwargs)
        self.assertEqual(manifest, second)
        self.assertEqual(manifest["holdout_status"], "retrospective_partition_prior_exposure_unverified")
        self.assertEqual(manifest["split_track_ids"]["test"], [0, 1])
        self.assertEqual(manifest["purged_cross_partition_track_ids"], [99])
        np.testing.assert_array_equal(scenes[0].reference_positions, again[0].reference_positions)


if __name__ == "__main__":
    unittest.main()
