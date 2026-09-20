"""Contract tests for the public-model adapters, independent of large datasets."""
from __future__ import annotations

import copy
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'New Baselines'))
from new_baselines.config import Config
from new_baselines.data import (action_tokens, decode_actions, inverse_controls,
    ctrl_features, ctg_features, tensor_tree, map_polylines)
from new_baselines.models import CtRLSim, CTGPlusPlus, bicycle_rollout, make_guidance
from new_baselines.vendor.social_attention import RelativeSocialAttentionLayer
from new_baselines.training import validate_splits
from new_baselines.controller import TrafficGenerationController
from RL.corridor import load_corridor
from utility_model import kinematic_bicycle_rollout


class NewBaselineTests(unittest.TestCase):
    def test_cache_rejects_changed_site_dynamics_and_visibility(self):
        import json
        from dataclasses import replace
        from new_baselines.training import OfflineScenes
        cfg = Config()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'train.npz'
            path.with_suffix('.json').write_text(json.dumps(dict(
                new_baselines_data_version=1, config=cfg.to_dict())))
            for key, value in [('wheelbase', .99), ('steer_from_rest', True),
                               ('min_steer_speed', 1.), ('perception_radius', 12.),
                               ('max_neighbors', 3)]:
                with self.subTest(key=key), self.assertRaisesRegex(ValueError, 'Cache/config mismatch: '+key):
                    OfflineScenes(path, 'ctrl_sim', replace(cfg, **{key: value}))

    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        cls.cfg = Config(max_agents=3, hidden=32, layers=1, decoder_layers=1,
                         context=4, history=3, horizon=4, diffusion_steps=4, return_bins=16)
        cls.corridor = load_corridor(2, 1)

    def fixture(self, length=4):
        c = self.cfg
        states = np.zeros((c.max_agents, length, 8), np.float32)
        for i in range(2):
            for t in range(length):
                pos, tangent = self.corridor.xy_from_frenet(80+i*15+t*2, 0.)
                states[i, t] = np.r_[pos, tangent*4, np.arctan2(tangent[1], tangent[0]), c.length, c.width, 1]
        goals = np.zeros((c.max_agents, 5), np.float32)
        goals[:, :2] = self.corridor.center[-2]
        roads, types = map_polylines(self.corridor, c)
        return states, goals, roads, types

    def test_action_codebook_roundtrip_and_limits(self):
        c = self.cfg
        tokens = np.arange(c.accel_bins*c.steer_bins)
        actions = decode_actions(tokens, c)
        np.testing.assert_array_equal(action_tokens(actions, c), tokens)
        self.assertAlmostEqual(actions[:, 0].min(), -c.max_accel)
        self.assertAlmostEqual(actions[:, 1].max(), c.max_steer)

    def test_inverse_and_differentiable_dynamics_match_simulator(self):
        c = self.cfg
        sim = dict(max_agent_speed=c.max_speed, max_accel=c.max_accel, wheelbase=c.wheelbase)
        initial = np.array([3., 4., 5., 0., 0., c.length, c.width, 1.], np.float32)
        actions = np.array([[.8, .07], [-1., -.1], [.2, 0.]], np.float32)
        states = [initial]
        for a, delta in actions:
            old = states[-1]
            predicted = kinematic_bicycle_rollout(old[:2], old[4], np.linalg.norm(old[2:4]), a, delta, c.dt, sim)
            states.append(np.r_[predicted['pos'], predicted['vel'], predicted['heading'], c.length, c.width, 1.])
        states = np.asarray(states, np.float32)[None]
        inverse, mask, diagnostics = inverse_controls(states, c)
        np.testing.assert_allclose(inverse[0, :-1], actions, atol=5e-6)
        self.assertEqual(mask.sum(), 3)
        self.assertLess(diagnostics['inverse_position_error_max_m'], 2e-6)
        position, heading = bicycle_rollout(torch.tensor(initial)[None, None], torch.tensor(actions)[None, None], c)
        np.testing.assert_allclose(position[0, 0].numpy(), states[0, 1:, :2], atol=2e-6)
        np.testing.assert_allclose(heading[0, 0].numpy(), states[0, 1:, 4], atol=2e-6)

    def test_ctrl_causality_current_labels_and_future_states(self):
        c = self.cfg; states, goals, roads, types = self.fixture()
        features = ctrl_features(states, np.zeros((3, 4, 2), np.float32), np.zeros((3, 4, 3), np.float32), goals, roads, types, c)
        data = tensor_tree(features)
        model = CtRLSim(c).eval()
        with torch.no_grad():
            original = model(data)
            changed = copy.deepcopy(data)
            changed['agent']['actions'][:, :, 1:] = 999
            changed['agent']['rtgs'][:, :, 2:] = 15
            changed['agent']['agent_states'][:, :, 2:, :5] += 100
            result = model(changed)
        # Action prediction at t=1 cannot see its own target, any future token,
        # or another agent's contemporaneous actions/returns.
        torch.testing.assert_close(original['action_preds'][:, :, 1], result['action_preds'][:, :, 1])
        changed = copy.deepcopy(data)
        changed['agent']['rtgs'][:, :, 1] = 15
        with torch.no_grad():
            result = model(changed)
        torch.testing.assert_close(original['rtg_preds'][:, :, 1], result['rtg_preds'][:, :, 1])
        changed = copy.deepcopy(data)
        changed['agent']['rtgs'][:, 1:, 1] = 15
        with torch.no_grad():
            result = model(changed)
        torch.testing.assert_close(original['action_preds'][:, 0, 1], result['action_preds'][:, 0, 1])

    def test_dense_attention_matches_edgewise_reference(self):
        torch.manual_seed(3)
        layer = RelativeSocialAttentionLayer(16, 4, 0., 32).eval()
        x = torch.randn(3, 2, 16)
        edge = torch.randn(2, 9, 16)
        mask = torch.tensor([[False, False, True], [False, False, False]])
        dense = layer._mha_block(x, mask, edge)
        expected = torch.zeros_like(x)
        for b in range(2):
            for target in range(3):
                aggregate = torch.zeros(16)
                if not mask[b, target]:
                    logits, values = [], []
                    for source in range(3):
                        if mask[b, source]:
                            continue
                        e = edge[b, target*3+source]
                        q = layer.lin_q_node(x[target, b]).reshape(4, 4)
                        k = (layer.lin_k_node(x[source, b])+layer.lin_k_edge(e)).reshape(4, 4)
                        logits.append((q*k).sum(-1)/2.)
                        values.append((layer.lin_v_node(x[source, b])+layer.lin_v_edge(e)).reshape(4, 4))
                    weights = torch.stack(logits).softmax(0)
                    aggregate = (weights[..., None]*torch.stack(values)).sum(0).flatten()
                gate = torch.sigmoid(layer.lin_ih(aggregate)+layer.lin_hh(x[target, b]))
                expected[target, b] = layer.out_proj(aggregate+gate*(layer.lin_self(x[target, b])-aggregate))
        torch.testing.assert_close(dense, expected)

    def test_diffusion_conditioning_does_not_contain_future(self):
        c = self.cfg; states, goals, roads, types = self.fixture(c.history)
        future = np.repeat(states[:, -1:], c.horizon, axis=1)
        actions = np.zeros((3, c.horizon, 2), np.float32)
        kwargs = dict(states=states, incoming=np.zeros((3, c.history, 2), np.float32), goals=goals,
                      roads=roads, road_types=types, cfg=c, future_actions=actions)
        cond, target = ctg_features(future=future, **kwargs)
        changed = future.copy(); changed[..., :2] += 123
        cond2, target2 = ctg_features(future=changed, **kwargs)
        for a, b in zip(cond, cond2):
            np.testing.assert_array_equal(a, b)
        self.assertFalse(np.allclose(target, target2))
        model = CTGPlusPlus(c)
        loss, _ = model.loss(tensor_tree(cond), tensor_tree(target), tensor_tree(future[..., -1]))
        loss.backward()
        self.assertTrue(np.isfinite(float(loss)))
        self.assertGreater(sum(float(p.grad.abs().sum()) for p in model.parameters() if p.grad is not None), 0)
        model.eval()
        conditioning = tensor_tree(cond)
        noisy = torch.randn(1, 3, c.horizon, 7)
        with torch.no_grad():
            before = model.denoiser(noisy, conditioning, torch.tensor([2]), eval=True)
            perturbed = noisy.clone(); perturbed[:, 2] += 100
            after = model.denoiser(perturbed, conditioning, torch.tensor([2]), eval=True)
        # An absent padded agent must not become a social-attention participant.
        torch.testing.assert_close(before[:, :2], after[:, :2])
        a = model.sample(tensor_tree(cond), torch.Generator().manual_seed(8))
        b = model.sample(tensor_tree(cond), torch.Generator().manual_seed(8))
        torch.testing.assert_close(a, b)
        self.assertTrue(torch.isfinite(a).all())

    def test_guidance_is_differentiable_and_padding_finite(self):
        states, goals, _, _ = self.fixture(1)
        guide = make_guidance(tensor_tree(states[:, -1]), tensor_tree(goals), self.corridor, self.cfg)
        sample = torch.zeros(1, 3, self.cfg.horizon, 7, requires_grad=True)
        cost = guide(sample)
        cost.backward()
        self.assertTrue(torch.isfinite(cost))
        self.assertTrue(torch.isfinite(sample.grad).all())
        self.assertGreater(float(sample.grad[..., -2:].abs().sum()), 0)

    def test_split_and_checkpoint_guards(self):
        manifest = dict(split='train', data_sha256='abc', run_id=2, lane_kf=1,
                        split_intervals_s={}, scenes=[dict(agent_ids=[1, 2])])
        val = copy.deepcopy(manifest); val['split'] = 'validation'
        with self.assertRaisesRegex(ValueError, 'identity leakage'):
            validate_splits(manifest, val)
        val['scenes'] = [dict(agent_ids=[3, 4])]
        validate_splits(manifest, val)
        val['split'] = 'test'
        with self.assertRaisesRegex(ValueError, 'forbidden'):
            validate_splits(manifest, val)
        with tempfile.TemporaryDirectory() as tmp:
            checkpoint = Path(tmp)/'smoke.pt'
            torch.save(dict(format_version=1, model='ctrl_sim', smoke_only=True), checkpoint)
            with self.assertRaisesRegex(ValueError, 'Smoke checkpoint'):
                TrafficGenerationController('ctrl_sim', checkpoint)
        from Baselines.registry import build_controller, controller_kwargs
        self.assertEqual(controller_kwargs('ctg_plus_plus', checkpoint_dir=Path('x'))['checkpoint'], Path('x/ctg_plus_plus_policy.pt'))
        with self.assertRaises(FileNotFoundError):
            build_controller('ctrl_sim', checkpoint=Path('nonexistent.pt'))


if __name__ == '__main__':
    unittest.main()
