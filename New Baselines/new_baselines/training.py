from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path

import numpy as np
import torch

from .config import Config
from .data import ctrl_features, ctg_features, pad_agents, select_group, tensor_tree
from .models import make_model


class OfflineScenes:
    def __init__(self, path, model, cfg):
        path = Path(path)
        self.manifest = json.loads(path.with_suffix('.json').read_text())
        if self.manifest.get('new_baselines_data_version') != 1:
            raise ValueError('Unsupported training cache')
        recorded = self.manifest['config']
        for key in ('dt', 'horizon', 'context', 'history', 'length', 'width', 'max_accel',
                    'max_speed', 'max_steer', 'wheelbase', 'map_segments', 'map_points',
                    'steer_from_rest', 'min_steer_speed', 'perception_radius',
                    'max_neighbors', 'decision_protocol_version'):
            if recorded[key] != getattr(cfg, key):
                raise ValueError('Cache/config mismatch: '+key)
        self.cfg, self.model = cfg, model
        self.scenes = []
        with np.load(path, allow_pickle=False) as archive:
            for i in range(self.manifest['scene_count']):
                prefix = str(i)+'_'
                self.scenes.append({k[len(prefix):]: archive[k] for k in archive.files if k.startswith(prefix)})
        self.indices = []
        for scene_idx, data in enumerate(self.scenes):
            states = data['states']
            for t in range(states.shape[1]-cfg.horizon):
                for focal in range(states.shape[0]):
                    if states[focal, t, -1] and data['return_mask'][focal, t]:
                        self.indices.append((scene_idx, t, focal))
        if not self.indices:
            raise ValueError('No complete training windows in cache')

    def sample(self, index):
        scene_idx, t, focal = self.indices[index]
        data, cfg = self.scenes[scene_idx], self.cfg
        ids = select_group(data['states'], focal, cfg, present=t)
        def window(key, length):
            values = data[key][ids, max(0, t-length+1):t+1]
            result = np.zeros((len(ids), length)+values.shape[2:], values.dtype)
            result[:, -values.shape[1]:] = values
            return pad_agents(result, cfg)
        goals = pad_agents(data['goals'][ids], cfg)
        if self.model == 'ctrl_sim':
            states = window('states', cfg.context)
            features = ctrl_features(states, window('actions', cfg.context), window('rtgs', cfg.context),
                                      goals, data['roads'], data['road_types'], cfg)
            return features, window('action_mask', cfg.context), window('return_mask', cfg.context)
        past = window('states', cfg.history)
        incoming = np.zeros((len(ids), cfg.history, 2), np.float32)
        start = max(0, t-cfg.history+1)
        for j, state_t in enumerate(range(start, t+1), cfg.history-(t-start+1)):
            if state_t > 0:
                incoming[:, j] = data['actions'][ids, state_t-1]
        future = pad_agents(data['states'][ids, t+1:t+cfg.horizon+1], cfg)
        actions = pad_agents(data['actions'][ids, t:t+cfg.horizon], cfg)
        cond, target = ctg_features(past, pad_agents(incoming, cfg), goals, data['roads'], data['road_types'], cfg,
                                    future=future, future_actions=actions)
        mask = future[..., -1]*past[:, -1:, -1]
        return cond, target, mask


def collate(values):
    if isinstance(values[0], dict):
        return {k: collate([v[k] for v in values]) for k in values[0]}
    if isinstance(values[0], tuple):
        return tuple(collate([v[i] for v in values]) for i in range(len(values[0])))
    return np.stack(values)


def validate_splits(train, validation):
    if train['split'] != 'train' or validation['split'] != 'validation':
        raise ValueError('Training requires train and validation caches; test data are forbidden')
    for key in ('data_sha256', 'run_id', 'lane_kf', 'split_intervals_s'):
        if train[key] != validation[key]:
            raise ValueError('Incompatible source/split manifests: '+key)
    if train.get('site_protocol') != validation.get('site_protocol'):
        raise ValueError('Incompatible source/split manifests: site_protocol')
    train_ids = {i for s in train['scenes'] for i in s['agent_ids']}
    val_ids = {i for s in validation['scenes'] for i in s['agent_ids']}
    if train_ids & val_ids:
        raise ValueError('Agent identity leakage between training and validation')


def train(model_name, train_cache, validation_cache, output, cfg, *, steps=10000,
          batch_size=4, lr=1e-4, seed=0, device='cpu', smoke=False, eval_every=250):
    if steps < 1 or batch_size < 1 or eval_every < 1:
        raise ValueError('Positive training steps, batch size and validation interval required')
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    train_data = OfflineScenes(train_cache, model_name, cfg)
    validation = OfflineScenes(validation_cache, model_name, cfg)
    validate_splits(train_data.manifest, validation.manifest)
    model = make_model(model_name, cfg).to(device)
    from RL.experiment_protocol import source_manifest
    sources = source_manifest()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    rng = np.random.default_rng(seed)
    val_indices = np.random.default_rng(12345).permutation(len(validation.indices))[:min(128, len(validation.indices))]
    best, best_step, best_state, history = float('inf'), 0, None, []
    for step in range(1, steps+1):
        model.train()
        rows = [train_data.sample(int(i)) for i in rng.integers(len(train_data.indices), size=batch_size)]
        batch = tensor_tree(collate(rows), device, batch=False)
        optimizer.zero_grad(set_to_none=True)
        loss, components = model.loss(*batch)
        if not torch.isfinite(loss):
            raise FloatingPointError('Non-finite training loss')
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
        optimizer.step()
        if step % eval_every == 0 or step == steps:
            model.eval(); losses = []
            # Isolate validation noise from training RNG; common random numbers
            # make checkpoint selection stable for the diffusion loss.
            with torch.random.fork_rng(devices=[torch.device(device)] if str(device).startswith('cuda') else []):
                torch.manual_seed(9876)
                with torch.no_grad():
                    for start in range(0, len(val_indices), batch_size):
                        rows = [validation.sample(int(i)) for i in val_indices[start:start+batch_size]]
                        val_loss, _ = model.loss(*tensor_tree(collate(rows), device, batch=False))
                        losses.extend([float(val_loss)]*len(rows))
            score = float(np.mean(losses))
            if not np.isfinite(score):
                raise FloatingPointError('Non-finite validation loss')
            if score < best:
                best, best_step = score, step
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            row = dict(step=step, train_loss=float(loss.detach()), validation_loss=score, **components)
            history.append(row)
            print(json.dumps(dict(model=model_name, **row)), flush=True)
            # Retain every validation candidate for a later matched closed-loop
            # selection pass. Offline loss alone is not the paper selector.
            candidate_dir = Path(output).with_suffix('.candidates')
            candidate_dir.mkdir(parents=True, exist_ok=True)
            torch.save(dict(format_version=1, model=model_name, config=cfg.to_dict(),
                site_protocol=train_data.manifest.get('site_protocol'),
                state_dict={k: v.detach().cpu() for k, v in model.state_dict().items()},
                smoke_only=smoke, selection_status='offline_candidate',
                decision_protocol_version=1,
                metadata=dict(seed=seed, steps=step, optimizer_steps=step,
                              offline_validation_loss=score, source_hashes=sources,
                              train_source_sha256=train_data.manifest['data_sha256'])),
                candidate_dir/f'step_{step:08d}.pt')
    root = Path(__file__).resolve().parents[1]
    metadata = dict(seed=seed, steps=steps, optimizer_steps=steps, source_hashes=sources,
        selected_step=best_step, best_validation_loss=best,
        train_examples=len(train_data.indices), validation_examples=len(validation.indices),
        validation_selection_examples=len(val_indices), source_sha256=train_data.manifest['data_sha256'],
        run_id=train_data.manifest['run_id'], lane_kf=train_data.manifest['lane_kf'],
        split_intervals_s=train_data.manifest['split_intervals_s'],
        provenance=json.loads((root/'PROVENANCE.json').read_text()),
        train_manifest_sha256=hashlib.sha256(Path(train_cache).with_suffix('.json').read_bytes()).hexdigest(),
        validation_manifest_sha256=hashlib.sha256(Path(validation_cache).with_suffix('.json').read_bytes()).hexdigest(),
        holdout_note=train_data.manifest['holdout_note'])
    payload = dict(format_version=1, model=model_name, config=cfg.to_dict(), state_dict=best_state,
        site_protocol=train_data.manifest.get('site_protocol'),
        decision_protocol_version=1,
        smoke_only=smoke, selection_status='smoke_only' if smoke else 'offline_validation_loss', metadata=metadata)
    output = Path(output); output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, output)
    output.with_suffix('.summary.json').write_text(json.dumps(dict(model=model_name, config=cfg.to_dict(),
        smoke_only=smoke, metadata=metadata, history=history), indent=2)+'\n')
    return payload


def select_checkpoints(model_name, checkpoints, output, args, *, device='cpu', allow_smoke=False):
    """Same validation scenes, safety rule and PDMS ranking as online learners."""
    from Baselines.training import PolicySelection
    from RL.experiment_protocol import (selection_key, regressions, validation_seeds,
                                        SELECTION_RULE, write_json, same_site_protocol)
    from .controller import TrafficGenerationController
    evaluator = PolicySelection.__new__(PolicySelection)
    evaluator.args = args
    seeds = validation_seeds(args)
    if not seeds or args.num_agents <= 0 or args.max_steps <= 0:
        raise ValueError('Positive validation episodes, agents and horizon required')
    if set(seeds) & set(validation_seeds(args, True)):
        raise ValueError('Validation and test scenario seeds must be disjoint')
    checkpoints = [Path(path) for path in checkpoints]
    if not checkpoints:
        raise ValueError('At least one candidate checkpoint is required')
    expected_site = evaluator.scenario(0).sim_config.get('site_protocol')
    for path in checkpoints:
        payload = torch.load(path, map_location='cpu', weights_only=True)
        if payload.get('decision_protocol_version') != 1:
            raise ValueError('Candidate was trained under an older decision protocol; retrain before matched selection')
        if not same_site_protocol(payload.get('site_protocol'), expected_site):
            raise ValueError('Candidate site protocol differs from current validation adapter; regenerate data and retrain')
    prior = evaluator.evaluate(seeds, prior=True)
    rows, best_key, selected = [], None, None
    for path in checkpoints:
        path = Path(path)
        controller = TrafficGenerationController(model_name, path, device=device, allow_smoke=allow_smoke)
        stats = evaluator.evaluate(seeds, controller=controller)
        key = selection_key(stats, prior)
        rows.append(dict(path=str(path.resolve()), sha256=hashlib.sha256(path.read_bytes()).hexdigest(), validation=stats))
        if best_key is None or key < best_key:
            best_key, selected = key, (path, stats)
    if selected is None:
        raise ValueError('At least one candidate checkpoint is required')
    path, stats = selected
    payload = torch.load(path, map_location='cpu', weights_only=True)
    if payload.get('decision_protocol_version') != 1:
        raise ValueError('Candidate was trained under an older decision protocol; retrain before matched selection')
    payload.update(selection_rule=SELECTION_RULE, val_seeds=seeds, validation=stats,
                   validation_config=evaluator.validation_config(),
                   prior_validation=prior, selection_candidates=rows,
                   selection_status='safety_failed' if regressions(stats, prior) else 'safety_feasible',
                   safety_regressions=regressions(stats, prior))
    output = Path(output); output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, output)
    write_json(output.with_suffix('.summary.json'), {k: v for k, v in payload.items() if k != 'state_dict'})
    return payload
