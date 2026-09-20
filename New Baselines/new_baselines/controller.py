"""Adapters implementing the existing Baselines Controller interface."""
from __future__ import annotations

from pathlib import Path
import numpy as np
import torch

from .config import Config
from .data import (ctrl_features, ctg_features, decode_actions, inverse_controls,
                   map_polylines, pad_agents, select_group, tensor_tree)
from .models import make_model, make_guidance


class TrafficGenerationController:
    def __init__(self, name, checkpoint=None, device='cpu', allow_smoke=False):
        self.name = name
        self.checkpoint = Path(checkpoint) if checkpoint else Path(__file__).resolve().parents[1]/'checkpoints'/(name+'_policy.pt')
        if not self.checkpoint.is_file():
            raise FileNotFoundError(f'{name}: train a local checkpoint first: {self.checkpoint}')
        payload = torch.load(self.checkpoint, map_location='cpu', weights_only=True)
        if payload.get('format_version') != 1 or payload.get('model') != name:
            raise ValueError('Wrong model or unsupported checkpoint format')
        if payload.get('smoke_only') and not allow_smoke:
            raise ValueError('Smoke checkpoint cannot be used as a trained benchmark baseline')
        self.cfg = Config(**payload['config']).validate()
        self.device = torch.device(device)
        self.model = make_model(name, self.cfg).to(self.device)
        self.model.load_state_dict(payload['state_dict'], strict=True)
        self.model.eval()
        self.model.selection_status = payload['selection_status']
        self.model.training_revision = 'new_baselines_v1'
        self.policy = self.model
        self.metadata = payload['metadata']

    def reset(self, scenario):
        cfg = self.cfg
        if not np.isclose(scenario.dt, cfg.dt):
            raise ValueError('Checkpoint timestep differs from scenario; retrain at the requested dt')
        for key, expected in [('max_accel', cfg.max_accel), ('max_agent_speed', cfg.max_speed),
                              ('wheelbase', cfg.wheelbase), ('vehicle_length', cfg.length), ('vehicle_width', cfg.width)]:
            if not np.isclose(scenario.sim_config.get(key, expected), expected):
                raise ValueError('Checkpoint dynamics mismatch: '+key)
        if bool(scenario.sim_config.get('steer_from_rest', False)) != cfg.steer_from_rest:
            raise ValueError('Checkpoint dynamics mismatch: steer_from_rest')
        self.generator = torch.Generator(device=self.device).manual_seed(int(scenario.seed))
        self.states, self.actions, self.returns = [], [], []
        self.roads, self.road_types = map_polylines(scenario.corridor, cfg)
        goals = []
        for a in scenario.agents:
            from RL.routing import agent_route
            _, tangent = agent_route(scenario.corridor, a).xy_from_frenet(a.dest_s, 0.)
            goals.append(np.r_[a.dest, tangent*a.desired_speed, np.arctan2(tangent[1], tangent[0])])
        self.goals = np.asarray(goals, np.float32)

    def compute_controls(self, agents, scenario, step):
        cfg = self.cfg
        current = np.asarray([np.r_[a.pos, a.vel, a.heading, cfg.length, cfg.width,
                                    float(not a.reached_destination)] for a in agents], np.float32)
        if not current[:, -1].any():
            return [(0., 0.) for _ in agents]
        if self.states:
            # Recover the executed control from the observed transition, including
            # interventions by shared filters. Never feed the rejected proposal back.
            observed, _, _ = inverse_controls(np.stack([self.states[-1], current], 1), cfg)
            self.actions[-1] = observed[:, 0]
        self.states.append(current)
        self.actions.append(np.zeros((len(agents), 2), np.float32))
        self.returns.append(np.zeros((len(agents), 3), np.float32))
        keep = max(cfg.context, cfg.history)+1
        self.states, self.actions, self.returns = [x[-keep:] for x in (self.states, self.actions, self.returns)]
        hist = np.stack(self.states, 1)
        act = np.stack(self.actions, 1)
        rtg = np.stack(self.returns, 1)
        controls = np.zeros((len(agents), 2), np.float32)
        # Scenes within capacity are sampled jointly once. Larger scenes use each
        # focal agent's nearest-neighbor group, without silently dropping agents.
        from RL.decision import neighbor_indices
        active_ids = np.flatnonzero(current[:, -1])
        whole_scene_visible = (len(agents) <= cfg.max_agents and
            all(len(neighbor_indices(agents, int(i), scenario.sim_config)) == len(active_ids)-1 for i in active_ids))
        focals = [int(active_ids[0])] if whole_scene_visible else active_ids
        for focal in focals:
            ids = np.asarray([focal]+neighbor_indices(agents, int(focal), scenario.sim_config))[:cfg.max_agents]
            goals = pad_agents(self.goals[ids], cfg)
            if self.name == 'ctrl_sim':
                states = self._window(hist[ids], cfg.context)
                actions = self._window(act[ids], cfg.context)
                returns = self._window(rtg[ids], cfg.context)
                features = ctrl_features(states, actions, returns, goals, self.roads, self.road_types, cfg)
                data = tensor_tree(features, self.device)
                tokens, selected = self.model.sample(data, self.generator)
                proposed = decode_actions(tokens[0].cpu().numpy(), cfg)
                selected = selected.cpu().numpy()/(cfg.return_bins-1)*2-1
            else:
                incoming = np.zeros_like(act); incoming[:, 1:] = act[:, :-1]
                states = self._window(hist[ids], cfg.history)
                actions = self._window(incoming[ids], cfg.history)
                cond, _ = ctg_features(states, actions, goals, self.roads, self.road_types, cfg)
                guidance = None
                if cfg.guidance_scale > 0:
                    initial = tensor_tree(pad_agents(current[ids], cfg), self.device)
                    guidance = make_guidance(initial, tensor_tree(goals, self.device), scenario.corridor, cfg)
                sample = self.model.sample(tensor_tree(cond, self.device), self.generator, guidance)
                proposed = sample[0, :, 0, -2:].cpu().numpy()*[cfg.max_accel, cfg.max_steer]
            rows = range(len(ids)) if whole_scene_visible else [0]
            for row in rows:
                controls[ids[row]] = proposed[row]
                if self.name == 'ctrl_sim':
                    self.returns[-1][ids[row]] = selected[row]
        if not np.isfinite(controls).all():
            raise FloatingPointError(self.name+' produced non-finite controls')
        controls = np.clip(controls, [-cfg.max_accel, -cfg.max_steer], [cfg.max_accel, cfg.max_steer])
        controls[current[:, -1] == 0] = 0
        return [tuple(map(float, row)) for row in controls]

    def _window(self, values, size):
        result = np.zeros((len(values), size)+values.shape[2:], values.dtype)
        length = min(size, values.shape[1])
        result[:, -length:] = values[:, -length:]
        return pad_agents(result, self.cfg)
