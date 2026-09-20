"""Causal scene tensors and offline targets for the local traffic simulator."""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace as NS

import numpy as np
import torch

from .config import Config


def wrap(x):
    return (x + np.pi) % (2*np.pi) - np.pi


def rotation(angle):
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, -s], [s, c]], dtype=np.float32)


def action_tokens(actions, cfg):
    bins = np.array([cfg.accel_bins, cfg.steer_bins])
    limits = np.array([cfg.max_accel, cfg.max_steer])
    indices = np.rint((np.clip(actions/limits, -1, 1)+1)/2*(bins-1)).astype(np.int64)
    return indices[..., 0]*cfg.steer_bins + indices[..., 1]


def decode_actions(tokens, cfg):
    indices = np.stack([tokens//cfg.steer_bins, tokens % cfg.steer_bins], -1)
    return (indices/np.array([cfg.accel_bins-1, cfg.steer_bins-1])*2-1)*np.array([cfg.max_accel, cfg.max_steer])


def inverse_controls(states, cfg):
    """Invert the repository's semi-implicit bicycle, with explicit clip diagnostics.

    States are [agent,time,(x,y,vx,vy,heading,length,width,exists)]. Labels at t
    cause state t+1. Fitting uses only consecutive recorded positions; no central
    differences or future values enter the state at t.
    """
    v = np.linalg.norm(states[..., 2:4], axis=-1)
    displacement = np.diff(states[..., :2], axis=1)
    target_speed = np.linalg.norm(displacement, axis=-1)/cfg.dt
    target_heading = np.arctan2(displacement[..., 1], displacement[..., 0])
    target_heading = np.where(target_speed > .05, target_heading, states[:, :-1, 4])
    yaw_speed = v[:, :-1]
    if cfg.steer_from_rest:
        yaw_speed = np.maximum(np.maximum(yaw_speed, target_speed), cfg.min_steer_speed)
    accel = (target_speed-v[:, :-1])/cfg.dt
    steer = np.arctan(wrap(target_heading-states[:, :-1, 4])*cfg.wheelbase/np.maximum(yaw_speed*cfg.dt, 1e-6))
    raw = np.stack([accel, steer], -1)
    valid = states[:, :-1, -1]*states[:, 1:, -1]
    clipped = np.clip(raw, [-cfg.max_accel, -cfg.max_steer], [cfg.max_accel, cfg.max_steer])
    actions = np.zeros(states.shape[:2]+(2,), np.float32)
    actions[:, :-1] = clipped*valid[..., None]
    mask = np.zeros(states.shape[:2], np.float32)
    mask[:, :-1] = valid
    v_next = np.clip(v[:, :-1]+clipped[..., 0]*cfg.dt, 0, cfg.max_speed)
    h_next = states[:, :-1, 4]+yaw_speed/cfg.wheelbase*np.tan(clipped[..., 1])*cfg.dt
    pred = states[:, :-1, :2]+v_next[..., None]*np.stack([np.cos(h_next), np.sin(h_next)], -1)*cfg.dt
    error = np.linalg.norm(pred-states[:, 1:, :2], axis=-1)
    return actions, mask, {'valid_transitions': int(valid.sum()),
        'clipped_transitions': int(((np.abs(raw-clipped) > 1e-6).any(-1)*valid).sum()),
        'inverse_position_error_sum_m': float((error*valid).sum()),
        'inverse_position_error_max_m': float((error*valid).max())}


def map_polylines(corridor, cfg):
    roads, types = [], []
    if hasattr(corridor, 'map_boundaries'):
        # A network has polygon rings, not paired edges or a global centerline.
        lines = corridor.map_boundaries
        for j in range(cfg.map_segments):
            line = lines[j % len(lines)]
            count = len(range(j % len(lines), cfg.map_segments, len(lines)))
            part = j // len(lines)
            indices = np.linspace((len(line)-1)*part/count, (len(line)-1)*(part+1)/count, cfg.map_points)
            xy = np.stack([np.interp(indices, np.arange(len(line)), line[:, k]) for k in range(2)], -1)
            roads.append(np.c_[xy, np.ones(cfg.map_points)])
            typ = np.zeros(8); typ[1] = 1
            types.append(typ)
        return np.asarray(roads, np.float32), np.asarray(types, np.float32)
    # Four segments each for centerline and two physical edges; known map only.
    for kind, line in enumerate((corridor.center, corridor.lower, corridor.upper)):
        chunks = np.linspace(0, len(line)-1, cfg.map_segments//3+1)
        for lo, hi in zip(chunks[:-1], chunks[1:]):
            indices = np.linspace(lo, hi, cfg.map_points)
            xy = np.stack([np.interp(indices, np.arange(len(line)), line[:, k]) for k in range(2)], -1)
            roads.append(np.c_[xy, np.ones(cfg.map_points)])
            typ = np.zeros(8); typ[0 if kind == 0 else 1] = 1
            types.append(typ)
    if len(roads) != cfg.map_segments:
        raise ValueError('map_segments must be divisible by three')
    return np.asarray(roads, np.float32), np.asarray(types, np.float32)


def scene_arrays(scene, cfg):
    from RL.corridor import boxes_overlap
    sc = scene.scenario
    initial_headings = np.array([a.heading for a in sc.agents])
    # Drop first history sample whose backward velocity is unknown.
    pos = scene.reference_positions[1:].transpose(1, 0, 2)
    vel = np.diff(scene.reference_positions, axis=0).transpose(1, 0, 2)/cfg.dt
    exists = np.isfinite(pos).all(-1) & np.isfinite(vel).all(-1)
    headings = np.arctan2(vel[..., 1], vel[..., 0])
    for t in range(headings.shape[1]):
        headings[:, t] = np.where(np.linalg.norm(vel[:, t], axis=-1) > .05,
                                  headings[:, t], headings[:, t-1] if t else initial_headings)
    states = np.concatenate([pos, vel, headings[..., None],
        np.broadcast_to([cfg.length, cfg.width], pos.shape), exists[..., None]], -1)
    states = np.nan_to_num(states).astype(np.float32)
    actions, action_mask, diagnostics = inverse_controls(states, cfg)
    goals = []
    for agent in sc.agents:
        from RL.routing import agent_route
        _, tangent = agent_route(sc.corridor, agent).xy_from_frenet(agent.dest_s, 0.)
        goals.append(np.r_[agent.dest, tangent*agent.desired_speed, np.arctan2(tangent[1], tangent[0])])
    roads, road_types = map_polylines(sc.corridor, cfg)
    n, length = states.shape[:2]
    rewards = np.zeros((n, length, 3), np.float32)
    # Environment-specific replacements for Waymo goal/vehicle/edge rewards.
    # Dense bounded progress, nearest footprint gap and signed road clearance.
    for t in range(length-1):
        for i in range(n):
            if not action_mask[i, t]:
                continue
            p, q = states[i, t, :2], states[i, t+1, :2]
            if hasattr(sc.corridor, 'remaining_to_goal'):
                goal = sc.agents[i].dest
                s0 = -sc.corridor.remaining_to_goal(p, goal)
                s1 = -sc.corridor.remaining_to_goal(q, goal)
            else:
                s0 = sc.corridor.project(p)[0]; s1 = sc.corridor.project(q)[0]
            rewards[i, t, 0] = np.clip((s1-s0)/(cfg.max_speed*cfg.dt), -1, 1)
            gap = 15.
            collision = False
            for j in range(n):
                if i != j and exists[j, t+1]:
                    gap = min(gap, np.linalg.norm(q-states[j, t+1, :2])-cfg.length)
                    collision |= boxes_overlap(q, states[i, t+1, 4], states[j, t+1, :2], states[j, t+1, 4], cfg.length, cfg.width)
            rewards[i, t, 1] = -1. if collision else np.clip(gap/15., 0, 1)
            heading = states[i, t+1, 4]
            corners = np.array([[x, y] for x in (-cfg.length/2, cfg.length/2) for y in (-cfg.width/2, cfg.width/2)]) @ rotation(heading).T + q
            clearance = min(min(sc.corridor.clearances(corner)[:2]) for corner in corners)
            rewards[i, t, 2] = np.clip(clearance/3., -1, 1)
    rtgs = np.zeros_like(rewards)
    return_mask = np.zeros_like(action_mask)
    for t in range(length-cfg.horizon):
        valid = action_mask[:, t:t+cfg.horizon].all(1)
        rtgs[:, t] = rewards[:, t:t+cfg.horizon].mean(1)
        return_mask[:, t] = valid
    return dict(states=states, actions=actions, action_mask=action_mask, rtgs=rtgs,
                return_mask=return_mask, goals=np.asarray(goals, np.float32),
                roads=roads, road_types=road_types), diagnostics


def prepare(csv, output, cfg, count=20, split='train', seed=0, run_id=2, lane_kf=1):
    from Baselines.data_evaluation import build_recorded_scenes
    if split not in ('train', 'validation'):
        raise ValueError('Training caches may contain train or validation data only')
    future_steps = cfg.context + cfg.horizon + 1
    scenes, manifest = build_recorded_scenes(Path(csv), count=count, seed=seed,
        run_id=run_id, lane_kf=lane_kf, dt=cfg.dt, horizons=[future_steps*cfg.dt],
        split=split, history_s=cfg.dt*max(cfg.history, 2))
    arrays, diagnostics = {}, []
    for i, scene in enumerate(scenes):
        values, diag = scene_arrays(scene, cfg)
        arrays.update({str(i)+'_'+k: v for k, v in values.items()})
        diagnostics.append(diag)
    output = Path(output); output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, **arrays)
    manifest.update(config=cfg.to_dict(), scene_count=len(scenes), inverse_dynamics=diagnostics,
                    new_baselines_data_version=1)
    output.with_suffix('.json').write_text(json.dumps(manifest, indent=2)+'\n')
    return manifest


def select_group(states, focal, cfg, present=-1):
    distance = np.linalg.norm(states[:, present, :2]-states[focal, present, :2], axis=-1)
    distance = np.where(states[:, present, -1] > 0, distance, np.inf)
    distance[focal] = -1
    eligible = np.flatnonzero(distance <= cfg.perception_radius)
    order = eligible[np.argsort(distance[eligible], kind='stable')]
    return order[:min(cfg.max_agents, cfg.max_neighbors+1)]


def pad_agents(array, cfg):
    out = np.zeros((cfg.max_agents,)+array.shape[1:], dtype=array.dtype)
    out[:len(array)] = array
    return out


def ctrl_features(states, actions, rtgs, goals, roads, road_types, cfg):
    """Inputs contain the supplied history only; current action/RTG masked by core."""
    states, actions, rtgs, goals = [x.copy() for x in (states, actions, rtgs, goals)]
    first = np.flatnonzero(states[0, :, -1])
    origin = states[0, first[0] if len(first) else -1, :2].copy()
    angle = -states[0, first[0] if len(first) else -1, 4]
    rot = rotation(angle)
    states[..., :2] = (states[..., :2]-origin)@rot.T
    states[..., 2:4] = states[..., 2:4]@rot.T
    states[..., 4] = wrap(states[..., 4]+angle)
    goals[..., :2] = (goals[..., :2]-origin)@rot.T
    goals[..., 2:4] = goals[..., 2:4]@rot.T
    goals[..., 4] = wrap(goals[..., 4]+angle)
    roads = roads.copy(); roads[..., :2] = (roads[..., :2]-origin)@rot.T
    types = np.zeros((cfg.max_agents, 5), np.float32); types[:, 0] = 1
    times = np.broadcast_to(np.arange(states.shape[1])[None, :, None], states.shape[:2]+(1,)).copy()
    returns = np.rint((np.clip(rtgs, -1, 1)+1)/2*(cfg.return_bins-1)).astype(np.int64)
    return {'agent': dict(agent_states=states, agent_types=types, goals=goals,
            actions=action_tokens(actions, cfg), rtgs=returns, timesteps=times),
            'map': dict(road_points=roads, road_types=road_types)}


def ctg_features(states, incoming, goals, roads, road_types, cfg, future=None, future_actions=None):
    """Agent-centric +y coordinates, joint relative context, no future conditioning."""
    n, h = states.shape[:2]
    past = states.copy(); goals = goals.copy()
    present = states[:, -1].copy()
    relative = np.zeros((n, n, h, 7), np.float32)
    maps = np.broadcast_to(roads[None], (n,)+roads.shape).copy()
    target = None if future is None else future[..., :5].copy()
    for i in range(n):
        angle = np.pi/2-present[i, 4]; rot = rotation(angle)
        past[i, :, :2] = (states[i, :, :2]-present[i, :2])@rot.T/cfg.pos_scale
        past[i, :, 2:4] = states[i, :, 2:4]@rot.T/cfg.vel_scale
        past[i, :, 4] = wrap(states[i, :, 4]+angle)
        goals[i, :2] = (goals[i, :2]-present[i, :2])@rot.T/cfg.pos_scale
        goals[i, 2:4] = goals[i, 2:4]@rot.T/cfg.vel_scale
        goals[i, 4] = wrap(goals[i, 4]+angle)
        maps[i, ..., :2] = (maps[i, ..., :2]-present[i, :2])@rot.T/cfg.pos_scale
        relative[i, ..., :2] = (states[..., :2]-present[i, :2])@rot.T
        dh = states[..., 4]-present[i, 4]
        relative[i, ..., 2] = np.cos(dh); relative[i, ..., 3] = np.sin(dh)
        relative[i, ..., 4:6] = (states[..., 2:4]-present[i, 2:4])@rot.T
        relative[i, ..., 6] = np.linalg.norm(states[..., :2]-states[i, :, :2], axis=-1)
        if target is not None:
            target[i, :, :2] = (future[i, :, :2]-present[i, :2])@rot.T/cfg.pos_scale
            target[i, :, 2:4] = future[i, :, 2:4]@rot.T/cfg.vel_scale
            target[i, :, 4] = wrap(future[i, :, 4]+angle)
    types = np.zeros((n, 5), np.float32); types[:, 0] = 1
    times = np.zeros((n, h+cfg.horizon, 1), np.int64)
    cond = (past, incoming/np.array([cfg.max_accel, cfg.max_steer], np.float32), relative,
            np.repeat(relative[:, :, -1:], cfg.horizon, axis=2), types, goals, times,
            np.zeros((n, h, 3), np.int64), maps, np.broadcast_to(road_types[None], (n,)+road_types.shape).copy(),
            present[:, -1], np.c_[present[:, :2], np.pi/2-present[:, 4]])
    if target is not None:
        target = np.concatenate([target, future_actions/np.array([cfg.max_accel, cfg.max_steer])], -1).astype(np.float32)
    return cond, target


def tensor_tree(value, device='cpu', batch=True):
    if isinstance(value, dict):
        return {k: tensor_tree(v, device, batch) for k, v in value.items()}
    if isinstance(value, tuple):
        return tuple(tensor_tree(v, device, batch) for v in value)
    result = torch.as_tensor(np.asarray(value), device=device)
    if result.is_floating_point():
        result = result.float()
    return result.unsqueeze(0) if batch else result


def ctrl_namespaces(data):
    return {k: NS(**v) for k, v in data.items()}
