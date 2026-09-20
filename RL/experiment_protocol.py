"""Shared validation ranking, scenario seeds, budgets and experiment provenance."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import time

SELECTION_RULE = 'common_safety_then_pdms_v1'
SAFETY_KEYS = ('collision_events', 'collisions', 'offroad_rate')


def regressions(candidate, prior):
    return [key for key in SAFETY_KEYS if key in prior and
            (not math.isfinite(float(candidate.get(key, float('nan')))) or
             float(candidate[key]) > float(prior[key])+1e-8)]


def validation_score(stats):
    from RL.closed_loop_score import closed_loop_score
    pdms = float(stats.get('closed_loop_score', closed_loop_score(stats)))
    values = (-pdms, float(stats.get('collision_events', stats['collisions'])),
              float(stats['collisions']), -float(stats['arrival_rate']), float(stats['metric']))
    return tuple(v if math.isfinite(v) else float('inf') for v in values)


def selection_key(stats, prior):
    failures = regressions(stats, prior)
    # Failed models retain their own least-violating checkpoint for honest reporting.
    if failures:
        excess = tuple(max(0., float(stats.get(k, float('inf')))-float(prior.get(k, 0.)))
                       if math.isfinite(float(stats.get(k, float('nan')))) else float('inf') for k in SAFETY_KEYS)
        return (1, *excess, *validation_score(stats))
    return (0, 0., 0., 0., *validation_score(stats))


def validation_seeds(args, test=False):
    start = int(getattr(args, 'test_seed_start' if test else 'validation_seed_start', 810000 if test else 910000))
    count = int(getattr(args, 'test_episodes' if test else 'val_episodes', 16))
    return list(range(start, start+count))


def training_seed(args, update, episode):
    seed = int(10_000_000+args.seed+update*1000+episode)
    if seed in set(validation_seeds(args)+validation_seeds(args, True)):
        raise ValueError('Training and evaluation seeds overlap; use another seed block')
    return seed


def validation_config(args, sim, prior_params):
    """Comparable validation conditions; excludes algorithm-specific architecture."""
    dense = bool(getattr(args, 'dense_spawn', False))
    keys = ('dt', 'run_id', 'lane_kf', 'perception_radius', 'max_neighbors',
            'vehicle_length', 'vehicle_width', 'wheelbase', 'max_agent_speed',
            'max_accel', 'conflict_horizon', 'conflict_substeps', 'boundary_margin')
    result = dict(num_agents=int(args.num_agents), max_steps=int(args.max_steps),
                spawn_s_range=[20., 80. if dense else 120.],
                spawn_lateral_frac=.55 if dense else .35,
                min_initial_spacing=5. if dense else 8.,
                dynamics={key: sim.get(key) for key in keys},
                boundary_filter=True, obb_filter=True,
                prior_params={key: float(value) for key, value in prior_params.items()})
    if 'site_protocol' in sim:
        for key in ('spawn_s_range', 'spawn_lateral_frac', 'min_initial_spacing'):
            result.pop(key)
        result['site_protocol'] = sim['site_protocol']
    return result


class TrainingBudget:
    def __init__(self, args, state=None):
        self.limit = getattr(args, 'total_env_steps', None)
        if self.limit is not None and self.limit <= 0:
            raise ValueError('total_env_steps must be positive')
        if getattr(args, 'val_every_env_steps', 0) < 0:
            raise ValueError('val_every_env_steps cannot be negative')
        state = state or {}
        self.env_steps = int(state.get('environment_steps', 0))
        self.agent_steps = int(state.get('active_agent_transitions', 0))
        self.optimizer_steps = int(state.get('optimizer_steps', 0))
        self.updates = int(state.get('policy_updates', 0))
        self.prior_time = float(state.get('wall_time_s', 0.))
        self.started = time.perf_counter()
        self.next_validation = int(state.get('next_validation', 0))

    @property
    def exhausted(self):
        return self.limit is not None and self.env_steps >= self.limit

    def remaining(self, horizon):
        return int(horizon if self.limit is None else min(horizon, max(0, self.limit-self.env_steps)))

    def add(self, env_steps, agent_steps):
        self.env_steps += int(env_steps); self.agent_steps += int(agent_steps)

    def watch(self, optimizer):
        if optimizer is not None:
            def hook(*_):
                self.optimizer_steps += 1
            optimizer.register_step_post_hook(hook)

    def validation_due(self, args, update, force=False):
        interval = int(getattr(args, 'val_every_env_steps', 0))
        if interval:
            if not self.next_validation:
                self.next_validation = interval
            due = self.env_steps >= self.next_validation
            if due:
                self.next_validation = (self.env_steps//interval+1)*interval
            return force or due or self.exhausted or update == args.updates
        return force or self.exhausted or update % max(1, args.val_every) == 0 or update == args.updates

    def state(self):
        return dict(environment_steps=self.env_steps, active_agent_transitions=self.agent_steps,
                    optimizer_steps=self.optimizer_steps, policy_updates=self.updates,
                    wall_time_s=self.prior_time+time.perf_counter()-self.started,
                    requested_environment_steps=self.limit, next_validation=self.next_validation,
                    budget_reached=self.exhausted)


def add_budget_args(parser):
    parser.add_argument('--total-env-steps', type=int, default=None,
                        help='Exact interaction cap; --updates remains a second stopping limit')
    parser.add_argument('--val-every-env-steps', type=int, default=0,
                        help='Validate at update boundaries crossing this interaction interval')


def source_manifest():
    root = Path(__file__).resolve().parents[1]
    sources = list((root/'RL').glob('*.py'))+list((root/'Baselines').glob('*.py'))
    sources += list((root/'New Baselines/new_baselines').rglob('*.py'))
    sources += list((root/'TGSIM Case').glob('*.py'))
    sources += list((root/'Roundabout Case').glob('*.py'))
    sources += [root/'Calibration/utility_calibration_tgsim.json',
                root/'data/TGSIM FB/derived_boundaries/street_boundaries.csv']
    sources += [root/'Calibration/utility_calibration_jounieh.json',
                root/'data/Lebanon_Jounieh/Jounieh_Road_Boundaries.csv']
    sources += [root/'utility_model.py', root/'Calibration/utility_calibration.json']
    return {p.relative_to(root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(sources) if p.is_file()}


def write_json(path, values):
    def clean(value):
        if isinstance(value, dict): return {str(k): clean(v) for k, v in value.items()}
        if isinstance(value, (tuple, list)): return [clean(v) for v in value]
        if isinstance(value, Path): return str(value)
        if isinstance(value, float) and not math.isfinite(value): return None
        if hasattr(value, 'item'): return clean(value.item())
        return value
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(clean(values), indent=2, allow_nan=False)+'\n')
    temporary.replace(path)
