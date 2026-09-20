"""Shared rollout storage for direct categorical and continuous PPO baselines."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import numpy as np
import torch

from Baselines.discrete_action import feasible_action_mask, grid_control
from Baselines.dynamics import observation
from RL.transition import advance_agents


@dataclass
class PPOMemory:
    observations: list = field(default_factory=list)
    actions: list = field(default_factory=list)
    masks: list = field(default_factory=list)
    log_probs: list = field(default_factory=list)
    values: list = field(default_factory=list)
    rewards: list = field(default_factory=list)
    dones: list = field(default_factory=list)
    timeouts: list = field(default_factory=list)
    bootstrap_values: list = field(default_factory=list)
    traj_ids: list = field(default_factory=list)


def collect_episode(scenario, policy, memory, collision_penalty=0.0, *, discrete=False,
                    collision_event_penalty=0.0):
    initial_samples = len(memory.actions)
    agents = scenario.spawn_agents()
    dest_s = [a.dest_s for a in scenario.agents]
    trajectory_base = max(memory.traj_ids, default=-1) + 1
    total_reward = 0.0
    collisions = steps = 0
    previous_pairs: set[tuple[int, int]] = set()
    collided: set[int] = set()
    for step in range(scenario.max_steps):
        active = [i for i, agent in enumerate(agents)
                  if not agent.reached_destination and i not in collided]
        if not active:
            break
        controls = [(0.0, 0.0)] * len(agents)
        for i in active:
            obs = observation(agents, i, scenario)
            if discrete:
                mask = feasible_action_mask(i, agents[i], agents, scenario)
                action, logp, value = policy.sample_action(obs, mask)
                memory.masks.append(mask)
                controls[i] = grid_control(action, scenario.sim_config)
            else:
                action, logp, value = policy.sample_action(obs)
                controls[i] = policy.to_control(action)
            memory.observations.append(obs)
            memory.actions.append(action)
            memory.log_probs.append(logp)
            memory.values.append(value)
            memory.traj_ids.append(trajectory_base + i)

        result = advance_agents(
            agents, controls, scenario.corridor, scenario.sim_config, dest_s,
            leftover_coef=scenario.sim_config.get("leftover_coef", 0.08),
            arrival_bonus=scenario.sim_config.get("arrival_bonus", 8.0),
            collision_penalty=collision_penalty,
            collision_event_penalty=collision_event_penalty,
            previous_collision_pairs=previous_pairs,
        )
        previous_pairs = set(result.collision_pairs)
        collided |= set(result.colliding_agents)
        collisions += len(result.collision_pairs)
        steps = step + 1
        for i in active:
            terminal = bool(agents[i].reached_destination) or i in collided
            timeout = steps == scenario.max_steps and not terminal
            memory.rewards.append(result.rewards[i])
            memory.dones.append(float(terminal))
            memory.timeouts.append(float(timeout))
            boot = 0.0
            if timeout:
                obs = torch.as_tensor(observation(agents, i, scenario), dtype=torch.float32)
                with torch.no_grad():
                    _, value = policy.distribution(obs)
                boot = float(value)
            memory.bootstrap_values.append(boot)
            total_reward += result.rewards[i]
    return {"reward": total_reward / max(len(agents), 1), "collisions": float(collisions),
            "arrival_rate": float(np.mean([a.reached_destination for a in agents])),
            "steps": float(steps), "agent_steps": len(memory.actions)-initial_samples}


class PolicySelection:
    """Common safety feasibility and PDMS ranking, with explicit failed checkpoints."""

    def __init__(self, args, policy, model):
        from RL.experiment_protocol import TrainingBudget, source_manifest, validation_seeds
        self.args, self.policy, self.model = args, policy, model
        self.budget = TrainingBudget(args)
        if (args.val_episodes <= 0 or args.episodes_per_update <= 0 or args.episodes_per_update >= 1000
                or args.updates <= 0 or args.max_steps <= 0):
            raise ValueError('Positive episode/update settings and fewer than 1000 episodes per update required')
        if set(validation_seeds(args)) & set(validation_seeds(args, True)):
            raise ValueError('Validation and test scenario seeds must be disjoint')
        self.sources = source_manifest()
        self.best_score = None
        self.best_state = None
        self.selected_update = 0
        self.val_stats, self.history = {}, []
        self.prior = self.evaluate(validation_seeds(args), prior=True)

    def validation_config(self):
        from RL.experiment_protocol import validation_config
        from Baselines.utility_prior import UtilityPriorController
        prior = UtilityPriorController(calibration=getattr(self.args, 'calibration', None))
        return validation_config(self.args, self.scenario(0).sim_config, prior.params)

    def scenario(self, seed, max_steps=None, training=False):
        from Baselines.scenario import build_scenario
        kwargs = dict(seed=seed, num_agents=self.args.num_agents,
                      max_steps=self.args.max_steps if max_steps is None else max_steps,
                      run_id=self.args.run_id, lane_kf=self.args.lane_kf,
                      obb_safety_filter=bool(getattr(self.args, 'train_obb_filter', True)) if training else True)
        if getattr(self.args, 'dense_spawn', False):
            kwargs.update(spawn_s_range=(20., 80.), spawn_lateral_frac=.55, min_initial_spacing=5.)
        return build_scenario(**kwargs)

    def training_scenarios(self, update):
        from RL.experiment_protocol import training_seed
        for episode in range(self.args.episodes_per_update):
            if self.budget.exhausted:
                break
            yield self.scenario(training_seed(self.args, update, episode),
                                self.budget.remaining(self.args.max_steps), training=True)

    def evaluate(self, seeds, prior=False, controller=None):
        from Baselines.runner import rollout
        from Baselines.metrics import rollout_metrics
        from Baselines.direct_discrete_rl import DirectDiscreteRLController
        from Baselines.pure_rl import PureRLController
        from Baselines.marl import MARLController
        from Baselines.utility_prior import UtilityPriorController
        if controller is not None:
            pass
        elif prior:
            controller = UtilityPriorController(calibration=getattr(self.args, 'calibration', None))
        elif self.model == 'direct_discrete_rl':
            controller = DirectDiscreteRLController(policy=self.policy)
        elif self.model == 'pure_rl':
            controller = PureRLController(policy=self.policy)
        else:
            controller = MARLController(algo=self.model, policy=self.policy)
        rows = []
        for seed in seeds:
            scenario = self.scenario(seed)
            result = rollout(scenario, controller)
            leftover = sum(float(np.linalg.norm(result.positions[-1, i]-a.dest))
                           for i, a in enumerate(scenario.agents) if result.active[-1, i])
            rows.append(dict(rollout_metrics(result), collisions=float(result.collision_steps),
                             metric=leftover+10.*result.collision_steps))
        keys = ('collision_events', 'collisions', 'metric', 'arrival_rate', 'offroad_rate',
                'closed_loop_score', 'goal_progress', 'unsafe_ttc_rate', 'rms_jerk',
                'mean_abs_accel', 'mean_abs_steering', 'shield_intervention_rate')
        return {k: float(np.mean([r[k] for r in rows])) for k in keys}

    def consider(self, update, force=False):
        from RL.experiment_protocol import selection_key, validation_seeds, write_json
        if not self.budget.validation_due(self.args, update, force):
            return
        stats = self.evaluate(validation_seeds(self.args))
        score = selection_key(stats, self.prior)
        if self.best_score is None or score < self.best_score:
            self.best_score, self.val_stats = score, stats
            self.selected_update = update
            self.best_state = {k: v.detach().cpu().clone() for k, v in self.policy.state_dict().items()}
        self.history.append(dict(update=update, **self.budget.state(), **stats))
        if self.args.save is not None:
            write_json(self.args.save.with_suffix('.curve.json'), self.history)
        print(f'Validation update {update}: {stats}', flush=True)

    def finish(self):
        from RL.traffic_env import SPAWN_PROTOCOL_VERSION
        from RL.transition import DRIVING_REWARD_REVISION
        from RL.experiment_protocol import SELECTION_RULE, regressions, validation_seeds, write_json
        if self.best_state is None:
            self.consider(self.budget.updates, force=True)
        self.policy.load_state_dict(self.best_state)
        if getattr(self.args, 'skip_test', False) or self.args.test_episodes <= 0:
            test = {'skipped': True}
        else:
            test = self.evaluate(validation_seeds(self.args, test=True))
        metadata = dict(selected_update=self.selected_update, validation=self.val_stats,
            prior_validation=self.prior, validation_history=self.history,
            selection_rule=SELECTION_RULE,
            validation_config=self.validation_config(),
            selection_status='safety_failed' if regressions(self.val_stats, self.prior) else 'safety_feasible',
            safety_regressions=regressions(self.val_stats, self.prior),
            val_seeds=validation_seeds(self.args), test_seeds=validation_seeds(self.args, True),
            training_seed_rule='10000000 + seed + update*1000 + episode',
            spawn_protocol_version=SPAWN_PROTOCOL_VERSION, decision_protocol_version=1,
            collision_filter_revision=2, driving_reward_revision=DRIVING_REWARD_REVISION,
            test=test, train_seed=self.args.seed, budget=self.budget.state(), source_hashes=self.sources,
            stopping_reason='environment_budget' if self.budget.exhausted else 'update_limit',
            train_obb_filter=bool(getattr(self.args, 'train_obb_filter', True)),
            boundary_safety_filter=True, boundary_margin=.1, collision_penalty=self.args.collision_penalty,
            num_agents=self.args.num_agents, max_steps=self.args.max_steps)
        if self.args.save is not None:
            write_json(self.args.save.with_suffix('.summary.json'), metadata)
        return metadata


def add_validation_args(parser):
    from RL.experiment_protocol import add_budget_args
    add_budget_args(parser)
    parser.add_argument("--validation-seed-start", type=int, default=910000)
    parser.add_argument("--test-seed-start", type=int, default=810000)
    parser.add_argument("--calibration", type=Path, default=None)
    parser.add_argument("--dense-spawn", action="store_true")
    parser.add_argument("--val-every", type=int, default=10)
    parser.add_argument("--val-episodes", type=int, default=16)
    parser.add_argument("--test-episodes", type=int, default=16)
    parser.add_argument("--skip-test", action="store_true",
                        help="Skip held-out test evaluation after training")
