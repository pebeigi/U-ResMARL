"""Shared rollout storage for direct categorical and continuous PPO baselines."""

from __future__ import annotations

from dataclasses import dataclass, field
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


def collect_episode(scenario, policy, memory, collision_penalty=0.0, *, discrete=False):
    agents = scenario.spawn_agents()
    dest_s = [a.dest_s for a in scenario.agents]
    trajectory_base = max(memory.traj_ids, default=-1) + 1
    total_reward = 0.0
    collisions = steps = 0
    for step in range(scenario.max_steps):
        active = [i for i, agent in enumerate(agents) if not agent.reached_destination]
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
            leftover_coef=scenario.sim_config.get("leftover_coef", 0.05),
            arrival_bonus=scenario.sim_config.get("arrival_bonus", 5.0),
            collision_penalty=collision_penalty,
        )
        collisions += len(result.collision_pairs)
        steps = step + 1
        for i in active:
            terminal = bool(agents[i].reached_destination)
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
            "steps": float(steps)}


class PolicySelection:
    """Deterministic, safety-first validation shared by the learned baselines."""

    def __init__(self, args, policy, model):
        self.args, self.policy, self.model = args, policy, model
        self.best_score = (float("inf"), float("inf"))
        self.best_state = None
        self.selected_update = 0
        self.val_stats = {}

    def evaluate(self, seeds):
        from Baselines.scenario import build_scenario
        from Baselines.runner import rollout
        from Baselines.direct_discrete_rl import DirectDiscreteRLController
        from Baselines.pure_rl import PureRLController
        from Baselines.marl import MARLController

        if self.model == "direct_discrete_rl":
            controller = DirectDiscreteRLController(policy=self.policy)
        elif self.model == "pure_rl":
            controller = PureRLController(policy=self.policy)
        else:
            controller = MARLController(algo=self.model, policy=self.policy)
        rows = []
        for seed in seeds:
            scenario = build_scenario(seed=seed, num_agents=self.args.num_agents,
                                      max_steps=self.args.max_steps, run_id=self.args.run_id,
                                      lane_kf=self.args.lane_kf, obb_safety_filter=True)
            result = rollout(scenario, controller)
            leftover = sum(float(np.linalg.norm(result.positions[-1, i] - a.dest))
                           for i, a in enumerate(scenario.agents) if result.active[-1, i])
            rows.append([result.collision_steps, leftover + 10.0 * result.collision_steps,
                         float(np.mean(result.arrival_step >= 0))])
        coll, metric, arrival = np.mean(rows, axis=0)
        return {"collisions": float(coll), "metric": float(metric), "arrival_rate": float(arrival)}

    def consider(self, update):
        if update % max(1, getattr(self.args, "val_every", 10)) and update != self.args.updates:
            return
        seeds = [self.args.seed + 900_000 + i for i in range(max(1, getattr(self.args, "val_episodes", 8)))]
        stats = self.evaluate(seeds)
        score = (stats["collisions"], stats["metric"])
        if score < self.best_score:
            self.best_score, self.val_stats = score, stats
            self.selected_update = update
            self.best_state = {k: v.detach().cpu().clone() for k, v in self.policy.state_dict().items()}
        print(f"Validation update {update}: {stats}")

    def finish(self):
        if self.best_state is not None:
            self.policy.load_state_dict(self.best_state)
        seeds = [self.args.seed + 700_000 + i for i in range(max(1, getattr(self.args, "test_episodes", 16)))]
        test = self.evaluate(seeds)
        print(f"Held-out TEST: {test}")
        return {"selected_update": self.selected_update, "validation": self.val_stats,
                "test": test, "train_seed": self.args.seed,
                "train_obb_filter": bool(getattr(self.args, "train_obb_filter", False)),
                "boundary_safety_filter": True, "boundary_margin": 0.1,
                "collision_penalty": self.args.collision_penalty,
                "num_agents": self.args.num_agents, "max_steps": self.args.max_steps}


def add_validation_args(parser):
    parser.add_argument("--val-every", type=int, default=10)
    parser.add_argument("--val-episodes", type=int, default=8)
    parser.add_argument("--test-episodes", type=int, default=16)
