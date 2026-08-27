"""PPO training for the pure-RL baseline (no behavioural prior).

Uses the same scenarios, dynamics and reward as the residual model, so the only
difference between the two is whether the policy acts through the calibrated
utility function or directly on the controls.

    python -m Baselines.train_pure_rl --updates 100 --episodes-per-update 4
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

import Baselines._paths  # noqa: F401
from Baselines.dynamics import (
    DEFAULT_REWARD_WEIGHTS,
    apply_control,
    compute_reward,
    hold_still,
    observation,
    observation_dim,
)
from Baselines.pure_rl import PureRLPolicy
from Baselines.scenario import Scenario, build_scenario
from RL.corridor import DEFAULT_LANE_KF, DEFAULT_RUN_ID, boxes_overlap

try:
    import torch
    import torch.nn as nn
except ImportError as exc:  # pragma: no cover
    raise SystemExit("PyTorch is required for training. Install with: pip install torch") from exc


@dataclass
class PPOMemory:
    observations: list = field(default_factory=list)
    actions: list = field(default_factory=list)
    log_probs: list = field(default_factory=list)
    values: list = field(default_factory=list)
    rewards: list = field(default_factory=list)
    dones: list = field(default_factory=list)


def run_episode(
    scenario: Scenario,
    policy: PureRLPolicy,
    memory: PPOMemory,
    collision_penalty: float = 0.0,
) -> dict[str, float]:
    agents = scenario.spawn_agents()
    dest_s = np.array([a.dest_s for a in scenario.agents], dtype=float)
    tol = float(scenario.sim_config.get("destination_threshold", 1.0))
    length = scenario.vehicle_length
    width = scenario.vehicle_width
    n = len(agents)

    total_reward = 0.0
    collisions = 0
    steps = 0

    for step in range(scenario.max_steps):
        active = [i for i, a in enumerate(agents) if not a.reached_destination]
        if not active:
            break

        reward_slots: list[int] = []
        controls: dict[int, tuple[float, float]] = {}
        for i in active:
            obs = observation(agents, i, scenario)
            action, log_prob, value = policy.sample_action(obs)
            memory.observations.append(obs)
            memory.actions.append(action)
            memory.log_probs.append(log_prob)
            memory.values.append(value)
            reward_slots.append(i)
            controls[i] = policy.to_control(action)

        for i in active:
            apply_control(agents[i], controls[i], scenario)

        for i in active:
            s, _, _, _, _ = scenario.corridor.project(agents[i].pos)
            if s >= dest_s[i] - tol:
                agents[i].reached_destination = True
                hold_still(agents[i])

        hit: set[int] = set()
        for a_idx in range(n):
            if agents[a_idx].reached_destination:
                continue
            for b_idx in range(a_idx + 1, n):
                if agents[b_idx].reached_destination:
                    continue
                if boxes_overlap(
                    agents[a_idx].pos,
                    agents[a_idx].heading,
                    agents[b_idx].pos,
                    agents[b_idx].heading,
                    length=length,
                    width=width,
                ):
                    hit.update((a_idx, b_idx))
        collisions += len(hit)

        steps = step + 1
        done = all(a.reached_destination for a in agents) or steps >= scenario.max_steps
        for i in reward_slots:
            r = compute_reward(agents, i, scenario, controls[i], DEFAULT_REWARD_WEIGHTS)
            if collision_penalty and i in hit:
                r -= collision_penalty
            memory.rewards.append(float(r))
            memory.dones.append(float(done))
            total_reward += float(r)

    arrived = sum(a.reached_destination for a in agents)
    return {
        "reward": total_reward / max(n, 1),
        "collisions": float(collisions),
        "arrival_rate": arrived / max(n, 1),
        "steps": float(steps),
    }


def compute_gae(
    rewards: np.ndarray,
    values: np.ndarray,
    dones: np.ndarray,
    gamma: float,
    gae_lambda: float,
) -> tuple[np.ndarray, np.ndarray]:
    advantages = np.zeros_like(rewards, dtype=np.float32)
    last_gae = 0.0
    next_value = 0.0
    for t in reversed(range(len(rewards))):
        nonterminal = 1.0 - dones[t]
        delta = rewards[t] + gamma * next_value * nonterminal - values[t]
        last_gae = delta + gamma * gae_lambda * nonterminal * last_gae
        advantages[t] = last_gae
        next_value = values[t]
    return advantages, advantages + values


def ppo_update(
    policy: PureRLPolicy,
    optimizer: "torch.optim.Optimizer",
    memory: PPOMemory,
    args: argparse.Namespace,
) -> dict[str, float]:
    obs = torch.as_tensor(np.array(memory.observations), dtype=torch.float32)
    actions = torch.as_tensor(np.array(memory.actions), dtype=torch.float32)
    old_log_probs = torch.as_tensor(np.array(memory.log_probs), dtype=torch.float32)
    values_np = np.array(memory.values, dtype=np.float32)
    rewards_np = np.array(memory.rewards, dtype=np.float32)
    dones_np = np.array(memory.dones, dtype=np.float32)

    advantages_np, returns_np = compute_gae(
        rewards_np, values_np, dones_np, args.gamma, args.gae_lambda
    )
    advantages = torch.as_tensor(advantages_np, dtype=torch.float32)
    returns = torch.as_tensor(returns_np, dtype=torch.float32)
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    n = obs.shape[0]
    indices = np.arange(n)
    stats = {"loss": 0.0, "policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0}
    for _ in range(args.ppo_epochs):
        np.random.shuffle(indices)
        for start in range(0, n, args.minibatch_size):
            batch = indices[start : start + args.minibatch_size]
            dist, value = policy.distribution(obs[batch])
            new_log_probs = dist.log_prob(actions[batch]).sum(dim=-1)
            entropy = dist.entropy().sum(dim=-1).mean()

            ratio = torch.exp(new_log_probs - old_log_probs[batch])
            unclipped = ratio * advantages[batch]
            clipped = torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef) * advantages[batch]
            policy_loss = -torch.min(unclipped, clipped).mean()
            value_loss = 0.5 * (returns[batch] - value).pow(2).mean()
            loss = policy_loss + args.value_coef * value_loss - args.entropy_coef * entropy

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(policy.parameters(), max_norm=0.5)
            optimizer.step()

            stats = {
                "loss": float(loss.detach()),
                "policy_loss": float(policy_loss.detach()),
                "value_loss": float(value_loss.detach()),
                "entropy": float(entropy.detach()),
            }

    # Refresh the observation whitening after the update so that the sampled
    # log-probabilities stay consistent with the statistics used to collect them.
    policy.obs_norm.update(obs)
    return stats


def train(args: argparse.Namespace) -> None:
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)

    probe = build_scenario(
        seed=args.seed,
        num_agents=args.num_agents,
        max_steps=args.max_steps,
        run_id=args.run_id,
        lane_kf=args.lane_kf,
    )
    policy = PureRLPolicy(
        obs_dim=observation_dim(probe),
        hidden_dim=args.hidden_dim,
        max_accel=float(probe.sim_config.get("max_accel", 4.0)),
        init_log_std=args.init_log_std,
    )
    optimizer = torch.optim.Adam(policy.parameters(), lr=args.lr)

    best_reward = -float("inf")
    for update in range(1, args.updates + 1):
        memory = PPOMemory()
        episode_stats: list[dict[str, float]] = []
        for _ in range(args.episodes_per_update):
            scenario = build_scenario(
                seed=int(rng.integers(0, 1_000_000)),
                num_agents=args.num_agents,
                max_steps=args.max_steps,
                run_id=args.run_id,
                lane_kf=args.lane_kf,
            )
            episode_stats.append(
                run_episode(scenario, policy, memory, collision_penalty=args.collision_penalty)
            )

        stats = ppo_update(policy, optimizer, memory, args)
        mean_reward = float(np.mean([s["reward"] for s in episode_stats]))
        best_reward = max(best_reward, mean_reward)
        if update == 1 or update % max(args.log_every, 1) == 0:
            print(
                f"Update {update:4d}/{args.updates} | reward={mean_reward:9.3f} | "
                f"best={best_reward:9.3f} | "
                f"collisions={np.mean([s['collisions'] for s in episode_stats]):6.2f} | "
                f"arrived={np.mean([s['arrival_rate'] for s in episode_stats]):5.2f} | "
                f"entropy={stats['entropy']:6.3f}"
            )

    if args.save is not None:
        args.save.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "state_dict": policy.state_dict(),
                "obs_dim": policy.obs_dim,
                "hidden_dim": policy.hidden_dim,
                "max_accel": float(policy.action_scale[0].item()),
                "max_steering": float(policy.action_scale[1].item()),
                "algo": "ppo",
                "model": "pure_rl",
                "run_id": args.run_id,
                "lane_kf": args.lane_kf,
            },
            args.save,
        )
        print(f"Saved pure-RL policy to {args.save}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the pure-RL baseline (no utility prior)")
    parser.add_argument("--updates", type=int, default=100)
    parser.add_argument("--episodes-per-update", type=int, default=4)
    parser.add_argument("--num-agents", type=int, default=10)
    parser.add_argument("--max-steps", type=int, default=240)
    parser.add_argument("--run-id", type=int, default=DEFAULT_RUN_ID)
    parser.add_argument("--lane-kf", type=int, default=DEFAULT_LANE_KF)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--clip-coef", type=float, default=0.2)
    parser.add_argument("--value-coef", type=float, default=0.5)
    parser.add_argument("--entropy-coef", type=float, default=0.01)
    parser.add_argument("--ppo-epochs", type=int, default=4)
    parser.add_argument("--minibatch-size", type=int, default=256)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--init-log-std", type=float, default=-1.6)
    parser.add_argument("--collision-penalty", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--log-every", type=int, default=5)
    parser.add_argument("--save", type=Path, default=Path("Baselines/checkpoints/pure_rl_policy.pt"))
    train(parser.parse_args())


if __name__ == "__main__":
    main()
