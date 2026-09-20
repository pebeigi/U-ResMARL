"""PPO training for the matched direct discrete RL baseline.

Uses the same scenarios, observations, reward, candidate grid, and OBB conflict
rejection as residual MARL. The only difference is whether the policy outputs
logits over discrete controls or residuals on the utility parameters.

    python -m Baselines.train_direct_discrete_rl --updates 100 --collision-penalty 8
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

import Baselines._paths  # noqa: F401
from Baselines.direct_discrete_rl import DirectDiscretePolicy
from Baselines.discrete_action import feasible_action_mask, grid_control, num_grid_actions
from Baselines.dynamics import (
    DEFAULT_REWARD_WEIGHTS,
    apply_control,
    compute_reward,
    hold_still,
    observation,
    observation_dim,
)
from Baselines.scenario import Scenario, build_scenario
from RL.corridor import DEFAULT_LANE_KF, DEFAULT_RUN_ID, boxes_overlap

try:
    import torch
    import torch.nn as nn
except ImportError as exc:  # pragma: no cover
    raise SystemExit("PyTorch is required for training. Install with: pip install torch") from exc


from Baselines.training import PPOMemory, collect_episode, PolicySelection, add_validation_args
from RL.train_ppo import compute_gae, compute_gae_by_trajectory


def run_episode(scenario, policy, memory, collision_penalty=0.0, collision_event_penalty=0.0):
    return collect_episode(
        scenario, policy, memory, collision_penalty, discrete=True,
        collision_event_penalty=collision_event_penalty,
    )


def ppo_update(
    policy: DirectDiscretePolicy,
    optimizer: "torch.optim.Optimizer",
    memory: PPOMemory,
    args: argparse.Namespace,
) -> dict[str, float]:
    obs = torch.as_tensor(np.array(memory.observations), dtype=torch.float32)
    actions = torch.as_tensor(np.array(memory.actions), dtype=torch.int64)
    masks = torch.as_tensor(np.array(memory.masks), dtype=torch.bool)
    old_log_probs = torch.as_tensor(np.array(memory.log_probs), dtype=torch.float32)
    values_np = np.array(memory.values, dtype=np.float32)
    rewards_np = np.array(memory.rewards, dtype=np.float32)
    dones_np = np.array(memory.dones, dtype=np.float32)

    advantages_np, returns_np = compute_gae_by_trajectory(
        rewards_np, values_np, np.asarray(memory.traj_ids), args.gamma, args.gae_lambda,
        dones=dones_np, timeouts=np.asarray(memory.timeouts),
        bootstrap_values=np.asarray(memory.bootstrap_values),
    )
    advantages = torch.as_tensor(advantages_np, dtype=torch.float32)
    returns = torch.as_tensor(returns_np, dtype=torch.float32)
    advantages = (advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1e-8)

    policy.update_value_stats(returns)
    targets = policy.normalize_return(returns)
    n = obs.shape[0]
    indices = np.arange(n)
    stats = {"loss": 0.0, "policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0}
    for _ in range(args.ppo_epochs):
        epoch_kl = []
        np.random.shuffle(indices)
        for start in range(0, n, args.minibatch_size):
            batch = indices[start : start + args.minibatch_size]
            dist, value = policy.distribution(obs[batch], masks[batch])
            new_log_probs = dist.log_prob(actions[batch])
            entropy = dist.entropy().mean()

            ratio = torch.exp(new_log_probs - old_log_probs[batch])
            unclipped = ratio * advantages[batch]
            clipped = torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef) * advantages[batch]
            policy_loss = -torch.min(unclipped, clipped).mean()
            value_loss = 0.5 * (targets[batch] - policy.normalize_return(value)).pow(2).mean()
            loss = policy_loss + args.value_coef * value_loss - args.entropy_coef * entropy

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(policy.parameters(), max_norm=0.5)
            optimizer.step()
            epoch_kl.append(float(((ratio - 1.0) - (new_log_probs - old_log_probs[batch])).mean().detach()))

            stats = {
                "loss": float(loss.detach()),
                "policy_loss": float(policy_loss.detach()),
                "value_loss": float(value_loss.detach()),
                "entropy": float(entropy.detach()),
            }
        if getattr(args, "target_kl", 0.02) > 0 and np.mean(epoch_kl) > getattr(args, "target_kl", 0.02):
            break
    return stats


def train(args: argparse.Namespace) -> None:
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    rng = np.random.default_rng(args.seed)

    probe = build_scenario(
        seed=args.seed,
        num_agents=args.num_agents,
        max_steps=args.max_steps,
        run_id=args.run_id,
        lane_kf=args.lane_kf,
                obb_safety_filter=bool(getattr(args, "train_obb_filter", True)),
    )
    num_actions = num_grid_actions(probe.sim_config)
    policy = DirectDiscretePolicy(
        obs_dim=observation_dim(probe),
        num_actions=num_actions,
        hidden_dim=args.hidden_dim,
        highway_length=float(probe.corridor.length),
    )
    optimizer = torch.optim.Adam(policy.parameters(), lr=args.lr)

    selection = PolicySelection(args, policy, "direct_discrete_rl")
    selection.budget.watch(optimizer)
    best_reward = -float("inf")
    for update in range(1, args.updates + 1):
        if selection.budget.exhausted:
            break
        if getattr(args, "anneal_lr", True):
            for group in optimizer.param_groups:
                group["lr"] = args.lr * (1.0 - (update - 1) / max(args.updates, 1))
        memory = PPOMemory()
        episode_stats: list[dict[str, float]] = []
        for scenario in selection.training_scenarios(update):
            episode_stats.append(
                run_episode(
                    scenario, policy, memory,
                    collision_penalty=args.collision_penalty,
                    collision_event_penalty=float(getattr(args, "collision_event_penalty", 0.0)),
                )
            )

            selection.budget.add(episode_stats[-1]["steps"], episode_stats[-1]["agent_steps"])

        stats = ppo_update(policy, optimizer, memory, args)
        selection.budget.updates = update
        selection.consider(update)
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

    selection_meta = selection.finish()
    if args.save is not None:
        args.save.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "protocol_version": 3,
                **selection_meta,
                "state_dict": policy.state_dict(),
                "obs_dim": policy.obs_dim,
                "num_actions": policy.num_actions,
                "hidden_dim": policy.hidden_dim,
                "highway_length": float(policy.highway_length.item()),
                "algo": "ppo",
                "model": "direct_discrete_rl",
                "run_id": args.run_id,
                "lane_kf": args.lane_kf,
                "collision_penalty": float(args.collision_penalty),
            },
            args.save,
        )
        print(f"Saved direct discrete RL policy to {args.save}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train the matched direct discrete RL baseline (same grid + OBB filter as residual)"
    )
    parser.add_argument("--updates", type=int, default=100)
    parser.add_argument("--episodes-per-update", type=int, default=4)
    parser.add_argument("--num-agents", type=int, default=10)
    parser.add_argument("--max-steps", type=int, default=240)
    parser.add_argument("--run-id", type=int, default=DEFAULT_RUN_ID)
    parser.add_argument("--lane-kf", type=int, default=DEFAULT_LANE_KF)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=0.95)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--clip-coef", type=float, default=0.2)
    parser.add_argument("--value-coef", type=float, default=0.5)
    parser.add_argument("--entropy-coef", type=float, default=0.0)
    parser.add_argument("--ppo-epochs", type=int, default=4)
    parser.add_argument("--minibatch-size", type=int, default=512)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--collision-penalty", type=float, default=0.0)
    parser.add_argument("--collision-event-penalty", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--log-every", type=int, default=5)
    parser.add_argument(
        "--save",
        type=Path,
        default=Path("Baselines/checkpoints/revision5/direct_discrete_policy.pt"),
    )
    parser.add_argument("--target-kl", type=float, default=0.02)
    parser.add_argument("--no-anneal-lr", dest="anneal_lr", action="store_false")
    add_validation_args(parser)
    parser.add_argument("--train-obb-filter", action="store_true", default=True)
    parser.add_argument("--no-train-obb-filter", dest="train_obb_filter", action="store_false")
    train(parser.parse_args())


if __name__ == "__main__":
    main()
