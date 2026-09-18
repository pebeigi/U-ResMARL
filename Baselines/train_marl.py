"""Train the cooperative MARL baselines: MAPPO, HAPPO, HATRPO (and IPPO).

Same scenarios and dynamics as the residual model. MAPPO/IPPO retain individual
driving rewards; HAPPO/HATRPO optimize their shared team mean.

    python -m Baselines.train_marl --algo mappo  --updates 200
    python -m Baselines.train_marl --algo happo  --updates 200
    python -m Baselines.train_marl --algo hatrpo --updates 200
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
from Baselines.marl import (
    ALGORITHMS,
    SEQUENTIAL_ALGORITHMS,
    Actor,
    MARLPolicy,
    agent_features,
    centralised_state,
    save_marl_policy,
)
from Baselines.scenario import Scenario, build_scenario
from RL.transition import advance_agents
from Baselines.training import PolicySelection, add_validation_args
from RL.train_ppo import compute_gae
from RL.corridor import DEFAULT_LANE_KF, DEFAULT_RUN_ID, boxes_overlap

try:
    import torch
    import torch.nn as nn
except ImportError as exc:  # pragma: no cover
    raise SystemExit("PyTorch is required for training. Install with: pip install torch") from exc


@dataclass
class Episode:
    """Per-episode tensors laid out as (timesteps, agents, ...)."""

    obs: np.ndarray
    states: np.ndarray
    actions: np.ndarray
    log_probs: np.ndarray
    values: np.ndarray
    rewards: np.ndarray
    masks: np.ndarray
    stats: dict = field(default_factory=dict)
    bootstrap_values: np.ndarray | None = None
    truncated: np.ndarray | None = None  # Per-agent horizon ends, excluding arrivals.
    cooperative: bool = False


def collect_episode(
    scenario: Scenario,
    policy: MARLPolicy,
    collision_penalty: float = 0.0,
    collision_event_penalty: float = 0.0,
) -> Episode:
    agents = scenario.spawn_agents()
    dest_s = np.array([a.dest_s for a in scenario.agents], dtype=float)
    tol = float(scenario.sim_config.get("destination_threshold", 1.0))
    length = scenario.vehicle_length
    width = scenario.vehicle_width
    n = len(agents)
    obs_dim = observation_dim(scenario)
    state_dim = policy.state_dim

    obs_buf, state_buf, action_buf = [], [], []
    logp_buf, value_buf, reward_buf, mask_buf = [], [], [], []
    collisions = 0
    steps = 0
    previous_pairs: set[tuple[int, int]] = set()
    collided: set[int] = set()

    for step in range(scenario.max_steps):
        if all(a.reached_destination or i in collided for i, a in enumerate(agents)):
            break

        features = agent_features(agents, scenario)
        step_obs = np.zeros((n, obs_dim), dtype=np.float32)
        step_state = np.zeros((n, state_dim), dtype=np.float32)
        step_action = np.zeros((n, 2), dtype=np.float32)
        step_logp = np.zeros(n, dtype=np.float32)
        step_value = np.zeros(n, dtype=np.float32)
        step_reward = np.zeros(n, dtype=np.float32)
        step_mask = np.zeros(n, dtype=np.float32)
        controls: dict[int, tuple[float, float]] = {}

        for i, agent in enumerate(agents):
            if i in collided and policy.algo not in SEQUENTIAL_ALGORITHMS:
                continue
            if agent.reached_destination and policy.algo not in SEQUENTIAL_ALGORITHMS:
                continue
            obs = observation(agents, i, scenario)
            state = (
                centralised_state(features, obs, policy.num_agents)
                if policy.centralised
                else obs
            )
            action, log_prob, value = policy.sample_action(obs, state, i)
            step_obs[i] = obs
            step_state[i] = state
            step_action[i] = action
            step_logp[i] = log_prob
            step_value[i] = value
            step_mask[i] = float(not agent.reached_destination and i not in collided)
            if not agent.reached_destination and i not in collided:
                controls[i] = policy.to_control(action)

        transition = advance_agents(
            agents, [controls.get(i, (0.0, 0.0)) for i in range(n)],
            scenario.corridor, scenario.sim_config, dest_s,
            leftover_coef=scenario.sim_config.get("leftover_coef", 0.08),
            arrival_bonus=scenario.sim_config.get("arrival_bonus", 8.0),
            collision_penalty=collision_penalty,
            collision_event_penalty=collision_event_penalty,
            previous_collision_pairs=previous_pairs,
        )
        previous_pairs = set(transition.collision_pairs)
        collided |= set(transition.colliding_agents)
        collisions += len(transition.collision_pairs)
        step_reward[:] = transition.rewards
        if policy.algo in SEQUENTIAL_ALGORITHMS:
            step_reward[:] = float(np.mean(transition.rewards))

        obs_buf.append(step_obs)
        state_buf.append(step_state)
        action_buf.append(step_action)
        logp_buf.append(step_logp)
        value_buf.append(step_value)
        reward_buf.append(step_reward)
        mask_buf.append(step_mask)
        steps = step + 1

    bootstrap = np.zeros(n, dtype=np.float32)
    truncated = np.zeros(n, dtype=bool)
    if steps >= scenario.max_steps:
        features = agent_features(agents, scenario)
        all_arrived = all(a.reached_destination for a in agents)
        for i, agent in enumerate(agents):
            if policy.algo in SEQUENTIAL_ALGORITHMS:
                # Team return continues until every agent finishes.
                need_boot = not all_arrived
            else:
                # Individual returns: only still-active agents are truncated.
                need_boot = (not agent.reached_destination) and (i not in collided)
            if not need_boot:
                continue
            truncated[i] = True
            obs = observation(agents, i, scenario)
            state = centralised_state(features, obs, policy.num_agents) if policy.centralised else obs
            with torch.no_grad():
                bootstrap[i] = float(policy.value(torch.as_tensor(state, dtype=torch.float32)))
    arrived = sum(a.reached_destination for a in agents)
    return Episode(
        obs=np.asarray(obs_buf),
        states=np.asarray(state_buf),
        actions=np.asarray(action_buf),
        log_probs=np.asarray(logp_buf),
        values=np.asarray(value_buf),
        rewards=np.asarray(reward_buf),
        masks=np.asarray(mask_buf),
        bootstrap_values=bootstrap,
        truncated=truncated,
        cooperative=policy.algo in SEQUENTIAL_ALGORITHMS,
        stats={
            "reward": float(np.sum(reward_buf) / max(n, 1)),
            "collisions": float(collisions),
            "arrival_rate": arrived / max(n, 1),
            "steps": float(steps),
        },
    )


def episode_gae(episode: Episode, gamma: float, gae_lambda: float) -> tuple[np.ndarray, np.ndarray]:
    """GAE per agent along the time axis, ignoring steps where the agent is done."""
    advantages = np.zeros_like(episode.rewards)
    returns = np.zeros_like(episode.rewards)
    for i in range(episode.rewards.shape[1]):
        # Team returns continue after an individual arrival: its earlier action
        # can affect the remaining agents. Actor updates still mask inactive rows.
        idx = (np.arange(len(episode.rewards)) if episode.cooperative
               else np.flatnonzero(episode.masks[:, i]))
        if not len(idx):
            continue
        dones = np.zeros(len(idx), dtype=np.float32)
        dones[-1] = 1.0
        timeouts = np.zeros_like(dones)
        boot = np.zeros_like(dones)
        if episode.truncated is not None and episode.truncated[i]:
            # Bootstrap only when the agent's last active index is the episode end.
            if (
                episode.bootstrap_values is not None
                and idx[-1] == len(episode.rewards) - 1
            ):
                timeouts[-1] = 1.0
                boot[-1] = episode.bootstrap_values[i]
            # Otherwise treat as a true terminal (already dones[-1]=1).
        advantages[idx, i], returns[idx, i] = compute_gae(
            episode.rewards[idx, i], episode.values[idx, i], dones, gamma, gae_lambda,
            timeouts=timeouts, bootstrap_values=boot)
    return advantages, returns


@dataclass
class Batch:
    obs: torch.Tensor  # (T, n, obs_dim)
    states: torch.Tensor
    actions: torch.Tensor
    log_probs: torch.Tensor  # (T, n)
    advantages: torch.Tensor
    returns: torch.Tensor
    masks: torch.Tensor
    critic_masks: torch.Tensor | None = None


def build_batch(episodes: list[Episode], gamma: float, gae_lambda: float) -> Batch:
    advantages, returns = [], []
    for episode in episodes:
        adv, ret = episode_gae(episode, gamma, gae_lambda)
        advantages.append(adv)
        returns.append(ret)

    def cat(key: str) -> torch.Tensor:
        return torch.as_tensor(
            np.concatenate([getattr(e, key) for e in episodes], axis=0), dtype=torch.float32
        )

    adv = torch.as_tensor(np.concatenate(advantages, axis=0), dtype=torch.float32)
    masks = cat("masks")
    valid = masks > 0
    if valid.any():
        mean = adv[valid].mean()
        std = adv[valid].std(unbiased=False).clamp_min(1e-6)
        adv = (adv - mean) / std * masks

    return Batch(
        obs=cat("obs"),
        states=cat("states"),
        actions=cat("actions"),
        log_probs=cat("log_probs"),
        advantages=adv,
        returns=torch.as_tensor(np.concatenate(returns, axis=0), dtype=torch.float32),
        masks=masks,
        critic_masks=torch.as_tensor(np.concatenate([
            np.ones_like(e.masks) if e.cooperative else e.masks for e in episodes
        ], axis=0), dtype=torch.float32),
    )


def _masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    total = mask.sum().clamp_min(1.0)
    return (values * mask).sum() / total


def update_critic(
    policy: MARLPolicy,
    optimizer: "torch.optim.Optimizer",
    batch: Batch,
    epochs: int,
    minibatch_size: int,
) -> float:
    states = batch.states.reshape(-1, batch.states.shape[-1])
    returns = batch.returns.reshape(-1)
    masks = (batch.critic_masks if batch.critic_masks is not None else batch.masks).reshape(-1)
    index = np.flatnonzero(masks.numpy() > 0)
    if index.size == 0:
        return 0.0

    loss_value = 0.0
    for _ in range(epochs):
        np.random.shuffle(index)
        for start in range(0, index.size, minibatch_size):
            sel = torch.as_tensor(index[start : start + minibatch_size], dtype=torch.long)
            value = policy.value(states[sel])
            loss = 0.5 * (returns[sel] - value).pow(2).mean()
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(policy.critic.parameters(), max_norm=0.5)
            optimizer.step()
            loss_value = float(loss.detach())
    return loss_value


def update_actors_simultaneous(
    policy: MARLPolicy,
    optimizer: "torch.optim.Optimizer",
    batch: Batch,
    args: argparse.Namespace,
) -> dict[str, float]:
    """IPPO / MAPPO: one shared actor, all agents' samples pooled."""
    obs = batch.obs.reshape(-1, batch.obs.shape[-1])
    actions = batch.actions.reshape(-1, batch.actions.shape[-1])
    old_log_probs = batch.log_probs.reshape(-1)
    advantages = batch.advantages.reshape(-1)
    masks = batch.masks.reshape(-1)
    index = np.flatnonzero(masks.numpy() > 0)

    stats = {"policy_loss": 0.0, "entropy": 0.0}
    for _ in range(args.ppo_epochs):
        np.random.shuffle(index)
        for start in range(0, index.size, args.minibatch_size):
            sel = torch.as_tensor(index[start : start + args.minibatch_size], dtype=torch.long)
            dist = policy.distribution(obs[sel], 0)
            new_log_probs = dist.log_prob(actions[sel]).sum(dim=-1)
            entropy = dist.entropy().sum(dim=-1).mean()

            ratio = torch.exp(new_log_probs - old_log_probs[sel])
            unclipped = ratio * advantages[sel]
            clipped = torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef) * advantages[sel]
            loss = -torch.min(unclipped, clipped).mean() - args.entropy_coef * entropy

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(policy.actors.parameters(), max_norm=0.5)
            optimizer.step()
            stats = {"policy_loss": float(loss.detach()), "entropy": float(entropy.detach())}
    return stats


def _log_prob_for(policy: MARLPolicy, batch: Batch, agent_idx: int) -> torch.Tensor:
    dist = policy.distribution(batch.obs[:, agent_idx], agent_idx)
    return dist.log_prob(batch.actions[:, agent_idx]).sum(dim=-1)


def update_actors_happo(
    policy: MARLPolicy,
    optimizers: list["torch.optim.Optimizer"],
    batch: Batch,
    args: argparse.Namespace,
) -> dict[str, float]:
    """HAPPO: sequential per-agent clipped updates with the multi-agent factor.

    The factor accumulates the probability ratios of the agents already updated
    in this pass. Finite-sample clipped updates do not certify monotonic improvement.
    """
    factor = torch.ones(batch.obs.shape[0], dtype=torch.float32)
    order = np.random.permutation(policy.num_agents)
    stats = {"policy_loss": 0.0, "entropy": 0.0}

    for agent_idx in order:
        agent_idx = int(agent_idx)
        mask = batch.masks[:, agent_idx]
        if float(mask.sum()) < 1.0:
            continue
        old_log_probs = batch.log_probs[:, agent_idx]
        advantages = batch.advantages[:, agent_idx]
        index = np.flatnonzero(mask.numpy() > 0)

        for _ in range(args.ppo_epochs):
            np.random.shuffle(index)
            for start in range(0, index.size, args.minibatch_size):
                sel = torch.as_tensor(index[start : start + args.minibatch_size], dtype=torch.long)
                dist = policy.distribution(batch.obs[sel, agent_idx], agent_idx)
                new_log_probs = dist.log_prob(batch.actions[sel, agent_idx]).sum(dim=-1)
                entropy = dist.entropy().sum(dim=-1).mean()

                ratio = torch.exp(new_log_probs - old_log_probs[sel])
                weighted = factor[sel] * advantages[sel]
                unclipped = ratio * weighted
                clipped = torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef) * weighted
                loss = -torch.min(unclipped, clipped).mean() - args.entropy_coef * entropy

                optimizers[agent_idx].zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(policy.actor_for(agent_idx).parameters(), max_norm=0.5)
                optimizers[agent_idx].step()
                stats = {"policy_loss": float(loss.detach()), "entropy": float(entropy.detach())}

        with torch.no_grad():
            updated = _log_prob_for(policy, batch, agent_idx)
            ratio = torch.exp(updated - old_log_probs)
            factor = factor * torch.where(mask > 0, ratio, torch.ones_like(ratio))
    return stats


def _flat_params(module: nn.Module) -> torch.Tensor:
    return torch.cat([p.data.reshape(-1) for p in module.parameters()])


def _set_flat_params(module: nn.Module, flat: torch.Tensor) -> None:
    offset = 0
    for p in module.parameters():
        count = p.numel()
        p.data.copy_(flat[offset : offset + count].view_as(p))
        offset += count


def _flat_grad(
    loss: torch.Tensor,
    params: list[torch.nn.Parameter],
    retain_graph: bool = False,
    create_graph: bool = False,
) -> torch.Tensor:
    grads = torch.autograd.grad(loss, params, retain_graph=retain_graph, create_graph=create_graph)
    return torch.cat([g.reshape(-1) for g in grads])


def _conjugate_gradient(matvec, b: torch.Tensor, iterations: int = 10, tol: float = 1e-10):
    x = torch.zeros_like(b)
    r = b.clone()
    p = b.clone()
    rr = torch.dot(r, r)
    for _ in range(iterations):
        Ap = matvec(p)
        alpha = rr / (torch.dot(p, Ap) + 1e-12)
        x = x + alpha * p
        r = r - alpha * Ap
        rr_new = torch.dot(r, r)
        if rr_new < tol:
            break
        p = r + (rr_new / rr) * p
        rr = rr_new
    return x


def _gaussian_kl(
    actor: Actor,
    obs: torch.Tensor,
    old_mean: torch.Tensor,
    old_log_std: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    mean = actor(obs)
    log_std = actor.log_std.expand_as(mean)
    kl = (
        log_std
        - old_log_std
        + (torch.exp(2 * old_log_std) + (old_mean - mean) ** 2) / (2 * torch.exp(2 * log_std))
        - 0.5
    )
    return _masked_mean(kl.sum(dim=-1), mask)


def update_actors_hatrpo(
    policy: MARLPolicy,
    batch: Batch,
    args: argparse.Namespace,
) -> dict[str, float]:
    """HATRPO: the same sequential scheme with a KL trust region per agent."""
    factor = torch.ones(batch.obs.shape[0], dtype=torch.float32)
    order = np.random.permutation(policy.num_agents)
    stats = {"policy_loss": 0.0, "kl": 0.0}

    for agent_idx in order:
        agent_idx = int(agent_idx)
        mask = batch.masks[:, agent_idx]
        if float(mask.sum()) < 1.0:
            continue

        actor = policy.actor_for(agent_idx)
        params = list(actor.parameters())
        obs = policy.obs_norm(batch.obs[:, agent_idx]).detach()
        actions = batch.actions[:, agent_idx]
        old_log_probs = batch.log_probs[:, agent_idx]
        advantages = batch.advantages[:, agent_idx]
        weighted = factor * advantages

        with torch.no_grad():
            old_mean = actor(obs).detach()
            old_log_std = actor.log_std.detach().clone().expand_as(old_mean)

        def surrogate() -> torch.Tensor:
            dist = torch.distributions.Normal(actor(obs), torch.exp(actor.log_std).expand_as(old_mean))
            log_probs = dist.log_prob(actions).sum(dim=-1)
            ratio = torch.exp(log_probs - old_log_probs)
            return _masked_mean(ratio * weighted, mask)

        loss = surrogate()
        gradient = _flat_grad(loss, params, retain_graph=True)
        if float(torch.norm(gradient)) < 1e-8:
            continue

        def fisher_vector_product(vector: torch.Tensor) -> torch.Tensor:
            kl = _gaussian_kl(actor, obs, old_mean, old_log_std, mask)
            kl_grad = _flat_grad(kl, params, retain_graph=True, create_graph=True)
            product = _flat_grad((kl_grad * vector).sum(), params, retain_graph=True)
            return product + args.cg_damping * vector

        step_direction = _conjugate_gradient(
            fisher_vector_product, gradient, iterations=args.cg_iterations
        )
        shs = 0.5 * torch.dot(step_direction, fisher_vector_product(step_direction))
        if float(shs) <= 0:
            continue
        step_size = torch.sqrt(args.max_kl / (shs + 1e-12))
        full_step = step_size * step_direction
        expected_improvement = float(torch.dot(gradient, full_step))

        old_params = _flat_params(actor)
        old_loss = float(loss.detach())
        kl_limit = float(args.max_kl)
        accepted = False
        for fraction in [args.line_search_decay**k for k in range(args.line_search_steps)]:
            _set_flat_params(actor, old_params + fraction * full_step)
            with torch.no_grad():
                new_loss = float(surrogate())
                kl = float(_gaussian_kl(actor, obs, old_mean, old_log_std, mask))
            improvement = new_loss - old_loss
            expected = expected_improvement * fraction
            improvement_ratio = improvement / expected if abs(expected) > 1e-12 else 0.0
            if kl <= kl_limit and improvement > 0 and improvement_ratio > args.accept_ratio:
                accepted = True
                stats = {"policy_loss": -new_loss, "kl": kl}
                break
        if not accepted:
            _set_flat_params(actor, old_params)

        with torch.no_grad():
            updated = _log_prob_for(policy, batch, agent_idx)
            ratio = torch.exp(updated - old_log_probs)
            factor = factor * torch.where(mask > 0, ratio, torch.ones_like(ratio))
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
    policy = MARLPolicy(
        obs_dim=observation_dim(probe),
        num_agents=args.num_agents,
        algo=args.algo,
        hidden_dim=args.hidden_dim,
        max_accel=float(probe.sim_config.get("max_accel", 4.0)),
        init_log_std=args.init_log_std,
    )
    critic_optimizer = torch.optim.Adam(policy.critic.parameters(), lr=args.critic_lr)
    if args.algo in SEQUENTIAL_ALGORITHMS:
        actor_optimizers = [torch.optim.Adam(a.parameters(), lr=args.lr) for a in policy.actors]
        shared_optimizer = None
    else:
        actor_optimizers = []
        shared_optimizer = torch.optim.Adam(policy.actors.parameters(), lr=args.lr)

    print(
        f"algo={args.algo} | actors={len(policy.actors)} | obs_dim={policy.obs_dim} | "
        f"state_dim={policy.state_dim} | agents={args.num_agents}"
    )

    selection = PolicySelection(args, policy, args.algo)
    best_reward = -float("inf")
    best_collisions = float("inf")
    for update in range(1, args.updates + 1):
        episodes = [
            collect_episode(
                build_scenario(
                    seed=int(rng.integers(10_000_000, 2_000_000_000)),
                    num_agents=args.num_agents,
                    max_steps=args.max_steps,
                    run_id=args.run_id,
                    lane_kf=args.lane_kf,
                    obb_safety_filter=bool(getattr(args, "train_obb_filter", True)),
                ),
                policy,
                collision_penalty=args.collision_penalty,
                collision_event_penalty=float(getattr(args, "collision_event_penalty", 0.0)),
            )
            for _ in range(args.episodes_per_update)
        ]
        batch = build_batch(episodes, args.gamma, args.gae_lambda)

        if args.algo in SEQUENTIAL_ALGORITHMS:
            if args.algo == "happo":
                stats = update_actors_happo(policy, actor_optimizers, batch, args)
            else:
                stats = update_actors_hatrpo(policy, batch, args)
        else:
            stats = update_actors_simultaneous(policy, shared_optimizer, batch, args)
        critic_loss = update_critic(
            policy, critic_optimizer, batch, args.critic_epochs, args.minibatch_size
        )

        policy.obs_norm.update(batch.obs[batch.masks > 0])
        policy.state_norm.update(batch.states[batch.critic_masks > 0])

        mean_reward = float(np.mean([e.stats["reward"] for e in episodes]))
        mean_collisions = float(np.mean([e.stats["collisions"] for e in episodes]))
        best_reward = max(best_reward, mean_reward)
        best_collisions = min(best_collisions, mean_collisions)
        selection.consider(update)
        if update == 1 or update % max(args.log_every, 1) == 0:
            print(
                f"Update {update:4d}/{args.updates} | reward={mean_reward:9.3f} | "
                f"best={best_reward:9.3f} | "
                f"collisions={np.mean([e.stats['collisions'] for e in episodes]):6.2f} | "
                f"arrived={np.mean([e.stats['arrival_rate'] for e in episodes]):5.2f} | "
                f"critic={critic_loss:8.3f} | "
                + " | ".join(f"{k}={v:7.3f}" for k, v in stats.items())
            )

    selection_meta = selection.finish()
    if args.save is not None:
        save_marl_policy(
            policy,
            args.save,
            extra={
                **selection_meta,
                "run_id": args.run_id,
                "lane_kf": args.lane_kf,
                "updates": args.updates,
                "obs": "frenet_body",
                "best_train_collisions": best_collisions,
            },
        )
        print(
            f"Saved {args.algo} policy to {args.save} "
            f"(selected validation update={selection.selected_update})"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Train MAPPO / HAPPO / HATRPO baselines")
    parser.add_argument("--algo", choices=ALGORITHMS, default="mappo")
    parser.add_argument("--updates", type=int, default=200)
    parser.add_argument("--episodes-per-update", type=int, default=4)
    parser.add_argument("--num-agents", type=int, default=10)
    parser.add_argument("--max-steps", type=int, default=240)
    parser.add_argument("--run-id", type=int, default=DEFAULT_RUN_ID)
    parser.add_argument("--lane-kf", type=int, default=DEFAULT_LANE_KF)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--critic-lr", type=float, default=1e-3)
    parser.add_argument("--gamma", type=float, default=0.95)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--clip-coef", type=float, default=0.2)
    parser.add_argument("--entropy-coef", type=float, default=0.01)
    parser.add_argument("--ppo-epochs", type=int, default=4)
    parser.add_argument("--critic-epochs", type=int, default=4)
    parser.add_argument("--minibatch-size", type=int, default=256)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--init-log-std", type=float, default=-1.6)
    parser.add_argument("--collision-penalty", type=float, default=0.0)
    parser.add_argument("--collision-event-penalty", type=float, default=1.0)
    parser.add_argument("--max-kl", type=float, default=0.01, help="HATRPO trust region")
    parser.add_argument("--cg-iterations", type=int, default=10)
    parser.add_argument("--cg-damping", type=float, default=0.1)
    parser.add_argument("--line-search-steps", type=int, default=10)
    parser.add_argument("--line-search-decay", type=float, default=0.5)
    parser.add_argument("--accept-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--log-every", type=int, default=20)
    parser.add_argument("--save", type=Path, default=None)
    add_validation_args(parser)
    parser.add_argument("--train-obb-filter", action="store_true", default=True)
    parser.add_argument("--no-train-obb-filter", dest="train_obb_filter", action="store_false")
    args = parser.parse_args()
    if args.save is None:
        args.save = Path(f"Baselines/checkpoints/revision5/{args.algo}_policy.pt")
    args.max_kl = torch.tensor(float(args.max_kl))
    train(args)


if __name__ == "__main__":
    main()
