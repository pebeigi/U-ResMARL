"""Replay saved training settings and audit reward, contact onset and persistence.

Diagnostic only: wraps the normal transition without changing any controls.
Run from the repository root with `python -m tools.diagnose_collisions`.
"""
import argparse
import copy
import json
from collections import Counter
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from Baselines.residual_marl import load_residual_policy
from RL.boundary import footprint_clearance
from RL.corridor import boxes_overlap
from RL.train_ppo import make_env
from RL.transition import advance_agents
from utility_model import (candidate_obb_conflict, kinematic_bicycle_rollout, build_step_context,
                           select_best_candidate, sanitize_control_command)


def replay(model, seed, checkpoint, output, *, latest=False):
    # The selected export may be a utility fallback. Diagnose latest learned
    # weights explicitly, using the exact horizon/reward/spawn settings saved by
    # the trainer rather than the old diagnostic's hard-coded 240-step horizon.
    saved = torch.load(checkpoint.with_suffix(".resume.pt"), map_location="cpu", weights_only=False)
    args = argparse.Namespace(**saved["args"])
    env = make_env(args, seed, obb_safety_filter=True)
    obs = env.reset()
    policy = load_residual_policy(checkpoint, env.obs_dim) if model == "residual" else None
    if policy is not None and latest:
        policy.load_state_dict(saved["state_dict"])
    sim = env.config.sim_config
    length, width = sim["vehicle_length"], sim["vehicle_width"]
    counts, onsets = Counter(), Counter()
    prev_pairs = set()
    events = []
    reward_components = Counter()
    returns, discounted_returns = np.zeros(len(env.agents)), np.zeros(len(env.agents))
    decisions = flips = 0
    min_clearance = float("inf")
    first_positions = [a.pos.tolist() for a in env.agents]
    initial_pairs = [(i,j) for i in range(len(env.agents)) for j in range(i+1,len(env.agents))
                     if boxes_overlap(env.agents[i].pos, env.agents[i].heading,
                                      env.agents[j].pos, env.agents[j].heading, length, width)]

    def traced(agents, controls, corridor, sim, dest_s, **kwargs):
        nonlocal prev_pairs, min_clearance
        before = copy.deepcopy(agents)
        result = advance_agents(agents, controls, corridor, sim, dest_s, **kwargs)
        step = env.step_count + 1
        weights = env.config.reward_weights
        components = Counter()
        for i, was_active in enumerate(result.active_before):
            if not was_active:
                continue
            old_s, new_s = corridor.project(before[i].pos)[0], corridor.project(agents[i].pos)[0]
            components["progress"] += weights["progress"] * float(np.clip(
                (min(new_s, dest_s[i]) - min(old_s, dest_s[i])) / (sim["max_agent_speed"] * sim["dt"]), -1., 1.))
            components["time"] -= weights.get("time", 0.)
            accel, steer = result.controls[i]
            components["effort"] -= weights["smooth"] * ((accel/sim["max_accel"])**2
                                    + sim["steering_penalty_weight"]*(steer/.45)**2)
            components["remaining"] -= env.config.leftover_coef * max(dest_s[i]-new_s, 0.)/max(corridor.length, 1.)
        components["arrival"] = env.config.arrival_bonus * len(result.arrived)
        components["contact"] = -env.config.collision_penalty * len(result.colliding_agents)
        components["proximity_and_boundary"] = sum(result.rewards) - sum(components.values())
        reward_components.update(components)
        counts.update(result.collision_pairs)
        for i, j in sorted(result.collision_pairs - prev_pairs):
            onsets[(i,j)] += 1
            def overlap(p,h,q,k):
                return bool(boxes_overlap(p,h,q,k,length,width))
            details = []
            prior_moves = {}
            for a in (i, j):
                prior = select_best_candidate(a, before[a], before, env.config.base_params, sim)
                control = sanitize_control_command(a, before[a], before,
                    (prior["accel_longitudinal"], prior["steering_angle"]), sim)
                prior_moves[a] = kinematic_bicycle_rollout(before[a].pos, before[a].heading,
                                                          before[a].speed, *control, sim["dt"], sim)
                prior_moves[a]["control"] = control
            for a,b in [(i,j),(j,i)]:
                old, other = before[a], before[b]
                move = result.candidates[a]
                context = build_step_context(a, old, before, sim)
                brake = kinematic_bicycle_rollout(old.pos,old.heading,old.speed,-sim["max_accel"],0.,sim["dt"],sim)
                details.append(dict(agent=a, before_pos=old.pos.tolist(), before_vel=old.vel.tolist(),
                                    before_heading=old.heading, before_speed=old.speed,
                                    requested=list(controls[a]), executed=list(result.controls[a]),
                                    utility_control_here=list(prior_moves[a]["control"]),
                                    utility_instead_avoids_pair_contact=not overlap(
                                        prior_moves[a]["pos"], prior_moves[a]["heading"],
                                        result.candidates[b]["pos"], result.candidates[b]["heading"]),
                                    after_pos=move["pos"].tolist(), after_heading=move["heading"],
                                    after_speed=move["speed"],
                                    filter_detects_conflict=bool(candidate_obb_conflict(move,a,before,sim)),
                                    cv_endpoint_pair_overlap=overlap(move["pos"],move["heading"],
                                        other.pos+other.vel*sim["dt"],other.heading),
                                    brake_endpoint_vs_actual_other=overlap(brake["pos"],brake["heading"],
                                        result.candidates[b]["pos"],result.candidates[b]["heading"]),
                                    sample_times=[float(x[0]) for x in context.conflict_samples]))
            event=dict(model=model,seed=seed,step=step,time_s=step*sim["dt"],pair=[i,j],details=details,
                       both_utility_controls_avoid_pair_contact=not overlap(
                           prior_moves[i]["pos"], prior_moves[i]["heading"],
                           prior_moves[j]["pos"], prior_moves[j]["heading"]))
            events.append(event)
            print(json.dumps({"event":event}),flush=True)
        prev_pairs = result.collision_pairs
        for a in agents:
            min_clearance = min(min_clearance, footprint_clearance(corridor,a.pos,a.heading,length,width))
        return result

    done = False
    with patch("RL.traffic_env.advance_agents", side_effect=traced):
        while not done:
            actions = None if policy is None else [policy.act(x,0.)[0] for x in obs]
            step = env.step_count
            obs,rewards,done,info = env.step(actions)
            returns += rewards
            discounted_returns += args.gamma**step * np.asarray(rewards)
            decisions += info["control_decisions"]
            flips += info["control_flips"]
            if env.step_count % 60 == 0:
                print(json.dumps(dict(progress=model,seed=seed,step=env.step_count)),flush=True)
    summary = dict(model=model,seed=seed,steps=env.step_count,initial_pairs=initial_pairs,
                   collision_events=env.collision_events,collision_pair_steps=env.collision_count,
                   arrival_rate=float(np.mean([a.reached_destination for a in env.agents])),
                   weights="latest_learned" if latest and policy is not None else
                           ("selected_export" if policy is not None else "utility"),
                   mean_return=float(returns.mean()), discounted_return=float(discounted_returns.mean()),
                   per_agent_return=returns.tolist(),
                   reward_components_per_agent={k:v/len(env.agents) for k,v in reward_components.items()},
                   control_flip_rate=flips/max(decisions,1),
                   metric=float(env.rollout_metric()), min_clearance_m=min_clearance,
                   pair_steps={str(k):v for k,v in counts.items()},
                   pair_events={str(k):v for k,v in onsets.items()})
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(dict(summary=summary,events=events,initial_positions=first_positions),indent=2),encoding="utf-8")
    print(json.dumps({"summary":summary}),flush=True)
    return summary


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model",choices=["utility","residual"],required=True)
    parser.add_argument("--seeds",type=int,nargs="+",required=True)
    parser.add_argument("--checkpoint",type=Path,required=True,
                        help="Selected .pt export with its matching .resume.pt training state")
    parser.add_argument("--latest",action="store_true",help="Replay the latest trained actor, including rejected actors")
    parser.add_argument("--output",type=Path,default=Path("RL/logs/collision_diagnosis"))
    args=parser.parse_args()
    for seed in args.seeds:
        suffix = "latest" if args.latest and args.model == "residual" else "selected"
        replay(args.model,seed,args.checkpoint,args.output/f"{args.model}_{suffix}_{seed}.json",latest=args.latest)


if __name__ == "__main__":
    main()
