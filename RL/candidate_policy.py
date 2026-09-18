"""The frozen utility grid as context for a categorical residual policy.

This adapter calls the existing utility functions without changing their scores
or the candidate eligibility rules. Only the training sampling distribution
changes: PPO can now score a selected grid index instead of 63 Gaussian noises.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from RL.boundary import BoundaryInfeasibleError, candidate_boundary_safe
from utility_model import (generate_candidate_actions, build_step_context,
                           evaluate_candidate_utility, candidate_obb_conflict)


@dataclass(frozen=True)
class CandidateIndex:
    index: int


@dataclass
class CandidateContext:
    candidates: list
    utilities: np.ndarray
    mask: np.ndarray
    prior_index: int


def candidate_context(index, agents, params, sim) -> CandidateContext:
    agent = agents[index]
    candidates = generate_candidate_actions(agent, sim["dt"], sim, dedupe=False)
    context = build_step_context(index, agent, agents, sim)
    utilities = np.zeros(len(candidates), dtype=np.float64)
    contained = np.zeros(len(candidates), dtype=bool)
    free = np.zeros(len(candidates), dtype=bool)
    for j, candidate in enumerate(candidates):
        if not candidate_boundary_safe(agent, candidate, sim):
            continue
        contained[j] = True
        utilities[j] = evaluate_candidate_utility(index, agent, candidate, agents,
                                                  params, sim, context=context)
        free[j] = (not sim.get("obb_safety_filter", True)
                   or not candidate_obb_conflict(candidate, index, agents, sim, context=context))
    # Match utility selection's documented fallback when all CV predictions
    # conflict. The shared execution filter still sanitizes the chosen command.
    mask = free if free.any() else contained
    if not mask.any():
        raise BoundaryInfeasibleError("No boundary-feasible utility candidate")
    if not np.isfinite(utilities[mask]).all():
        raise ValueError("Non-finite utility score")
    prior = int(np.argmax(np.where(mask, utilities, -np.inf)))
    # Center before converting to float32 so common utility offsets do not
    # swamp the small score differences that drive the categorical policy.
    utilities -= utilities[prior]
    utilities[~mask] = 0.
    return CandidateContext(candidates, utilities.astype(np.float32), mask, prior)
