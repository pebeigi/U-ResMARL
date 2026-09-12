# Residual RL method notes (ICLR revision)

## Framing

This project is **closed-loop multi-agent trajectory simulation** on a weak-lane
freeway corridor, not map-based multimodal forecasting.

- **Prior:** calibrated discrete utility controller
  \(a^* = \arg\max_{a \in \mathcal{G}} U(a;\Theta_{\mathrm{base}})\) over a fixed
  bicycle control grid (not prospect theory).
- **Default residual:** candidate-logit residual
  \(a^* = \arg\max_{a} [U(a;\Theta_{\mathrm{base}}) + \Delta(o)_a]\).
- **Ablation residual:** parameter residual
  \(\Theta = \mathrm{gauge}(\Theta_{\mathrm{base}}, \Delta\Theta(o))\).
- **Training:** shared-policy decentralized PPO (IPPO-style). Hard OBB filter is
  **off** during training so collisions remain a learning signal; evaluation and
  the benchmark turn the filter **on** (same path as all planners via
  `sanitize_control_command`).
- **Observation:** Frenet ego features including remaining station \(s^*-s\), plus
  body-frame neighbors.
- **Success gate:** held-out test leftover distance / arrival vs prior, plus a
  non-trivial control-flip rate and a learning curve that improves over updates.

## Scenario counts

Paper tables and ablation/stress suites use **30 scenarios × 3 training seeds**
unless otherwise noted.

## Ablation naming

| Name | Meaning |
| --- | --- |
| `residual_weights_only` | Freeze `sigma_long`, `sigma_lat` (Δσ off) on a **param_delta** policy |
| `residual_sigma_only` | Only σ residuals may adapt on a **param_delta** policy |
| `residual_marl` | Full candidate-logit residual (default) |

Kernel ablations that freeze Θ coordinates only apply to `param_delta`
checkpoints; they are no-ops for candidate-logit residuals.

## Protocol v2 implementation

`RL/transition.py` is the shared synchronous simulator step for residual RL,
learned baselines, and benchmark controllers. Reward is evaluated before motion;
arrival and collision adjustments follow the shared transition. Individual
arrivals terminate value targets; horizon truncations bootstrap the final value.

Use `Baselines.paper_rerun` for the corrected three-seed training/evaluation
workflow. Old checkpoints/results are historical; v2 paths and checkpoint
metadata prevent accidental reuse. The full parameter policy is `residual_param`;
weights-only and sigma-only are inference masks of that separately trained
parameter checkpoint. The nominal-prior residual also has its own training run.
These masks are sensitivity tests, not retrained constrained-policy ablations.
Candidate-logit checkpoints cannot be used for parameter masks.

The shared OBB filter is a heuristic, not a proof of collision avoidance. If both
the requested control and the braking candidate are infeasible, the fallback
brakes rather than coasts. Report empirical safety and residual usage after full
retraining; passing implementation tests does not establish a performance gain.

## Road containment (protocol v3)

A 0.10 m road-polygon margin now constrains the full vehicle footprint, its swept
interpolation between discrete poses, and a straight braking backup. This filter
is active in both training and evaluation, independently of car-to-car filtering.
Utility and direct-RL masks use the same predicate. Continuous commands are
filtered at execution; the shared step validates all agents before committing.
If no feasible command exists, the step raises an explicit error without moving
any agent. Off-road metrics include body overhang. This changes the experiment
contract: v2 artifacts are historical and new runs use v3 directories/checkpoints.

The brake uses an acceleration available on the discrete grid, ensuring the
reserved backup can be selected on the next step. Closed-loop utility candidate
prediction now uses the same speed cap as execution; historical calibration
candidate generation retains its previous per-agent speed limit.
