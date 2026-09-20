# Shared highway comparison protocol

This revision changes controller information and validation selection. Old tables
are historical results and must be regenerated; code tests do not establish a
performance improvement or convergence. The calibrated utility formula and
calibration coefficients are unchanged.

All decision helpers use up to six nearest active neighbors within 60 m by
default. Utility/residual candidate masks and direct-discrete masks use this
same visible set. DWA, MPPI and Frenet retain their native prediction horizons
and costs, with constant-velocity OBB checks and common first-command feasibility.
MPPI rechecks its weighted average and falls back to a feasible sampled trajectory
when needed. All controllers pass through the global execution shield. Reports
include proposed-versus-executed intervention rate (including command clipping,
boundary and vehicle filtering); it is not a count of predicted collisions.

All online learners select on the same validation seeds, hard non-regression in
collision events, collision pair-steps and off-road rate against utility, then
maximum PDMS. A baseline with no safety-feasible checkpoint retains its own
least-violating checkpoint and reports `safety_failed`. Residual's zero-residual
fallback is labeled `utility_fallback`, not learned improvement. Held-out test
scores never select weights.

Use `--total-env-steps` to set a common interaction cap and set `--updates` high
enough to reach it. Both are stopping limits. `--val-every-env-steps` schedules
validation at update boundaries crossing the interval and at the final update.
Summaries record environment steps, active-agent transitions, actual optimizer
steps, outer policy updates, wall time, validation conditions, source hashes and
stopping reason. HATRPO actor line searches are not optimizer `.step()` calls;
do not compare optimizer counts as equivalent compute across algorithms.

Example future experiment configuration (not launched by these code changes):

```powershell
python -m RL.train_ppo --updates 10000 --total-env-steps 100000 --val-every-env-steps 10000 --seed 0 --save RL/checkpoints/fair/residual_policy_seed0.pt
python -m Baselines.train_marl --algo mappo --updates 10000 --total-env-steps 100000 --val-every-env-steps 10000 --seed 0 --save Baselines/checkpoints/fair/mappo_policy_seed0.pt
python -m Baselines.train_direct_discrete_rl --updates 10000 --total-env-steps 100000 --val-every-env-steps 10000 --seed 0 --save Baselines/checkpoints/fair/direct_discrete_policy_seed0.pt
python -m Baselines.benchmark --models utility_pt residual_marl mappo direct_discrete_rl --train-seeds 0 --seed 810000 --scenarios 16 --residual-checkpoint RL/checkpoints/fair/residual_policy.pt --checkpoint-dir Baselines/checkpoints/fair --require-matched-protocol --output-dir Baselines/results/fair
```

Keep agent count, horizon, density, calibration and validation seeds identical.
Use several training seeds for the eventual paper. Report learning curves against
actual interactions at increasing budgets; a fixed 100,000-step example is not
proof of convergence. Separately report compute and tuning effort.

For lookahead ablation use benchmark `--conflict-lookahead endpoint`; it shortens
the common conflict filter to one simulation step without removing native planner
horizons. `--no-obb-safety-filter` disables common vehicle filtering; the boundary
filter stays on, and native planner obstacle avoidance remains. These are
inference ablations, not retrained policies. Keep residual gate ablations separately
labeled (`residual_no_gate`, `residual_random_gate`).

CtRL-Sim/CTG++ save every validation candidate; run their `select` command with
the same synthetic validation settings before a matched closed-loop comparison.
Offline examples, optimizer steps and dataset exposure must be reported separately
from online environment interactions. Strict benchmark mode rejects legacy
selection/version metadata and mismatched validation configurations. It does not
certify an arbitrary geometry adapter: see `TGSIM Case/AUDIT.md` for site-specific
issues that invalidate that case's current scores.

Verification: `python -m unittest discover -s tests -p test_fair_comparison.py`
includes real tiny CLI training for residual, direct-discrete, continuous PPO,
MAPPO, HAPPO and HATRPO, each capped at five environment steps.
