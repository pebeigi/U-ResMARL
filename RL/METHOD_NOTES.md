# Residual RL: current method and workflow

Use one primary trainer: `python -m RL.train_ppo`.
The calibrated utility formula and calibration remain frozen. PPO learns a
bounded adjustment to candidate scores, then the shared filters and simulator
execute the selected control.

## What the files are for

- `train_ppo.py`: policy, rollout collection, PPO, validation and checkpointing.
- `traffic_env.py`, `transition.py`, `obs.py`: environment, shared synchronous
  dynamics/reward, and observations. Baselines reuse these modules.
- `candidate_policy.py`, `boundary.py`, `spawn_safety.py`: candidate masks, road
  containment, and valid initial states.
- `corridor.py`, `calibration_io.py`, `param_gauge.py`, `protocol.py`,
  `value_normalization.py`: geometry, frozen prior loading, compatibility, and
  value normalization. These are supporting modules, not separate experiments.
- `demo.py` and `visualize.py`: optional inspection of the specified PPO checkpoint.

Use module commands from the repository root. Visualization uses the checkpoint
explicitly specified by `--checkpoint`.

## Controller and learning

The frozen prior scores the 63 acceleration/steering candidates with calibrated
utility. A shared two-layer, 128-unit actor produces a bounded score residual:

`score(a) = utility(a) + 0.5 * tanh(actor(observation)[a])`

The zero-initialized output layer reproduces utility argmax at deterministic
inference. Training samples a masked categorical distribution with temperature
0.005; inference selects the largest score. Both use the same road and car-to-car
filters. Road containment includes the vehicle footprint, swept motion and a
braking backup. The car-to-car filter remains a heuristic, not a safety proof.

PPO stores each decision's utilities, mask, sampled index and old probability.
Per-agent GAE ends at arrival and bootstraps horizon truncations. Value
normalization preserves unnormalized critic predictions. Categorical KL is
computed from finite log probabilities in float64 and checked before another
optimizer step. `param_delta` is an explicit ablation, not another default method.

## Why the reward changed

On development seed 910100, the previous learned actor achieved 7/16 arrivals,
versus utility's 14/16, yet earned more reward: 31.80 versus 30.04 per car.
Neither collided. Reduced proximity and effort costs outweighed lost completion.
A later diagnostic with additive duration+event contact costs improved events
and arrival at update 9, then lost the checkpoint because jerk/TTC were slightly
worse than utility. Additive competing terms and comfort vetoes were the problem.

Shared driving reward revision 6 follows CaRL. Forward station progress is the
only dense positive term and is multiplied by TTC/proximity and in-band comfort
factors in (0, 1]. Hard contact is a terminal reward: the onset cost is charged
once, then that agent collects no further driving return (physics still continue
so evaluation metrics see the full scene). Comfort is a band, not a request to
undercut the prior's RMS jerk. Arrival still has a one-shot bonus of 8; leftover
remaining-distance stays a small potential. Default contact duration is 0 and the
terminal onset is 1, matching CaRL's `RC × Π p − T`.

Default training discount remains `gamma=0.95` (`dt=0.5`). Utility itself does
not use this reward. Coordinated collision *filtering* for simultaneous turns
remains a separate simulator change.

## Evaluation and selection

Training validation and `Baselines.benchmark` share one rollout recorder and one
metric implementation. They report collisions, off-road events, arrival, remaining
distance/progress, travel time, speed, proximity/TTC, acceleration, jerk, steering,
clearance, and runtime. Arrival transitions are included in control metrics.

`mean_travel_time_s` conditions on the cars that arrived. Also report
`mean_capped_travel_time_s`: unfinished cars contribute the observed horizon.
The latter avoids making a controller look faster merely because fewer cars finish.

A selected learned checkpoint must not regress vs utility on collision events,
collision pair-steps or off-road rate. Among those admissible actors, selection
ranks the NAVSIM-style closed-loop score

`NC × DAC × (5·progress + 5·TTC + 2·comfort) / 12`

Comfort is in-band / out-of-band (accel 2.4 m/s², jerk 8.37 m/s³, |steer| 0.10),
not a request to beat utility's mean jerk. The leftover-distance metric remains
a reported tie-break, not a veto. Deterministic evaluation also applies a
PDM-Closed gate: a residual grid index is executed only if its one-step closed-
loop proposal score strictly beats the frozen utility proposal. Training samples
are not rewritten, so PPO stays on-policy.

This gate describes the sampled validation set, not a population guarantee. Zero
collisions can be matched, not strictly improved. If learning fails the gate,
`selection_status` is `utility_fallback`, with `selected_update: 0`. Never label
that fallback as a learned improvement. The latest trained actor remains in the
resume file, and the full validation history includes rejected actors' metrics.

Utility is evaluated once on the same validation seeds as RL; the redundant
initial baseline-only evaluation was removed. Training, validation and test
seeds must be disjoint. Development uses
`--skip-test`; do not tune against the final held-out test block. A two- or
four-update smoke test checks mechanics, not convergence or superiority.

## Commands and outputs

Current defaults write to `revision5`, keeping previous reward experiments apart.
Simulator/spawn protocol 3, driving reward 6 and training revision 7 identify
different parts of the experiment contract.

A bounded development check:

```powershell
$env:OMP_NUM_THREADS='1'
$env:MKL_NUM_THREADS='1'
$env:OPENBLAS_NUM_THREADS='1'
python -u -m RL.train_ppo --updates 4 --episodes-per-update 2 --num-agents 16 --dense-spawn --max-steps 120 --val-every 2 --val-episodes 2 --validation-seed-start 910100 --skip-test --seed 7 --save RL/checkpoints/revision5/development.pt
```

A short residual-vs-utility diagnostic after the reward/discount fix:

```powershell
python -u -m RL.train_ppo --updates 12 --episodes-per-update 4 --num-agents 16 --dense-spawn --max-steps 120 --val-every 3 --val-episodes 8 --validation-seed-start 910100 --skip-test --seed 7 --log-every 1 --save RL/checkpoints/revision5/diag_reward5/residual_policy.pt
```

Success signal: any validation update with empty `regressions_vs_utility` and
`selection_status=learned_residual`.

Each run writes the selected `.pt`, complete training state `.resume.pt`, a
readable `.summary.json` with per-scenario metrics and rejected validation
results, and learning-curve CSV/PNG files. No separate smoke-summary script is
needed. Resume with only the saved run's path:

```powershell
python -u -m RL.train_ppo --resume RL/checkpoints/revision5/development.resume.pt
python -m unittest discover -s tests -q
```

Resume rejects changed RL code, shared evaluation code or calibration. It restores
optimizer and random states and retries pending validation without repeating a
completed training update. Existing older checkpoints can still be inspected by
explicit path, but are not the defaults for new experiments.

Use `Baselines.benchmark` for matched controller comparisons and
`Baselines.paper_rerun` only after the development recipe shows a learned gain.
Multi-seed, held-out evaluation is required before claiming superiority.

## What the five-hour development run established

The 36-update run `development_5h_20260915_194407` completed 144 training
episodes. On eight matched validation scenarios its latest actor had 8 collision
events and 107/128 arrivals, versus utility's 2 events and 110/128 arrivals.
No evaluated learned checkpoint passed selection; the export is a labeled
utility fallback. The latest trained actor is preserved in `.resume.pt`.

Replays of that actor isolated an objective mismatch on two development cases:

- Seed 910101: RL had 2 events versus 0, but discounted return was -26.7167
  versus -26.9286 (higher is better). Undiscounted return was worse: -41.5628
  versus -41.1156.
- Seed 910107: RL had 4 events versus 2 and fewer arrivals, but discounted return
  was -72.4317 versus -72.5022. Undiscounted return was again worse: -143.5951
  versus -143.3267.

These are counterexamples to the assumption that maximizing the current reward
must improve safety-first metrics. They do not establish a global cause for all
validation failures. With gamma=0.99 and dt=0.5, a penalty 55 seconds into an
episode receives about one third of its undiscounted weight. The reward charges
contact duration; selection also prioritizes the number of distinct events.

Contact traces also show the common constant-velocity filter missing simultaneous
turns, and cases in which maximum braking is already too late. At the observed
contact-onset states in these replays, the executed commands matched what utility
would choose in those same states. This does not absolve earlier residual actions:
they changed the traffic configuration. It means changing only the last command
or assigning all blame to its residual is an incomplete diagnosis.

The categorical distribution's mode and deterministic residual execution now
have an interacting-trajectory regression test. Primary training and validation
also report both total and discounted reward per initial agent, alongside the
unchanged benchmark metrics. Previously only the benchmark score was logged,
so reward improvement could not be distinguished from metric improvement.

Replay diagnostics using the original saved settings and the actual latest actor:

```powershell
python -m tools.diagnose_collisions --model residual --latest --seeds 910101 910107 --checkpoint RL/checkpoints/revision5/development_5h_20260915_194407/residual_policy.pt --output RL/logs/revision5/development_5h_20260915_194407/diagnosis
python -m tools.diagnose_collisions --model utility --seeds 910101 910107 --checkpoint RL/checkpoints/revision5/development_5h_20260915_194407/residual_policy.pt --output RL/logs/revision5/development_5h_20260915_194407/diagnosis
```

Next experiments should verify that learning responds on a short diagnostic with
reward revision 6 / training revision 7 (CaRL terminal contact, PDMS selection,
PDM residual-accept at eval) before another long run. Keep the frozen utility,
action bounds, shared filters and evaluation physics fixed during reward
experiments. Treat coordinated collision filtering as a separate simulator
change that would apply equally to every controller.
# Recorded-data evaluation (2026-09-16)

The --data-evaluation mode in Baselines.benchmark evaluates fixed recorded
scenes with ADE/FDE, speed RMSE, wrapped heading MAE, and matched behavioral
W1/JS for speed, acceleration, lateral offset, yaw rate, gaps, and TTC.
See Baselines/README.md for the protocol, CLI, and output definitions.

This is an evaluation addition: utility, reward, residual limits, safety filters,
and training checkpoint selection are unchanged. The default source has a
retrospective chronological split, but independence from historical calibration
cannot be certified: one-step calibration samples were selected before the
closed-loop split. Manifests and raw rows explicitly mark that exposure as
unverified. Do not present these diagnostics as a newly independent test set.
