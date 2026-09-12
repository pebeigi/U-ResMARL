# Baselines — benchmarking suite

Comparison models for the residual RL paper on **closed-loop multi-agent
trajectory simulation**. This package **imports** the `RL/` package and evaluates
every controller on identical scenarios with identical bicycle dynamics and
metrics.

The method adapts a calibrated discrete utility controller with a learned
residual. The shared conflict filter can reduce collisions; it does not guarantee
collision-free motion. Safety and performance claims require the corrected runs.

## Hard road boundaries (v3)

Road containment is enforced during training and evaluation for every controller,
independently of `--no-obb-safety-filter` (which only disables car-to-car filtering).
The whole oriented vehicle footprint must remain within the road polygon with a
0.10 m margin. Candidate masks reject actions whose conservatively enclosed swept
footprint leaves that region or whose successor lacks a straight braking backup.
Continuous controls use the same filter, and the shared transition validates all
moves before updating any agent. An infeasible state raises `BoundaryInfeasibleError`;
it is not converted to an off-road step, teleported position, or artificial stop.

The swept enclosure covers linear position/heading interpolation between the
simulator's discrete bicycle poses. It is conservative and model-specific, not
a certificate for a real vehicle or collision avoidance between moving vehicles.
Off-road metrics now use the full footprint. Boundary constraints may reduce
throughput; a zero off-road rate alone does not establish a better driving policy.

Protocol v2 checkpoints/results remain historical. Retrain for v3 comparisons.
The new Shapely dependency performs polygon containment and swept-envelope checks.
The design follows [Nav2 footprint checking](https://docs.nav2.org/jazzy/configuration_and_development/configuration_guide/controller_plugins/dwb_controller/trajectory_critics/obstacle_footprint/)
and the [DWA safe-stopping criterion](https://rse-lab.cs.washington.edu/abstracts/colli-ieee.abstract.html).

## Corrected experiment protocol (v3)

Historical checkpoints and results are preserved. They predate fixes to the
simulation and learning targets and must not be used for new comparisons.
New checkpoints go to `RL/checkpoints/v3/` and `Baselines/checkpoints/v3/`;
new results go to `Baselines/results/v3/`. Evaluation rejects incompatible
checkpoints and requires every requested training seed.

Training and evaluation share synchronous movement, pre-transition driving
rewards, arrival bonuses, and collision accounting in `RL/transition.py`.
PPO returns follow each agent separately, terminate on arrival, and bootstrap
at time limits. The paper trainers use collision penalty 8, filter off during
training, filter on for validation and evaluation, and safety-first checkpoint
selection on held-out validation scenarios. Test scores do not select weights.
The legacy continuous baseline defaults to collision penalty 0; set it explicitly
when making a reward-matched comparison.

`residual_param` trains a separate parameter-residual policy. Its weights-only
and sigma-only variants mask that policy at inference: these measure sensitivity
to removing learned residual components, not separately retrained architectures.
`residual_nominal` is trained independently with the nominal prior. Parameter
masks on candidate-logit checkpoints raise an error instead of reporting a no-op.

From the repository root, the corrected paper workflow is:

```bash
python -m Baselines.paper_rerun status
python -m Baselines.paper_rerun train --jobs 1
python -m Baselines.paper_rerun eval
python -m Baselines.paper_figures --all
```

This trains five families across seeds 0, 1, and 2. The short regression/smoke
checks validate implementation only; full retraining and evaluation remain
necessary before interpreting policy performance. Metric bar error bars show
standard deviations; bootstrap intervals are in the statistics CSV and CI table.

## Design

Controllers share scenarios, dynamics, and evaluation metrics. The individual
driving reward is shared by the matched PPO comparison; HAPPO/HATRPO use its
team mean for their cooperative objective. Shared infrastructure:

| Shared component | Where |
| --- | --- |
| Initial conditions (positions, speeds, destinations) | `scenario.py` — generated once per seed from `RL.traffic_env` spawn logic |
| Kinematic bicycle integrator, observations, reward | `dynamics.py` |
| Rollout loop, oriented-box collisions, arrival rule | `runner.py` |
| Shared OBB safety filter (1.5 s / 4-substep lookahead) | `utility_model.sanitize_control_command` via `RL.transition.advance_agents` — applied to **every** controller in `runner.py` at evaluation; off during training by default |
| Safety / efficiency / comfort metrics | `metrics.py` |
| Distributional realism vs. measured data | `realism.py` |

A controller only has to implement:

```python
def compute_controls(self, agents, scenario, step) -> list[tuple[float, float]]:
    ...  # one (accel, steering) per agent
```

## Models

### Matched direct discrete RL (isolates the utility structure)

The controlled comparison is `direct_discrete_rl` vs. `residual_marl`. Both use
the same observations, reward, bicycle candidates, and OBB conflict rejection;
only the action parameterization differs:

| Model | Policy output | Action selection |
| --- | --- | --- |
| `direct_discrete_rl` | logits over the 7×9 `(accel, δ)` grid | masked categorical sample |
| `residual_marl` (default) | additive residual on the same discrete utilities | `argmax_a [U(a; Θ_base) + Δ(a)]` |
| `residual_marl` (`--residual-mode param_delta`) | `ΔΘ` on utility parameters (ablation) | `argmax_a U(a; gauge(Θ_base, ΔΘ))` |

Train (defaults match residual: 240 steps, collision penalty 8):

```bash
python -m Baselines.train_direct_discrete_rl --updates 100 --collision-penalty 8
python -m Baselines.train_seeds --model residual_marl --seeds 0 1 2 --overwrite -- \
    --updates 100 --collision-penalty 8 --residual-mode candidate_logits
```

`pure_rl` remains as a **legacy** continuous-Gaussian IPPO baseline (different
action space; not the matched comparison).

### Interaction models

| Name | File | Description |
| --- | --- | --- |
| `orca` | `orca.py` | Optimal Reciprocal Collision Avoidance (van den Berg et al., 2011). One reciprocal half-plane per neighbour plus two static half-planes for the corridor edges, solved with the standard 2-D linear program. |
| `social_force` | `social_force.py` | Self-driven particle / social force model (Helbing & Molnár, 1995): relaxation toward the desired velocity, anisotropic exponential repulsion between agents, exponential wall repulsion. |

### Robotic trajectory generation

| Name | File | Description |
| --- | --- | --- |
| `dwa` | `dwa.py` | Dynamic Window Approach (Fox, Burgard & Thrun, 1997). Enumerates the commands reachable under the acceleration and steering-rate limits, forward-simulates each, and scores with the classical heading / clearance / velocity objective. |
| `mppi` | `mppi.py` | Model Predictive Path Integral control (Williams et al., 2016/2017). Perturbs a nominal control sequence, rolls the samples out through the bicycle model, and updates the sequence with the exponentially weighted average. |
| `frenet` | `frenet_planner.py` | Frenet-frame trajectory generation (Werling et al., 2010). Quintic lateral and quartic velocity-keeping longitudinal polynomials sampled over terminal offsets, speeds and horizons, filtered for feasibility and collision, then scored on jerk, time and deviation. |

All three predict neighbours with a constant-velocity model and replan every
step. Conflict is evaluated as footprint overlap in corridor coordinates rather
than with a circumscribed disc, because a disc that covers a 4.5 m long vehicle
forbids the side-by-side passing that a lane-free corridor is full of.

### Learned policies

| Name | File | Description |
| --- | --- | --- |
| `pure_rl` | `pure_rl.py` | IPPO: independent PPO with a shared actor and a decentralised critic, mapping the observation directly to `(accel, steering)`. No utility function, no behavioural prior. The Gaussian lives in a normalised action space and observations are whitened by a running mean/variance estimate, so the baseline is not handicapped by scaling. |
| `pure_rl_safe` | `pure_rl.py` | The same architecture trained with an explicit collision penalty on top of the shared reward. Included because the shared reward only discourages proximity softly, and pure RL exploits that: without the penalty it learns to reach every goal by driving through other vehicles. |
| `mappo` | `marl.py` | MAPPO (Yu et al., NeurIPS 2022): shared actor, centralised critic on the joint corridor state, simultaneous PPO-clip updates. |
| `happo` | `marl.py` | HAPPO (Kuba et al., ICLR 2022): one actor per agent, centralised critic, sequential updates in a random agent order with the accumulated multi-agent probability-ratio factor. Finite-sample updates do not certify monotonic improvement. |
| `hatrpo` | `marl.py` | HATRPO (Kuba et al., ICLR 2022): the same sequential scheme with a KL trust region per agent — conjugate-gradient natural gradient plus a backtracking line search — instead of clipping. |
| `utility_pt` | `utility_prior.py` | Calibrated discrete utility controller (additive one-step planner) with no learning (`temperature=0` gives the deterministic argmax; `utility_pt_logit` samples from a logit choice model over the candidate set). |
| `residual_marl` | `residual_marl.py` | The proposed model: same utility controller plus a learned residual. Default residual is an additive bias on the discrete candidate utilities (continuous credit); `--residual-mode param_delta` restores the older Θ residual for ablations. |

The four RL baselines share the actor architecture, the observation and the
reward, so the comparison isolates the algorithm:

| Algorithm | Actors | Critic input | Update |
| --- | --- | --- | --- |
| IPPO (`pure_rl`) | shared | local observation | PPO-clip, simultaneous |
| MAPPO | shared | centralised state | PPO-clip, simultaneous |
| HAPPO | one per agent | centralised state | PPO-clip, sequential with the advantage factor |
| HATRPO | one per agent | centralised state | KL trust region, sequential with the same factor |

The centralised state is agent-specific: every agent's along-corridor station,
lateral offset, speed, heading and arrival flag, concatenated with the ego
agent's own local observation.

Because ORCA and the social-force model are holonomic, their velocity commands
are inverted through the same bicycle model that constrains every other model,
so no model gets actuation it is not entitled to.

## Usage

Train the learned baselines. The interaction models and planners need no
training, and `residual_marl` uses the checkpoint produced by `RL/train_ppo.py`.

```bash
python -m Baselines.train_pure_rl --updates 200 --episodes-per-update 4 --num-agents 10 \
    --save Baselines/checkpoints/pure_rl_policy.pt

python -m Baselines.train_pure_rl --updates 200 --episodes-per-update 4 --num-agents 10 \
    --collision-penalty 5.0 --save Baselines/checkpoints/pure_rl_safe_policy.pt

python -m Baselines.train_marl --algo mappo  --updates 200
python -m Baselines.train_marl --algo happo  --updates 200
python -m Baselines.train_marl --algo hatrpo --updates 200

# Optional: collision-aware residual (sparse ablation), or dense-trained variant
python -m RL.train_ppo --updates 100 --collision-penalty 5.0 \
    --save RL/checkpoints/residual_collpen_policy.pt
python -m RL.train_ppo --updates 100 --num-agents 16 --collision-penalty 5.0 --dense-spawn \
    --save RL/checkpoints/residual_collpen_dense_policy.pt
```

Paper figures (metrics with realism, distribution panel, stress Frenet):

```bash
python -m Baselines.paper_figures --all
```

### Paper ablation package (comment 4 minimum)

`Baselines/ablation_models.py` defines the full sparse ablation list:

| Model | Role |
| --- | --- |
| `utility_pt` | Calibrated prior, no RL |
| `utility_nominal` | Uncalibrated nominal prior |
| `residual_marl` | Full residual MARL |
| `residual_weights_only` | Freeze Δσ∥, Δσ⊥ (weights only) |
| `residual_sigma_only` | Freeze weight residuals (σ only) |
| `direct_discrete_rl` | Matched direct discrete PPO |
| `mappo` | MAPPO + shared OBB safety layer |
| `residual_nominal` | Residual on nominal prior |

Run with 30 scenarios, 5 training seeds, paired bootstrap CIs, and both
full-lookahead (`1.5 s / 4 substeps`) and no-lookahead (`1 step`) suites:

```bash
python -m Baselines.ablation_stress --lookahead both --train-seeds 0 1 2
python -m Baselines.paper_rerun eval
```

Outputs per suite: `*_raw.csv`, `*_stats.csv` (bootstrap CIs), `*_comparisons.csv`
(vs. full residual), and `*_paired.csv` (key pairwise contrasts).

### Paper revision protocol (multi-seed RL + fair planner comparison)

The utility family still searches over discrete candidates with OBB rejection, but
geometric planners and learned policies now share the same **closed-loop** OBB
filter before any command is executed (`obb_safety_filter` in `sim_config`,
default on). Disable with `--no-obb-safety-filter` for ablations.

```bash
# Check which training-seed checkpoints are missing
python -m Baselines.paper_rerun status

# Train MAPPO + dense collision-penalty residual (5 seeds each)
python -m Baselines.paper_rerun train --jobs 2

# Re-run sparse benchmark + dense stress with training-seed bootstrap CIs
python -m Baselines.paper_rerun eval
```

Or train individual models:

```bash
python -m Baselines.train_seeds --model mappo --seeds 0 1 2 \
    -- --updates 100 --max-steps 240 --collision-penalty 8

python -m Baselines.train_seeds --model residual_collpen_dense --seeds 0 1 2 \
    -- --updates 100 --num-agents 16 --collision-penalty 5 --dense-spawn --max-steps 240
```

Run the benchmark:

```bash
python -m Baselines.benchmark --scenarios 20 --num-agents 12
```

Useful flags:

- `--models orca social_force dwa mppi frenet pure_rl mappo happo hatrpo utility_pt residual_marl` — subset to run
- `--run-id 2 --lane-kf 1` — which measured corridor to simulate on
- `--train-seeds 0 1 2` — evaluate per-seed checkpoints; bootstrap CIs over training seeds
- `--no-obb-safety-filter` — disable shared closed-loop OBB filter (fairness ablation)
- `--residual-checkpoint`, `--pure-rl-checkpoint`, `--checkpoint-dir` — override checkpoint paths
- `--no-realism`, `--no-figures` — skip the data-distribution metrics / plots

Outputs land in `Baselines/results/v3/`:

- `benchmark_raw.csv` — one row per (model, scenario)
- `benchmark_summary.csv` — mean and standard deviation per model
- `benchmark_table.tex` — paper-ready `mean ± std` table
- `benchmark_metrics.png` — headline metric bar panels
- `benchmark_trajectories.png` — all models on the same scenario, in world coordinates
- `benchmark_trajectories_frenet.png` — the same rollouts in corridor coordinates (station vs. lateral offset), which is the readable view for a 590 m × 10 m corridor
- `benchmark_distributions.png` — simulated vs. measured speed / acceleration / lateral-offset distributions

## Metrics

**Safety** — collision events (new overlapping pairs), fraction of agents ever in
a collision, off-corridor rate, minimum surface gap, minimum time-to-collision
(two-disc vehicle approximation), and the fraction of interactions below a
1.5 s TTC.

**Efficiency** — arrival rate, fraction of the along-corridor distance covered,
travel time, mean speed.

**Comfort / plausibility** — mean absolute acceleration, RMS jerk, mean absolute
steering, mean lateral offset from the centreline.

**Realism** — 1-Wasserstein distance and Jensen–Shannon divergence between the
simulated and measured distributions of speed, longitudinal acceleration and
lateral offset on the same corridor. `realism_score` averages the Wasserstein
distances after normalising each feature by its observed standard deviation
(lower is better).

## Adding a model

Create a controller subclassing `BaseController`, then register it in
`registry.py`:

```python
REGISTRY["my_model"] = lambda **kw: MyController(**kw)
LABELS["my_model"] = "My model"
```

It is then available via `--models my_model` with no other changes.

## Baseline algorithm corrections

MPPI is a bounded-sampling variant: it samples truncated Gaussian controls,
weights them using the zero-mean reference/proposal density ratio, and updates
with the weighted bounded controls. It never scores clipped controls and then
updates from different raw perturbations. Proposal parameters are recentered on
the resulting control mean (a bounded moment-update adaptation, not an exact
unconstrained Gaussian projection). Scenario reset deterministically reseeds its
sampler from controller and scenario seeds. SciPy supplies the Gaussian CDFs.

Frenet jerk cost is integrated analytically over each candidate's own horizon;
collision checks match neighbours to actual candidate times and exclude padding.
Sampled Cartesian speed, total acceleration and curvature must satisfy the
shared speed/acceleration limits and the bicycle steering limit. Finite-difference
feasibility checks remain an approximation; execution retains the shared filters.

DWA checks the proposed command followed by straight maximum braking to rest,
including constant-velocity predicted neighbours, even beyond its scoring
horizon. Neighbour geometry remains the documented Frenet-box approximation.
No-admissible-candidate fallback brakes rather than choosing an unsafe accelerator.

HAPPO/HATRPO optimize mean team driving reward. Their critic returns continue
through individual arrivals until team termination or horizon truncation;
inactive actors receive no policy updates, while their team-value targets remain
trainable. HATRPO line search obeys the configured KL threshold without the old
1.5 multiplier. Checkpoint metadata rejects pre-correction sequential policies.
MAPPO/IPPO retain individual rewards; this distinction must be reported in tables.
These fixes do not change the utility formulation or calibrated parameters.
Old planner result tables must be regenerated. No full training run is launched
by these implementation checks.
