# CtRL-Sim and CTG++ in the local traffic simulator

This folder contains runnable **local adaptations**, with offline training,
checkpoint loading, and controllers for the existing benchmark. The calibrated
utility, its parameters, and the residual RL algorithm are not part of either
new policy. Both policies execute through the same bicycle dynamics and shared
boundary/OBB filters as the existing models.

**Status:** smoke-tested implementations, not trained paper benchmarks. A smoke
checkpoint is explicitly rejected by the normal benchmark loader. No performance
claim follows from a short smoke rollout or from matching model architecture.

## Public code and attribution

- [CtRL-Sim official repository](https://github.com/montrealrobotics/ctrl-sim),
  commit `4c3ffa260f9e39b7c930614612bae01f0f85cd0f` (MIT).
  The encoder, decoder, map encoder, MLP and initialization/causal-mask code are
  vendored under `new_baselines/vendor/`, with package-relative imports.
- **CTG++ uses the CtRL-Sim authors' public reimplementation**, from that same
  commit, specifically its `modules/ctg_arch.py` scene diffusion transformer.
  This is the reimplementation distinguished from the original in the CtRL-Sim
  paper. It retains temporal, relative social and per-agent map attention. It is
  not an independent per-agent trajectory MLP.
- [Original NVIDIA CTG/CTG++ repository](https://github.com/NVlabs/CTG), inspected
  at `f916c008c3ecf2360bfa050639606eaab7c207f5`. Its source carries an NVIDIA
  non-commercial research/evaluation license. No NVIDIA source or pretrained
  weights are copied into the runnable local package. The optional local clone
  is under ignored `upstream/` for reference.

`PROVENANCE.json` records upstream revisions and file hashes. The vendored MIT
license is `new_baselines/vendor/LICENSE`. Both upstream checkouts are ignored;
the selected vendored model cores are self-contained, so cloning upstream is not
required to run this package. No Nocturne, Lightning, PyG, Waymo download, language
model service, or additional compiled dependency is required.

References:
[CtRL-Sim (CoRL 2024)](https://arxiv.org/abs/2403.19918),
[CTG++: Language-Guided Traffic Simulation via Scene-Level Diffusion (CoRL 2023)](https://proceedings.mlr.press/v229/zhong23a.html).

## One entry point

Run these commands from the repository root with the existing Python environment
(tested with Python 3.8.10, PyTorch 2.1.2; NumPy, pandas and the existing simulator
dependencies). Paths containing spaces must be quoted.

```powershell
# Tiny real-data training + checkpoint round trip + shared-simulator rollouts.
python "New Baselines/run.py" smoke

# Build disjoint training and validation caches from the recorded highway data.
python "New Baselines/run.py" prepare --train-scenes 20 --validation-scenes 6

# Train each model independently. These commands are not launched automatically.
python "New Baselines/run.py" train ctrl_sim --steps 10000 --device cuda
python "New Baselines/run.py" train ctg_plus_plus --steps 10000 --device cuda

# Select retained training candidates using the common safety/PDMS rule.
python "New Baselines/run.py" select ctrl_sim --candidates "New Baselines/checkpoints/ctrl_sim_policy.candidates" --output "New Baselines/checkpoints/ctrl_sim_policy.pt"
python "New Baselines/run.py" select ctg_plus_plus --candidates "New Baselines/checkpoints/ctg_plus_plus_policy.candidates" --output "New Baselines/checkpoints/ctg_plus_plus_policy.pt"

# Existing metrics, identical scenario seeds, shared filters and utility comparator.
python -m Baselines.benchmark --models utility_pt ctrl_sim ctg_plus_plus --scenarios 20 --seed 810000 --checkpoint-dir "New Baselines/checkpoints" --require-matched-protocol --output "New Baselines/results/closed_loop"

# Recorded-data trajectory errors and behavior realism on the untouched test split.
python -m Baselines.benchmark --models utility_pt ctrl_sim ctg_plus_plus --scenarios 10 --data-evaluation --data-split test --checkpoint-dir "New Baselines/checkpoints" --output "New Baselines/results/recorded_test"

python -m unittest discover -s tests -p test_new_baselines.py -v
```

The default checkpoint files are `checkpoints/ctrl_sim_policy.pt` and
`checkpoints/ctg_plus_plus_policy.pt`. Use `--output` and `--seed` to train multiple
seeds; the existing benchmark accepts `_seed0.pt`, `_seed1.pt`, etc. via
`--train-seeds`. New models are available by name but are not added to the default
benchmark list, because an untrained policy is not a valid silent fallback.

The 10,000-step examples specify a starting training budget, not evidence of
convergence. Inspect training and validation curves, then run closed-loop
validation before freezing the final test protocol. Training exports the minimum
offline-loss checkpoint and saves every validation candidate. The `select` stage
is required for a matched closed-loop comparison; offline loss alone is not the
paper selector. Use the same validation agent count, horizon, seeds, density and
calibration as the online models. Training records steps, seed, source hashes,
manifest hashes and provenance. Training does not resume an optimizer state; each invocation starts a
new run. Generated data, results and checkpoints stay in ignored directories.

## What is preserved, and what is adapted

**CtRL-Sim** retains factored return prediction, return-conditioned discrete
action prediction, auxiliary future position prediction, agent/state/return/action
tokenization and the upstream multi-agent causal mask. At deployment, returns are
sampled from the model, tilted by `exp(kappa * return)`, and actions are sampled
conditional on those returns. Recorded future returns never enter deployment.
Current other-agent actions/returns and future tokens cannot affect the current
prediction. Past control tokens are inferred from executed transitions so that
rejected controls are not recorded as if executed.

**CTG++ reimplementation** retains its scene DiT, cosine diffusion schedule,
clean-sample (`x0`) prediction, state-plus-action diffusion and receding-horizon
execution of the first action. The repository's diffusion wrapper imported a
missing guidance module and skipped reverse timesteps while using single-step
posteriors. The local wrapper uses every consecutive reverse step. Its initial
and reverse noise scale of 0.5 follows the released CtRL-Sim reimplementation.
The local dense relative-attention port uses the same projections, per-target
softmax and gated aggregation; a numerical edgewise reference test checks it.
The temporal attention padding sentinel is isolated from the social existence
mask so that padded agents cannot become active interaction partners.

Local adaptations that must be disclosed in the paper:

- Decision-time groups use the shared nearest-six neighbor set within 60 m by
  default. Joint sampling is used only when every active agent can see the whole
  group; otherwise each focal gets its own local group. Offline groups use the
  same configured visibility limits. Historical checkpoints trained before this
  protocol must not be relabeled as matched models.

- Lebanon corridor polylines replace Waymo maps. Twelve polylines with twenty
  points encode the centerline and both boundaries. The default model widths and
  encoder/decoder depths match the released CtRL-Sim architecture; map counts,
  context lengths and the training budget differ.
- `dt=0.5 s`, local bicycle dynamics, acceleration bounds `[-4,4] m/s²` and steering
  bounds `[-0.45,0.45] rad` replace Nocturne timing/action ranges. CtRL-Sim retains
  20 by 50 action bins and 350 return bins. Its even action codebook does not have
  an exact zero bin, as in the upstream discretization.
- Known corridor exit, desired speed and exit heading provide goal conditioning.
  No held-out future endpoint is supplied as a privileged goal. These goals differ
  materially from Waymo endpoint-conditioned experiments.
  This describes the highway preparation. The separate TGSIM and roundabout
  offline builders use the last on-road recorded point in each window as the
  training goal. They are endpoint-conditioned adaptations. Both sites' recorded
  closed-loop initialization likewise supplies each track's exit as explicit
  navigation intent to all controllers. These site results must not be described
  as endpoint-free forecasting or as exact recorded traffic-rule reproduction.
- Offline controls invert this simulator's semi-implicit bicycle from consecutive
  observations. This is one-step, observation-conditioned inverse dynamics, not
  Nocturne's full physics replay preprocessing. Each data manifest reports action
  clipping and position reconstruction error; inspect these before a large run.
  Labels and generated trajectories still have to pass the common execution
  filters. The state and action heads are not asserted to be dynamically consistent
  before execution.
- CtRL-Sim's three local reward components are bounded route progress, nearest
  vehicle clearance/contact and signed road-footprint clearance. They are distinct
  from the frozen utility and from our residual RL reward. Finite-horizon mean
  returns in `[-1,1]` replace Waymo reward scales. Targets with incomplete future
  coverage are masked, rather than treating track disappearance as a safe outcome.
- CTG++ uses three observed states and a ten-step future by default. Relative
  future conditioning repeats the last observed relation, never ground truth.
  Relative velocity uses an actual rotated velocity difference; the released
  helper's inconsistent velocity equations are not copied. Agent-centric axes
  point forward along local +y, with positions/velocities scaled by 100/40.
- All initial agents remain in simulation. Up to 24 agents are generated jointly;
  larger scenes use a nearest-neighbor group for each focal agent. This differs
  from unlimited scene-wide attention and must be included in scalability claims.
  Each controller cold-starts with the initial observed state, then accumulates
  history. It receives no extra logged history or future compared with competitors.
- The original CTG++ language-to-constraint interface is omitted. Optional local
  differentiable guidance uses two-disc collision, footprint road-clearance and
  route-goal costs through bicycle rollouts; it is not the original LLM/STL layer.
  `guidance_scale=0` is the default, matching the unguided released reimplementation.
  Road/OBB execution filters remain active regardless of this guidance setting.

## Data and comparison protocol

`prepare` reuses `Baselines.data_evaluation.build_recorded_scenes`: chronological
60/20/20 partitions, purged cross-partition tracks, nonoverlapping scene windows,
causal initialization and identical maps/footprints. Only train and validation
caches may be prepared here. Training checks identity disjointness and source
hashes. The test split is used only by the existing evaluator. Existing utility
calibration predates the split and has unverified historical exposure; the data
manifests preserve that limitation rather than claiming a newly independent test
for the old calibration.

Model settings live in `new_baselines/config.py`. `prepare --config settings.json`
accepts a JSON object of Config fields. `train --config settings.json` can adjust
model/inference settings while checking data geometry and horizons match the
cache. Omitted fields use the documented defaults. Return tilts default to zero;
choose goal/vehicle/road tilts and diffusion guidance using validation scenes,
record all candidates, then freeze them before test evaluation. A high tilt is
not guaranteed to improve behavior. No utility-score acceptance gate is used.

Report the existing arrival/collision/boundary, progress, comfort and runtime
metrics alongside held-out ADE/FDE, speed/heading errors and traffic-distribution
distances. Use paired scenario seeds and several training seeds. Report shared
filter settings and intervention rates if using safety results to claim policy
quality; current shared benchmark outcomes include filtering and cannot establish
unfiltered policy safety. Local results must not be presented as reproduced
Waymo/nuScenes scores or as an exact reproduction of the original CTG++ system.
