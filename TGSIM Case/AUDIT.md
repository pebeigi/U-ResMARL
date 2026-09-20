# TGSIM wiring and completed-run audit — 2026-09-20

The pipeline completed at 06:40 UTC on September 20. Its 112 rows cover 14 models,
eight synthetic scenarios, six agents per scenario, at most 80 half-second steps,
and one training seed. These are synthetic closed-loop tests on the TGSIM curb
map, not held-out recorded-trajectory accuracy tests.

**Do not use the current ranking as paper evidence.** A reproduction of utility
scene 3 exactly matched the saved arrival, progress, PDMS and collision results,
and demonstrated incorrect destination accounting. Existing result files and
checkpoints have been preserved.

## Blocking wiring errors

1. **Destinations leak between scenarios and agents.** `network.load_site_corridor`
   caches one mutable `SiteNetworkCorridor`. Spawning binds destinations on that
   object. `Baselines.benchmark` constructs all scenarios before rolling them out,
   so the final scenario replaces the bindings for earlier scenarios. In addition,
   `_dest_for(point)` selects a destination by the nearest stored initial position,
   rather than the identity of the moving agent. The shared runner does not refresh
   these bindings. Residual environment steps do refresh them, introducing another
   training/evaluation difference.

   Reproduction: building seed 7 after seed 3 changed seed 3's station values by
   as much as **111.799 m without moving a car**. Utility seed 3 reports 100% arrival
   and 0.246269 goal progress. One reported arrival remains **48 m geodesically
   from its own goal**. Recomputing progress against each own goal gives 0.928067
   for that same trace. This is not a corrected score: the false arrival already
   changed the rollout by removing that agent. Re-evaluation is required.

2. **The polygon does not implement the corridor geometry expected by planners.**
   `network.py` supplies the outer curb as both `center` and `lower`, and the inner
   block boundary as `upper`. `build_local_frame` treats those as a route with
   paired road edges. Network station is negative geodesic distance, whereas
   `cumulative_s` is positive outer-curb arc length. `xy_from_frenet(s, lateral)`
   ignores both inputs and always returns the same representative point. This
   misdirects ORCA/Social Force preferred velocities and invalidates DWA/MPPI/Frenet
   route generation and costs. Their low arrivals cannot be attributed to their
   algorithms. CTG++ road guidance also assumes paired corridor edges; the current
   map adapter does not satisfy that assumption.

3. **Residual categorical scoring bypasses the TGSIM direction adapter.**
   `activate.apply` imports `RL.candidate_policy` before replacing
   `utility_model.evaluate_candidate_utility`. Its imported function reference
   stays old and never sets the adapter's `_tls['sim']`. The monkeypatched direction
   term therefore falls back to its original implementation. On seed 3 candidates,
   the two utility entry points differed by up to **4.859934 utility units**.
   Fixing this requires consistent adapter dispatch, not changing calibrated
   utility coefficients.

4. **Offline progress targets use the same broken implicit destination lookup.**
   `prepare_new_baselines` constructs recorded scenes without binding their goals;
   `new_baselines.data.scene_arrays` calls `corridor.project(p/q)` for progress.
   Depending on invocation history, the target is a previous synthetic agent's
   goal or the representative fallback point. Regenerate CtRL-Sim return targets
   after replacing this with explicit per-agent goal distance. The map features
   also label the outer curb as a centerline. Recorded endpoint goals should be
   disclosed as conditioning; they differ from the highway exit-goal protocol.

## Protocol gaps

- Existing online checkpoint summaries record the shared safety/PDMS selector,
  but the geometry bugs contaminate its inputs. MAPPO, HAPPO and continuous PPO
  correctly report `safety_failed`; this does not diagnose convergence.
- Online training is still **24 updates**. Residual consumed **5,085 environment
  steps**, each other online learner **7,680**. These are short pilots, not
  matched interaction budgets or asymptotic comparisons. Setting only
  `TOTAL_ENV_STEPS=100000` cannot extend training past `TRAIN_UPDATES=24` because
  the update limit remains a second stopping condition.
- CtRL-Sim and CTG++ trained for 10,000 optimizer steps and were selected by
  offline validation loss (steps 500 and 9,750 respectively). `run.py` bypasses
  the new common closed-loop `select` stage and does not request strict protocol
  validation at benchmark time.
- The manifest hashes shared code but omits TGSIM adapter files, curb geometry,
  and the site calibration contents. Its highway spawn-window metadata does not
  describe polygon sampling. Add a site-specific protocol/configuration manifest.
- The README's "hold still" fallback returns acceleration 0, steering 0 at the
  current speed; this is coasting, not holding. The shared boundary filter still
  rejects infeasible commands. Do not interpret that fallback as a physical stop.
- Plot helpers can load a live resume actor in place of the selected export;
  verify plotted checkpoints separately before using figures to explain tables.

## What the saved numbers say, before correcting wiring

Utility: 0 collision events per scene, 72.9167% arrival, PDMS 0.526571.
Residual: 0 collision events per scene, 70.8333% arrival, PDMS 0.542655.
Thus residual has a slightly higher reported composite score, but one fewer
reported arrival among 48 agents. It does not beat utility on every metric.

All models report zero off-road exposure. DWA/Frenet average speeds are only
0.398/0.238 m/s, consistent with stopping under unsuitable route geometry.
CtRL-Sim/CTG++ arrival is 8.3333%/2.0833%, with average execution intervention
rates of 40.37%/32.43%. Those results need adapter correction before interpretation.
Collision counts measure simulated vehicle contacts, but false arrivals can
remove vehicles prematurely, so even safety totals need a new evaluation.

The new-model caches do have separated train/validation tracks, 20/6 scenes,
and small mean inverse-dynamics reconstruction error (0.00156/0.00630 m).
That is a useful preprocessing check, not validation of goal/route semantics.
Prior exposure remains explicitly unverified in the recorded-data manifests.

## Repair and verification order

1. Make geometry immutable across scenarios. Pass each agent's destination
   explicitly to progress, observations, reward, arrival, recorder and gate.
   Use physical destination distance for arrival and test order invariance.
2. Give controllers a valid agent-specific route through the curb network and
   hole-aware boundary queries. Verify route/frame inverse consistency. Adapt
   CTG++ polygon guidance and map types separately.
3. Fix utility dispatch and regenerate offline goals/returns/maps. Verify zero
   residual equals the calibrated prior through both execution paths.
4. Wire common closed-loop checkpoint selection and strict configuration checks;
   record site hashes and actual interaction budgets. Then run small regression
   tests, followed by new training/evaluation only after those pass.

Evidence: `results/benchmark_raw.csv`, `results/experiment_manifest.json`,
`logs/pipeline.log`, checkpoint summaries, and the isolated read-only reproduction
in `Baselines/results/tgsim_audit/probe.py` / `probe.json`.
