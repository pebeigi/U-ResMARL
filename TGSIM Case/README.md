# TGSIM Case

Isolated Foggy Bottom experiment. Freeway files, checkpoints, and paper tables are not modified.

Uses the already-fitted prior in `Calibration/utility_calibration_tgsim.json` and the extracted **network curb** in `data/TGSIM FB/derived_boundaries/street_boundaries.csv` (outer curb minus the block hole). That is the same `path_mode=boundary` / destination-frame geometry as TGSIM calibration. PCA lane tubes (`lane_kf`) are not used.

If the bicycle grid has no in-bounds candidate, TGSIM Case holds still for that step on the **same 7×9 action index** used by the highway trainers (freeway code is unchanged).

## Shared protocol

Closed-loop decision, observations, and OBB checks are the highway stack:

- `RL.decision.local_agents` / `neighbor_indices` for every actor and predictor
- shared 7×9 bicycle grid and `obb_safety_filter`
- `RL.experiment_protocol` validation ranking (`common_safety_then_pdms_v1`), 16 val episodes every 10 updates
- same 2-day learned set as the freeway pipeline: residual MARL, residual-param, direct discrete, MAPPO
- same closed-loop benches: ORCA, social force, DWA, MPPI, Frenet, utility prior, the four learners, CtRL-Sim, CTG++
- same follow-up tables: param / no-lookahead / dense stress and gate ablation

Site-specific: recorded traffic initialization, TGSIM calibration, 6 agents /
80 steps, 20 m perception radius, and vehicle dimensions from TGSIM. Dense
stress uses 8 agents (the site analogue of freeway 10→16).

## Commands

Run from the repository root:

```powershell
python "TGSIM Case/run.py" smoke
python "TGSIM Case/run.py" all --jobs 2 --run-dir "TGSIM Case/runs/paper_2day"
```

`all` runs smoke, trains the 2-day learned set (24 PPO updates, seeds 0/1/2),
trains CtRL-Sim/CTG++, then the freeway-matched benchmark + stress + gate
ablation. Scene geometry stays TGSIM-specific; the update budget, validation
cadence, model roster and eval suite match `Baselines/_run_2day_pipeline.py`.

`smoke` runs actual site regressions and a bounded learning integration check:
two offline training steps plus selection/reload for each new baseline, and
one update plus reload for each of the six online learners. These verify
execution, not convergence or performance. Smoke outputs live under
`runs/<run>/smoke/`.

Checkpoints, logs and tables live under `TGSIM Case/runs/<run>/`.
Closed-loop initialization samples a simultaneous subcohort of eligible moving
recorded vehicles. Positions,
headings and backward-difference velocities are preserved; overlapping,
off-road or braking-infeasible samples are rejected rather than relocated or
slowed down. Desired speed remains the fixed protocol value (8 m/s).

Each destination is that vehicle's last recorded pose with its complete
footprint inside the curb, strictly later than its sampled start. The endpoint
is supplied equally to all controllers as navigation intent. Future trajectories
are not replayed as actions or provided as planned routes.

Tracks crossing the chronological 60/20/20 split are purged. Seeds
910000–919999 sample validation, 810000–819999 sample test, and other seeds
sample train. Sampling is reproducible for a given seed. Routing still uses
the curb visibility graph; signals, right-of-way and lane-direction rules
are not enforced.

This is protocol v3 with the freeway 2-day recipe. Use `--run-dir "TGSIM Case/runs/paper_2day"`;
previous random-spawn or 30k-step checkpoints must be regenerated.
