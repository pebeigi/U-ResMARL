# Roundabout Case

Isolated Jounieh roundabout experiment. Freeway files, TGSIM Case, checkpoints, and paper tables are not modified.

Uses the fitted prior in `Calibration/utility_calibration_jounieh.json` and the extracted **site curb** in `data/Lebanon_Jounieh/Jounieh_Road_Boundaries.csv` (outer ring minus two islands). Same `path_mode=boundary` / destination-frame geometry as Jounieh calibration. PCA lane tubes are unused.

The supplied `0.04354578 m/pixel` scale is retained. Closed-loop boxes are
1.6 × 0.64 in that frame, with a 0.99 wheelbase. The absolute metre scale is
**not independently verified**. The legacy Jounieh calibration used 4.5 × 1.8
footprints, so it must be refitted before claiming a scale-matched calibrated
comparison. Original recordings and calibration files are preserved.

`python tools/audit_roundabout_scale.py` regenerates
[`figures/scale_audit.png`](figures/scale_audit.png): identical recorded cars
with the current footprint, calibration footprint, and recorded image-axis
boxes. 99.985% of recorded centres lie inside the curb. This supports map/track
alignment, not the correctness of the absolute metre conversion.

## Shared protocol

Closed-loop decision, observations, and OBB checks are the highway stack:

- `RL.decision.local_agents` / `neighbor_indices`
- shared 7×9 bicycle grid and `obb_safety_filter`
- `RL.experiment_protocol` ranking, 16 val episodes every 10 updates, 24 PPO updates
- same 2-day learned set as the freeway pipeline: residual MARL, residual-param, direct discrete, MAPPO
- same closed-loop benches and param/gate follow-ups as `Baselines/_run_2day_pipeline.py`

Site-specific: 6 agents / 80 steps, 12 m perception, fixed 4 m/s desired speed.
Dense stress uses 8 agents.
Initialization now samples simultaneous recorded positions, backward-difference
velocities and headings. It selects a nonoverlapping subcohort with a shared
braking backup, without slowing vehicles to make them fit. Tracks crossing the
chronological 60/20/20 split are purged. Seeds 910000–919999 use validation,
810000–819999 use test, and other seeds use train.

Each car's last footprint-contained recorded endpoint is supplied to every
controller as navigation intent. Future motion is not replayed as control or
used as the planned route. This is goal-conditioned driving, not endpoint-free
prediction. Routing still uses the curb visibility graph; lane direction,
right-of-way and later vehicle arrivals are not enforced. Initial traffic is
recorded, but the closed-loop result is not a full traffic-rule reproduction.

## Commands

Run from the repository root:

```powershell
python "Roundabout Case/run.py" smoke
python "Roundabout Case/run.py" all --jobs 2 --run-dir "Roundabout Case/runs/paper_2day"
```

`all` runs smoke, trains the 2-day learned set (24 PPO updates, seeds 0/1/2), trains CtRL-Sim/CTG++, then the freeway-matched benchmark + stress + gate ablation.

`smoke` executes the site regression tests, then tiny real-data training,
selection and reload checks for CtRL-Sim/CTG++ and one update of every online
learner. Its outputs stay under `runs/<run>/smoke/` and establish execution
only. Earlier random-spawn/wrong-wheelbase checkpoints require fresh training;
they are not compatible with this protocol. Use `--run-dir` consistently and
`--resume` to continue completed stages in an unchanged run.

Checkpoints / logs / tables: `Roundabout Case/runs/<run>/`
