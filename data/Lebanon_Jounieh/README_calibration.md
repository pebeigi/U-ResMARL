# Lebanon_Jounieh — calibration-ready outputs

## Files
- `prepared/trajectories_calibration.csv` — highway-compatible schema, subsampled to ≈0.1 s; parked and tracks shorter than 10 s are dropped
- `prepared/lane_code_map.csv` — string `lane_kf` (e.g. `6-2`) → integer code
- `prepared/vehicle_filter_report.csv` — keep/drop flags per vehicle
- `Jounieh_Road_Boundaries.csv` — **site curb** (one outer + island polygons for every ID)

## Regenerate trajectories
```bat
python data\Lebanon_Jounieh\prepare_for_calibration.py
```

## Plot boundary + trajectories
```bat
python data\_plot_site_boundaries.py
```
Output: `data/_qa_plots/jounieh_boundaries_with_traj.png`

## Calibrate
Uses `Jounieh_Road_Boundaries.csv` for path cost. Direction and distance follow each ID’s destination **along the roadway** (not the Euclidean chord). Off-road path cost keeps growing with distance; `v_max` is 1.15× p99 speed (no 12 m/s floor).

```bat
python -m Calibration.calibrate_utility_from_data --csv data\Lebanon_Jounieh\prepared\trajectories_calibration.csv --class-id 2 --output Calibration\utility_calibration_jounieh.json --diagnostics-dir Calibration\diagnostics_jounieh --n-trials 400 --n-restarts 3 --closed-loop-candidates 240 --verbose
```

## Notes
Vehicle lengths in this file look unusually small (~1 m); inspect before trusting footprint-sensitive terms. Class 2 has most of the traffic. Prep drops vehicles present < 10 s or with 85th-percentile speed < 1 m/s (or path < 8 m).
