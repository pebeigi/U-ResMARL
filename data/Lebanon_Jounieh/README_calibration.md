# Lebanon_Jounieh — calibration-ready outputs

## Files
- `prepared/trajectories_calibration.csv` — all IDs (ego + parked/short neighbors), snapped to a shared 0.1 s clock; `keep_ego` marks calibration targets
- `prepared/per_id_origin_dest.csv` — first sample (initial x,y,speed,heading) and last sample (destination) per ID
- `prepared/lane_code_map.csv` — string `lane_kf` (e.g. `6-2`) → integer code
- `prepared/vehicle_filter_report.csv` — ego vs scene-only flags
- `Jounieh_Road_Boundaries.csv` — **site curb** (one outer + island polygons for every ID)

## Regenerate trajectories
```bat
python data\Lebanon_Jounieh\prepare_for_calibration.py
```

## Plot boundary + origin/destination
```bat
python data\_plot_site_boundaries.py --site jounieh
```
Outputs in `data/_qa_plots/`: `jounieh_boundaries_only.png`, `jounieh_boundaries_with_traj.png`, `jounieh_origin_dest.png`, `jounieh_example_ids.png`

## Calibrate
Per-ID: each vehicle starts from its first sample and aims at its last (x,y). Neighbors at that time (including parked) enter the collision term. Path cost uses `Jounieh_Road_Boundaries.csv`. Class labels are collapsed to the mode per ID; do **not** pass `--class-id` (mixed 1/2 labels would split the same vehicle).

```bat
python -m Calibration.calibrate_utility_from_data --csv data\Lebanon_Jounieh\prepared\trajectories_calibration.csv --output Calibration\utility_calibration_jounieh.json --diagnostics-dir Calibration\diagnostics_jounieh --n-trials 400 --n-restarts 3 --closed-loop-candidates 240 --verbose
```

## Notes
Recorded `length_smoothed` is ~1.1 m (not a usable car footprint); calibration uses 4.5 m × 1.8 m. Ego IDs need more than 20 s on scene and at least 10 m of travel; dropped IDs stay as neighbors.
