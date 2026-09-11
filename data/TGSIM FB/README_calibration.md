# TGSIM Foggy Bottom — calibration-ready outputs

## Files
- `prepared/trajectories_calibration.csv` — all IDs (ego + parked/short/other-class neighbors), 0.1 s grid; `keep_ego` marks calibration targets
- `prepared/per_id_origin_dest.csv` — first sample (initial) and last sample (destination) per ID
- `prepared/vehicle_filter_report.csv` — ego vs scene-only flags
- `prepared/type_code_note.csv` — counts/sizes per class
- `Foggy_Bottom_boundaries.txt` — provided lane polygons (pixels)
- `derived_boundaries/street_boundaries.csv` — **site curb** (union exterior + holes, meters; one curb for every ID)
- `derived_boundaries/street_boundaries_meta.csv`

## Regenerate trajectories
```bat
python "data\TGSIM FB\prepare_for_calibration.py"
```

## Regenerate curb boundaries
```bat
python "data\TGSIM FB\build_street_boundaries.py"
```

## Plot boundary + origin/destination
```bat
python data\_plot_site_boundaries.py --site tgsim
```
Outputs in `data/_qa_plots/`: `tgsim_boundaries_only.png`, `tgsim_boundaries_with_traj.png`, `tgsim_origin_dest.png`, `tgsim_example_ids.png`, `tgsim_boundaries_on_reference.png`

## Calibrate
Per-ID: each ego vehicle starts from its first sample and aims at its last (x,y). Other agents present at that time (parked cars, other classes) are neighbors. Path cost uses `derived_boundaries/street_boundaries.csv`. Default ego class is passenger cars (`class=3`).

```bat
python -m Calibration.calibrate_utility_from_data --csv "data\TGSIM FB\prepared\trajectories_calibration.csv" --output Calibration\utility_calibration_tgsim.json --diagnostics-dir Calibration\diagnostics_tgsim --n-trials 400 --n-restarts 3 --closed-loop-candidates 240 --verbose
```

## Notes
Passenger cars ≈ `class=3` (median box ~6.2 m × 2.2 m). Class 0 has no size in the file and is scene-only unless you override `--class-id`. Ego IDs need more than 20 s on scene and at least 10 m of travel; dropped IDs stay as neighbors. The provided curb extends farther north than the trajectories (~360 m vs ~230 m).
