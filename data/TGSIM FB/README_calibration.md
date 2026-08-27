# TGSIM Foggy Bottom — calibration-ready outputs

## Files
- `prepared/trajectories_calibration.csv` — highway-compatible schema (`speed_kf`/`acceleration_kf` from x/y, `run_id=1`, `class=type_most_common`); parked and tracks shorter than 10 s are dropped
- `prepared/vehicle_filter_report.csv` — keep/drop flags per vehicle
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

## Plot boundary + trajectories
```bat
python data\_plot_site_boundaries.py
```
Output: `data/_qa_plots/tgsim_boundaries_with_traj.png`

## Calibrate
Uses `derived_boundaries/street_boundaries.csv` for path cost. Direction and distance follow each ID’s destination **along the street** (not the Euclidean chord). Off-road path cost keeps growing with distance.

```bat
python -m Calibration.calibrate_utility_from_data --csv "data\TGSIM FB\prepared\trajectories_calibration.csv" --class-id 3 --output Calibration\utility_calibration_tgsim.json --diagnostics-dir Calibration\diagnostics_tgsim --n-trials 400 --n-restarts 3 --closed-loop-candidates 240 --verbose
```

## Notes
Passenger cars ≈ `class=3`. Prep drops vehicles present < 10 s or with 85th-percentile speed < 1 m/s (or path < 8 m).
