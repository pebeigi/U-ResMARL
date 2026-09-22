# Lebanon_Jounieh — calibration-ready outputs

## Active files (use these)

- `Jounieh_Road_Boundaries.csv` — **3× site curb** (outer ring + two islands)
- `prepared/trajectories_calibration.csv` — **3×** trajectories on a shared 0.1 s clock; `keep_ego` marks calibration targets
- `prepared/per_id_origin_dest.csv` — first sample (initial) and last sample (destination) per ID
- `prepared/lane_code_map.csv` — string `lane_kf` → integer code
- `prepared/vehicle_filter_report.csv` — ego vs scene-only flags
- `Jounieh_scale_note.txt` / `scale_3x_manifest.json` — recorded scale (`0.13063734` m/px)

## Do not use for runs

- `Final_Jounieh.csv` — raw **1×** trajectories (input to prepare only)
- `_source_1x/Jounieh_Road_Boundaries_extended_1x.csv` — raw **1×** curb (input to scale script only)

## Regenerate active 3× curb + trajectories
```bat
python data\Lebanon_Jounieh\scale_jounieh_to_3x.py
python data\Lebanon_Jounieh\prepare_for_calibration.py
```

## Plot boundary + origin/destination
```bat
python data\_plot_site_boundaries.py --site jounieh
```
Outputs in `data/_qa_plots/`: `jounieh_boundaries_only.png`, `jounieh_boundaries_with_traj.png`, `jounieh_origin_dest.png`, `jounieh_example_ids.png`

## Calibrate
Per-ID: each vehicle starts from its first sample and aims at its last (x,y). Neighbors at that time (including parked) enter the collision term. Path cost uses `Jounieh_Road_Boundaries.csv`. Class labels are collapsed to the mode per ID; do **not** pass `--class-id`.

```bat
python Calibration\run_three_site_recalibration.py
```

## Notes
Coordinates and image-axis boxes are ×3 in the active prepared table and curb.
Calibration and simulation use a 4.5 m × 1.8 m oriented car with a 2.8 m wheelbase.
Ego IDs need more than 20 s on scene and at least 10 m of travel in the active
metre frame; dropped IDs stay as neighbors.
