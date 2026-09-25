# TGSIM Foggy Bottom data

- `prepared/trajectories_calibration.csv` — 0.1 s tracks; `keep_ego` marks calibration egos
- `prepared/per_id_origin_dest.csv`, `vehicle_filter_report.csv`, `type_code_note.csv`
- `derived_boundaries/street_boundaries.csv` — site curb (metres)
- `Foggy_Bottom_boundaries.txt` — source lane polygons (pixels)

```bash
python "data/TGSIM FB/prepare_for_calibration.py"
python "data/TGSIM FB/build_street_boundaries.py"
```

Working prior: `Calibration/utility_calibration_tgsim.json` (default ego class = passenger cars).
