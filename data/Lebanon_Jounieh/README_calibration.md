# Lebanon Jounieh (roundabout) data

Active (3× metre frame):

- `Jounieh_Road_Boundaries.csv` — site curb (outer ring + islands)
- `prepared/trajectories_calibration.csv` — 0.1 s tracks; `keep_ego` marks calibration egos
- `prepared/per_id_origin_dest.csv`, `lane_code_map.csv`, `vehicle_filter_report.csv`
- `scale_3x_manifest.json` — scale `0.13063734` m/px

Raw 1× inputs (prep only): `Final_Jounieh.csv`, `_source_1x/`.

```bash
python data/Lebanon_Jounieh/scale_jounieh_to_3x.py
python data/Lebanon_Jounieh/prepare_for_calibration.py
```

Working prior: `Calibration/utility_calibration_jounieh.json`.
