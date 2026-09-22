# 1× provenance only — do not use for calibration or simulation

Active metre-frame files live one level up:
- ../Jounieh_Road_Boundaries.csv
- ../prepared/trajectories_calibration.csv

`Jounieh_Road_Boundaries_extended_1x.csv` is the pre-3× curb used only by
`../scale_jounieh_to_3x.py` to regenerate the active curb. Raw trajectories
remain at `../Final_Jounieh.csv` (also 1×); `prepare_for_calibration.py`
applies the ×3 scale when writing `prepared/`.
