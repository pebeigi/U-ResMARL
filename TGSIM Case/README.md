# TGSIM Case

Matched closed-loop transfer on TGSIM Foggy Bottom using `Calibration/utility_calibration_tgsim.json` and the curb in `data/TGSIM FB/derived_boundaries/street_boundaries.csv` (destination-plus-curb frame; no lane IDs).

Same learning / benchmark recipe as the freeway paper protocol (24 PPO updates, seeds 0/1/2, gate and \(\Delta\Theta\)/lookahead/stress suites, CtRL-Sim / CTG++). Site defaults: 6 agents / 80 steps (dense stress: 8 agents).

```bash
python "TGSIM Case/run.py" smoke
python "TGSIM Case/run.py" all --jobs 2
```

Run from the repository root. Outputs go under `TGSIM Case/runs/` (gitignored).
