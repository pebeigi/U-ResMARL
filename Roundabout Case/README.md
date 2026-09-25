# Roundabout Case

Matched closed-loop transfer on the Jounieh roundabout (3× metre scale) using `Calibration/utility_calibration_jounieh.json` and `data/Lebanon_Jounieh/Jounieh_Road_Boundaries.csv` (destination-plus-curb frame; no lane IDs).

Same learning / benchmark recipe as the freeway paper protocol (24 PPO updates, seeds 0/1/2, gate and \(\Delta\Theta\)/lookahead/stress suites, CtRL-Sim / CTG++). Site defaults: 6 agents / 80 steps (dense stress: 8 agents).

```bash
python "Roundabout Case/run.py" smoke
python "Roundabout Case/run.py" all --jobs 2
```

Run from the repository root. Outputs go under `Roundabout Case/runs/` (gitignored).
