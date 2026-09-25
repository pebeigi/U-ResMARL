# Baselines

Shared closed-loop harness for the paper comparisons. All models use the same bicycle dynamics, OBB / boundary filters, scenario seeds, and metrics (`Baselines/metrics.py`).

## Main commands

From the repository root:

```bash
python -m Baselines.benchmark --help
python -m Baselines.ablation_stress --help
python -m Baselines._run_2day_pipeline
python -m Baselines._run_param_stress_pipeline
python -m Baselines._run_new_baselines
```

Paper freeway protocol: `20` scenes, `10` agents, `240` steps, seeds `{0,1,2}` for learned models; dense stress uses `16` agents. Models include ORCA, social force, DWA, MPPI, Frenet lattice, MAPPO, matched discrete RL, utility prior, residual MARL, and local CtRL-Sim / CTG++.

Constants and selection rules match the paper appendix *Reproducibility and Training Configuration*.
