# U-ResMARL (anonymous code)

Anonymous code for the ICLR submission on utility-guided residual multi-agent RL for closed-loop interactive motion generation.

## Layout

| Path | Role |
|------|------|
| `utility_model.py` | Discrete utility prior / bicycle candidates |
| `RL/` | Residual PPO, env, reward, PDMS / accept-if-better gate |
| `Baselines/` | Shared closed-loop harness and paper baselines |
| `Calibration/` | Site calibration + working `utility_calibration*.json` |
| `New Baselines/` | Local CtRL-Sim / CTG++ adaptations |
| `TGSIM Case/`, `Roundabout Case/` | Matched transfer sites |
| `Sensitivity/` | Sensitivity analysis scripts |
| `data/` | Boundaries and prep scripts (raw trajectory CSVs not shipped) |
| `tests/` | Protocol / unit checks |

Ignored locally (not in this release): manuscript, runs, results, checkpoints, plots.

## Setup

```bash
pip install -r requirements.txt
```

Run from the repository root.

```bash
python -m pytest tests/ -q
```

## Paper protocol (freeway)

Matched low-budget online study: `24` PPO updates, four `240`-step episodes per update, `10` agents, seeds `{0,1,2}` (`23{,}040` environment steps per seed). Checkpointing uses safety non-regression versus the frozen prior, then PDMS. Deterministic eval uses the accept-if-better gate. Exact constants are in the paper appendix *Reproducibility and Training Configuration*.

Entry points:

- Residual / online learners: `python -m RL.train_ppo` (see also site `run.py` launchers)
- Benchmark / ablations: `python -m Baselines.benchmark`, `python -m Baselines.ablation_stress`
- Freeway pipelines: `python -m Baselines._run_2day_pipeline`, `python -m Baselines._run_param_stress_pipeline`
- Calibration: `python calibrate_utility_from_data.py` or `python -m Calibration.calibrate_utility_from_data`
- CtRL-Sim / CTG++: `New Baselines/run.py` (see that folder’s README)

Raw multi-GB trajectories are omitted; use the data sources cited in the paper with the shipped boundary / prep scripts under `data/`.