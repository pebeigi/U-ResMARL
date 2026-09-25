# New Baselines (CtRL-Sim / CTG++)

Local adaptations of CtRL-Sim and CTG++ for the shared bicycle harness (same dynamics, boundary/OBB filters, and safety/PDMS selection as residual MARL).

## Attribution

- CtRL-Sim: [montrealrobotics/ctrl-sim](https://github.com/montrealrobotics/ctrl-sim) (MIT); encoder/decoder cores vendored under `new_baselines/vendor/`.
- CTG++: scene DiT from that public reimplementation (no NVIDIA CTG source or language stack). See `PROVENANCE.json`.

Upstream clones under `upstream/` are optional and gitignored.

## Run

```bash
python "New Baselines/run.py" --help
```

Offline training budgets and reverse-diffusion settings used in the paper are listed in the reproducibility appendix.
