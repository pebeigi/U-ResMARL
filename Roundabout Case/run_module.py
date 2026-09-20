"""Run an existing trainer/benchmark module after applying the roundabout patch."""
from __future__ import annotations

import os
import runpy
import sys
from pathlib import Path

CASE_ROOT = Path(__file__).resolve().parent
REPO_ROOT = CASE_ROOT.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(CASE_ROOT))

import activate  # noqa: E402
import torch

torch.set_num_threads(int(os.environ.get("ROUNDABOUT_TORCH_THREADS", "2")))

activate.apply()

if len(sys.argv) < 2:
    raise SystemExit("usage: python run_module.py MODULE [--args ...]")
module = sys.argv[1]
sys.argv = [module, *sys.argv[2:]]
if module == "new_baselines_cli":
    runpy.run_path(str(REPO_ROOT / "New Baselines" / "run.py"), run_name="__main__")
else:
    runpy.run_module(module, run_name="__main__", alter_sys=True)
