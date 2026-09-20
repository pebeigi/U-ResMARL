"""Moved to ``Ref Images/over_time/plot_over_time.py``."""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

script = Path(__file__).resolve().parents[1] / "Ref Images" / "over_time" / "plot_over_time.py"
cmd = [sys.executable, "-u", str(script), "--site", "tgsim", *[a for a in sys.argv[1:] if a != "--site"]]
raise SystemExit(subprocess.call(cmd, cwd=str(script.parents[2]), env={**os.environ, "CUDA_VISIBLE_DEVICES": ""}))
