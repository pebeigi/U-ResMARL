"""Compatibility shim — proposal plots live in Sensitivity/."""

import runpy
from pathlib import Path

if __name__ == "__main__":
    runpy.run_path(
        str(Path(__file__).resolve().parent / "Sensitivity" / "Proposal_Plots.py"),
        run_name="__main__",
    )
