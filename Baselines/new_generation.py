"""Lazy import bridge for the user-requested 'New Baselines' directory."""
from pathlib import Path
import sys


def build_new_controller(name, **kwargs):
    package_root = str(Path(__file__).resolve().parents[1]/'New Baselines')
    if package_root not in sys.path:
        sys.path.insert(0, package_root)
    import torch
    kwargs.setdefault("device", "cuda" if torch.cuda.is_available() else "cpu")
    from new_baselines.controller import TrafficGenerationController
    return TrafficGenerationController(name=name, **kwargs)
