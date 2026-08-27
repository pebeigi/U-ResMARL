"""Load calibrated utility parameters for residual MARL."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from RL._paths import REPO_ROOT
from RL.param_gauge import (
    AMPLITUDE_LOGIT_KEYS,
    DEFAULT_RESIDUAL_SCALES,
    LEGACY_RESIDUAL_PARAM_KEYS,
    LEGACY_RESIDUAL_SCALES,
    PARAM_GAUGE,
    RESIDUAL_PARAM_KEYS,
    SHAPE_RESIDUAL_KEYS,
    apply_gauge,
    is_legacy_delta,
    residual_vector_to_dict,
)

DEFAULT_CALIBRATION_PATH = REPO_ROOT / "Calibration" / "utility_calibration.json"

# Re-export for callers that imported from here historically.
__all__ = [
    "AMPLITUDE_LOGIT_KEYS",
    "DEFAULT_CALIBRATION_PATH",
    "DEFAULT_RESIDUAL_SCALES",
    "LEGACY_RESIDUAL_PARAM_KEYS",
    "LEGACY_RESIDUAL_SCALES",
    "PARAM_GAUGE",
    "RESIDUAL_PARAM_KEYS",
    "SHAPE_RESIDUAL_KEYS",
    "apply_residual",
    "clip_params_rl",
    "load_base_params",
    "residual_scales_for_checkpoint",
    "residual_vector_to_dict",
]

# Clip bounds aligned with calibrated / near-optimal ranges (not the old GSA box).
RL_PARAM_BOUNDS: dict[str, tuple[float, float]] = {
    "S_theta": (0.05, 12.0),
    "S_v": (0.05, 12.0),
    "xi_i": (1.0, 10.0),
    "S_d": (0.05, 12.0),
    "gamma": (0.05, 10.0),
    "w_x": (0.05, 15.0),
    "w_y": (0.05, 15.0),
    "w_c": (0.01, 1200.0),
    "w_ell": (0.1, 300.0),
    "beta": (0.01, 12.0),
    "sigma_long": (0.3, 6.0),
    "sigma_lat": (0.2, 3.0),
}


def clip_params_rl(params: dict[str, float]) -> dict[str, float]:
    import numpy as np
    from utility_model import DEFAULT_KERNEL_PARAMS

    out = {**DEFAULT_KERNEL_PARAMS, **params}
    for key, (lo, hi) in RL_PARAM_BOUNDS.items():
        if key in out:
            out[key] = float(np.clip(out[key], lo, hi))
    return out


def apply_residual_additive(
    base_params: dict[str, float],
    delta_theta: dict[str, float] | None,
) -> dict[str, float]:
    """Legacy: Theta = Theta_base + Delta (absolute amplitudes)."""
    from utility_model import DEFAULT_KERNEL_PARAMS

    merged = {**DEFAULT_KERNEL_PARAMS, **base_params}
    if delta_theta:
        for key in LEGACY_RESIDUAL_PARAM_KEYS:
            merged[key] = float(merged.get(key, 0.0)) + float(delta_theta.get(key, 0.0))
    return clip_params_rl(merged)


def apply_residual(
    base_params: dict[str, float],
    delta_theta: dict[str, float] | None,
    *,
    legacy: bool | None = None,
) -> dict[str, float]:
    """Compose calibrated Theta_base with a residual action.

    Default (``PARAM_GAUGE='logit_simplex'``): ``delta`` holds logits ``z_*`` for
    amplitude proportions and additive shape deltas.  Legacy additive checkpoints
    are detected automatically when their keys are present.
    """
    use_legacy = is_legacy_delta(delta_theta) if legacy is None else legacy
    if use_legacy:
        return apply_residual_additive(base_params, delta_theta)
    return apply_gauge(base_params, delta_theta, clip_params_rl)


def residual_scales_for_checkpoint(blob: dict[str, Any] | None) -> dict[str, float]:
    if blob is None:
        return dict(DEFAULT_RESIDUAL_SCALES)
    if blob.get("param_gauge") == PARAM_GAUGE:
        scales = blob.get("residual_scales")
        if isinstance(scales, dict):
            return {k: float(scales[k]) for k in RESIDUAL_PARAM_KEYS if k in scales}
        return dict(DEFAULT_RESIDUAL_SCALES)
    if blob.get("param_gauge") == "additive":
        return dict(LEGACY_RESIDUAL_SCALES)
    # Heuristic for old blobs without param_gauge field.
    scales = blob.get("residual_scales") or {}
    if isinstance(scales, dict) and "S_v" in scales:
        return {k: float(scales[k]) for k in LEGACY_RESIDUAL_PARAM_KEYS if k in scales}
    return dict(DEFAULT_RESIDUAL_SCALES)


def load_base_params(
    path: Path | None = None,
    prefer: str = "robust",
) -> dict[str, float]:
    """
    Load Theta_base from a calibration JSON.

    prefer: "robust" (default) | "best" | "nominal"

    Older calibration files without sigma_* get vehicle-scale defaults filled in.
    """
    if prefer == "nominal":
        from utility_model import DEFAULT_BASE_PARAMS

        return dict(DEFAULT_BASE_PARAMS)

    path = Path(path) if path is not None else DEFAULT_CALIBRATION_PATH
    if not path.exists():
        raise FileNotFoundError(f"Calibration file not found: {path}")

    from utility_model import DEFAULT_SIGMA_LAT, DEFAULT_SIGMA_LONG

    payload = json.loads(path.read_text(encoding="utf-8"))
    if prefer == "best" and "best_params" in payload:
        params = payload["best_params"]
    elif "robust_params" in payload:
        params = payload["robust_params"]
    elif "best_params" in payload:
        params = payload["best_params"]
    else:
        params = payload

    out = {str(k): float(v) for k, v in params.items()}
    out.setdefault("sigma_long", DEFAULT_SIGMA_LONG)
    out.setdefault("sigma_lat", DEFAULT_SIGMA_LAT)
    return out
