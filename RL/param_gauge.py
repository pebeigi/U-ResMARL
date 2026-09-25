"""Logit-simplex gauge for utility amplitude weights used by residual MARL.

Amplitude weights are W * softmax(z) with W fixed from the calibrated prior.
The residual edits logits z and the shape coordinates; absolute scale stays
tied to calibration. Older additive-delta checkpoints still load when needed.
"""

from __future__ import annotations

from typing import Any

import numpy as np

PARAM_GAUGE = "logit_simplex"

# Physical amplitude coordinates used inside utility_model.evaluate_candidate_utility.
AMPLITUDE_KEYS = ("S_theta", "S_v", "S_d", "w_c", "w_ell")

# Residual policy outputs (5 logits + 4 shape deltas).
AMPLITUDE_LOGIT_KEYS = ("z_theta", "z_v", "z_d", "z_c", "z_l")
SHAPE_RESIDUAL_KEYS = ("xi_i", "gamma", "sigma_long", "sigma_lat")
RESIDUAL_PARAM_KEYS = AMPLITUDE_LOGIT_KEYS + SHAPE_RESIDUAL_KEYS

# Pre-gauge additive parameterization (old checkpoints).
LEGACY_RESIDUAL_PARAM_KEYS = (
    "S_v",
    "S_theta",
    "S_d",
    "w_c",
    "xi_i",
    "gamma",
    "w_ell",
    "sigma_long",
    "sigma_lat",
)

DEFAULT_RESIDUAL_SCALES: dict[str, float] = {
    "z_theta": 0.75,
    "z_v": 0.75,
    "z_d": 0.75,
    "z_c": 0.75,
    "z_l": 0.75,
    "xi_i": 1.0,
    "gamma": 1.0,
    "sigma_long": 1.5,
    "sigma_lat": 0.6,
}

LEGACY_RESIDUAL_SCALES: dict[str, float] = {
    "S_v": 1.5,
    "S_theta": 1.5,
    "S_d": 2.0,
    "w_c": 250.0,
    "xi_i": 1.0,
    "gamma": 1.0,
    "w_ell": 15.0,
    "sigma_long": 1.5,
    "sigma_lat": 0.6,
}


def is_legacy_delta(delta: dict[str, float] | None) -> bool:
    if not delta:
        return False
    keys = set(delta.keys())
    if keys & set(AMPLITUDE_LOGIT_KEYS):
        return False
    return bool(keys & set(LEGACY_RESIDUAL_PARAM_KEYS))


def amplitude_scale_W(base_params: dict[str, float]) -> float:
    """Total amplitude mass W = sum_k amplitude_k at the calibrated prior."""
    return float(sum(max(float(base_params.get(k, 0.0)), 1e-8) for k in AMPLITUDE_KEYS))


def amplitudes_to_logits(base_params: dict[str, float]) -> np.ndarray:
    """Map calibrated amplitudes to logits z with softmax(z) = proportions."""
    amps = np.array([max(float(base_params.get(k, 1e-8)), 1e-8) for k in AMPLITUDE_KEYS], dtype=float)
    props = amps / amps.sum()
    return np.log(props)


def logits_to_amplitudes(z: np.ndarray, W: float) -> dict[str, float]:
    z = np.asarray(z, dtype=float).reshape(-1)
    z = z - z.max()
    props = np.exp(z)
    props = props / props.sum()
    return {key: float(W * p) for key, p in zip(AMPLITUDE_KEYS, props)}


def base_logits(base_params: dict[str, float]) -> np.ndarray:
    return amplitudes_to_logits(base_params)


def residual_vector_to_dict(residual: Any, legacy: bool = False) -> dict[str, float]:
    keys = LEGACY_RESIDUAL_PARAM_KEYS if legacy else RESIDUAL_PARAM_KEYS
    arr = np.asarray(residual, dtype=float).reshape(-1)
    if arr.size != len(keys):
        raise ValueError(f"Expected {len(keys)} residuals, got {arr.size}")
    return dict(zip(keys, arr.astype(float)))


def merge_delta_logits(
    base_params: dict[str, float],
    delta: dict[str, float] | None,
) -> np.ndarray:
    z = base_logits(base_params)
    if not delta:
        return z
    # The softmax is invariant to a common shift of all logits, so the residual is
    # centred to drop that unidentifiable direction from the action space.
    dz = np.array([float(delta.get(key, 0.0)) for key in AMPLITUDE_LOGIT_KEYS], dtype=float)
    z += dz - dz.mean()
    return z


def apply_gauge(
    base_params: dict[str, float],
    delta: dict[str, float] | None,
    clip_fn,
) -> dict[str, float]:
    """Compose Theta_base with logit-simplex amplitude residuals + shape deltas."""
    merged = dict(base_params)
    W = amplitude_scale_W(base_params)
    amps = logits_to_amplitudes(merge_delta_logits(base_params, delta), W)
    merged.update(amps)
    for key in SHAPE_RESIDUAL_KEYS:
        merged[key] = float(base_params.get(key, 0.0))
        if delta:
            merged[key] += float(delta.get(key, 0.0))
    return clip_fn(merged)


def amplitude_proportions(params: dict[str, float]) -> dict[str, float]:
    """Human-readable simplex weights (sum to 1) for logging / paper tables."""
    W = amplitude_scale_W(params)
    if W <= 0:
        return {k: 0.0 for k in AMPLITUDE_KEYS}
    return {k: float(params.get(k, 0.0)) / W for k in AMPLITUDE_KEYS}
