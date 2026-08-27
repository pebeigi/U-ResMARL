"""Model lists for the paper ablation / stress experimental package (reviewer comment 4)."""

from __future__ import annotations

# Sparse (10-agent) ablation — full ICLR/CoRL minimum package.
PAPER_ABLATION_MODELS: list[str] = [
    "utility_pt",  # calibrated prior, no RL
    "utility_nominal",  # uncalibrated / nominal prior, no RL
    "residual_marl",  # full residual MARL (current method)
    "residual_weights_only",  # freeze Δσ∥, Δσ⊥ (weights-only residual)
    "residual_sigma_only",  # freeze the seven weight residuals; σ only
    "direct_discrete_rl",  # matched direct discrete PPO on same grid + OBB filter
    "mappo",  # MAPPO with shared closed-loop OBB safety layer
    "residual_nominal",  # residual on uncalibrated nominal prior
]

# Dense stress subset (same seeds; collpen-dense optional when trained).
STRESS_ABLATION_MODELS: list[str] = [
    "utility_pt",
    "utility_nominal",
    "residual_marl",
    "residual_weights_only",
    "residual_sigma_only",
    "residual_collpen_dense",
    "direct_discrete_rl",
    "mappo",
]

# Paired comparisons reported against the full method.
ABLATION_REFERENCE = "residual_marl"

KEY_PAIRED_COMPARISONS: list[tuple[str, str]] = [
    ("residual_marl", "utility_pt"),
    ("residual_marl", "utility_nominal"),
    ("residual_marl", "residual_nominal"),
    ("residual_marl", "residual_weights_only"),
    ("residual_marl", "residual_sigma_only"),
    ("residual_marl", "direct_discrete_rl"),
    ("residual_marl", "mappo"),
]

DEFAULT_TRAIN_SEEDS: list[int] = [0, 1, 2, 3, 4]
DEFAULT_ABLATION_SCENARIOS = 30
DEFAULT_STRESS_SCENARIOS = 30
