"""Model lists for the paper ablation / stress experimental package (reviewer comment 4)."""

from __future__ import annotations

# Sparse (10-agent) ablation — full ICLR/CoRL minimum package.
PAPER_ABLATION_MODELS: list[str] = [
    "utility_pt",  # calibrated discrete utility controller, no RL
    "utility_nominal",  # uncalibrated / nominal prior, no RL
    "residual_marl",  # full residual (candidate-logit residual by default)
    "residual_param",
    "residual_weights_only",  # freeze Δσ∥, Δσ⊥ (weights-only residual; param mode)
    "residual_sigma_only",  # freeze non-σ residuals; σ only (param mode)
    "direct_discrete_rl",  # matched direct discrete PPO on same grid + OBB filter
    "mappo",  # MAPPO with shared closed-loop OBB safety layer
    "residual_nominal",  # residual on uncalibrated nominal prior
]

# Dense stress subset (same seeds; collpen-dense optional when trained).
STRESS_ABLATION_MODELS: list[str] = [
    "utility_pt",
    "utility_nominal",
    "residual_marl",
    "residual_param",
    "residual_weights_only",
    "residual_sigma_only",
    "residual_collpen_dense",
    "direct_discrete_rl",
    "mappo",
]

# Paired comparisons reported against the full method.
ABLATION_REFERENCE = "residual_marl"

# Gate vs residual-learning isolation (matched discrete grid + OBB filter).
GATE_ABLATION_MODELS: list[str] = [
    "utility_pt",  # calibrated discrete utility, no residual
    "residual_no_gate",  # trained residual, always execute residual argmax
    "residual_marl",  # trained residual + PDM-Closed accept-if-better
    "residual_random_gate",  # untrained residual + the same gate
    "direct_discrete_rl",  # matched discrete PPO, no utility prior
]

KEY_PAIRED_COMPARISONS: list[tuple[str, str]] = [
    ("residual_marl", "utility_pt"),
    ("residual_marl", "residual_no_gate"),
    ("residual_marl", "residual_random_gate"),
    ("residual_marl", "residual_param"),
    ("residual_marl", "utility_nominal"),
    ("residual_marl", "residual_nominal"),
    ("residual_param", "residual_weights_only"),
    ("residual_param", "residual_sigma_only"),
    ("residual_marl", "direct_discrete_rl"),
    ("residual_marl", "mappo"),
]

DEFAULT_TRAIN_SEEDS: list[int] = [0, 1, 2]
DEFAULT_ABLATION_SCENARIOS = 30
DEFAULT_STRESS_SCENARIOS = 30
