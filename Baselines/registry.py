"""Name -> controller factory, so the benchmark can be driven from the CLI."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

import Baselines._paths  # noqa: F401
from Baselines.controllers import Controller


def _utility_pt(**kwargs: Any) -> Controller:
    from Baselines.utility_prior import UtilityPriorController

    return UtilityPriorController(**kwargs)


def _utility_pt_logit(**kwargs: Any) -> Controller:
    from Baselines.utility_prior import UtilityPriorController

    kwargs.setdefault("temperature", 1.0)
    kwargs.setdefault("name", "utility_pt_logit")
    return UtilityPriorController(**kwargs)


def _residual_marl(**kwargs: Any) -> Controller:
    from Baselines.residual_marl import ResidualMARLController

    return ResidualMARLController(**kwargs)


# Only the candidate residual uses the main checkpoint override.
RESIDUAL_INFERENCE_VARIANTS = frozenset({"residual_marl"})


def _residual_param(**kwargs: Any) -> Controller:
    from Baselines.residual_marl import ResidualMARLController
    kwargs.setdefault("checkpoint", Path("RL/checkpoints/v3/residual_param_policy.pt"))
    kwargs.setdefault("name", "residual_param")
    return ResidualMARLController(**kwargs)


from RL.param_gauge import AMPLITUDE_LOGIT_KEYS, SHAPE_RESIDUAL_KEYS

# Shape coordinates except collision-kernel scales (for sigma-only ablation).
_SHAPE_EXCEPT_SIGMA = ("xi_i", "gamma")


def _utility_nominal(**kwargs: Any) -> Controller:
    from Baselines.utility_prior import UtilityPriorController

    kwargs.setdefault("prefer", "nominal")
    kwargs.setdefault("name", "utility_nominal")
    return UtilityPriorController(**kwargs)


def _residual_sigma_frozen(**kwargs: Any) -> Controller:
    """Legacy name: freeze collision-kernel shape residuals (sigma only at base)."""
    from Baselines.residual_marl import ResidualMARLController

    kwargs.setdefault("checkpoint", Path("RL/checkpoints/v3/residual_param_policy.pt"))
    kwargs.setdefault("freeze_keys", ("sigma_long", "sigma_lat"))
    kwargs.setdefault("name", "residual_sigma_frozen")
    return ResidualMARLController(**kwargs)


def _residual_weights_only(**kwargs: Any) -> Controller:
    """Freeze collision-kernel σ residuals; only amplitude logits + other shape adapt.

    Matches the paper's "weights-only / Δσ frozen" ablation naming.
    """
    from Baselines.residual_marl import ResidualMARLController

    kwargs.setdefault("checkpoint", Path("RL/checkpoints/v3/residual_param_policy.pt"))
    kwargs.setdefault("freeze_keys", ("sigma_long", "sigma_lat"))
    kwargs.setdefault("name", "residual_weights_only")
    return ResidualMARLController(**kwargs)


def _residual_sigma_only(**kwargs: Any) -> Controller:
    """Freeze amplitude logits and non-sigma shape; only sigma_* may adapt."""
    from Baselines.residual_marl import ResidualMARLController

    kwargs.setdefault("checkpoint", Path("RL/checkpoints/v3/residual_param_policy.pt"))
    kwargs.setdefault("freeze_keys", AMPLITUDE_LOGIT_KEYS + _SHAPE_EXCEPT_SIGMA)
    kwargs.setdefault("name", "residual_sigma_only")
    return ResidualMARLController(**kwargs)


def _residual_nominal(**kwargs: Any) -> Controller:
    """Residual on the uncalibrated nominal prior (tests calibration importance)."""
    from Baselines.residual_marl import ResidualMARLController

    kwargs.setdefault("checkpoint", Path("RL/checkpoints/v3/residual_nominal_policy.pt"))
    kwargs.setdefault("prefer", "nominal")
    kwargs.setdefault("name", "residual_nominal")
    return ResidualMARLController(**kwargs)


def _residual_collpen(**kwargs: Any) -> Controller:
    """Residual trained with an OBB collision penalty (sparse training by default)."""
    from Baselines.residual_marl import ResidualMARLController

    kwargs.setdefault("checkpoint", Path("RL/checkpoints/v3/residual_collpen_policy.pt"))
    kwargs.setdefault("name", "residual_collpen")
    return ResidualMARLController(**kwargs)


def _residual_collpen_dense(**kwargs: Any) -> Controller:
    """Collision-penalty residual trained under dense spawn (stress distribution)."""
    from Baselines.residual_marl import ResidualMARLController

    kwargs.setdefault("checkpoint", Path("RL/checkpoints/v3/residual_collpen_dense_policy.pt"))
    kwargs.setdefault("name", "residual_collpen_dense")
    return ResidualMARLController(**kwargs)


def _orca(**kwargs: Any) -> Controller:
    from Baselines.orca import ORCAController

    return ORCAController(**kwargs)


def _social_force(**kwargs: Any) -> Controller:
    from Baselines.social_force import SocialForceController

    return SocialForceController(**kwargs)


def _dwa(**kwargs: Any) -> Controller:
    from Baselines.dwa import DWAController

    return DWAController(**kwargs)


def _mppi(**kwargs: Any) -> Controller:
    from Baselines.mppi import MPPIController

    return MPPIController(**kwargs)


def _frenet(**kwargs: Any) -> Controller:
    from Baselines.frenet_planner import FrenetPlannerController

    return FrenetPlannerController(**kwargs)


def _direct_discrete_rl(**kwargs: Any) -> Controller:
    from Baselines.direct_discrete_rl import DirectDiscreteRLController

    return DirectDiscreteRLController(**kwargs)


def _pure_rl(**kwargs: Any) -> Controller:
    from Baselines.pure_rl import PureRLController

    return PureRLController(**kwargs)


def _pure_rl_safe(**kwargs: Any) -> Controller:
    """Pure RL trained with an explicit collision penalty added to the shared reward."""
    from Baselines.pure_rl import PureRLController

    kwargs.setdefault("checkpoint", Path("Baselines/checkpoints/v3/pure_rl_safe_policy.pt"))
    kwargs.setdefault("name", "pure_rl_safe")
    return PureRLController(**kwargs)


def _marl(algo: str) -> Callable[..., Controller]:
    def factory(**kwargs: Any) -> Controller:
        from Baselines.marl import MARLController

        kwargs.setdefault("algo", algo)
        kwargs.setdefault("name", algo)
        return MARLController(**kwargs)

    return factory


REGISTRY: dict[str, Callable[..., Controller]] = {
    "utility_pt": _utility_pt,
    "utility_pt_logit": _utility_pt_logit,
    "utility_nominal": _utility_nominal,
    "residual_marl": _residual_marl,
    "residual_param": _residual_param,
    "residual_sigma_frozen": _residual_sigma_frozen,
    "residual_weights_only": _residual_weights_only,
    "residual_sigma_only": _residual_sigma_only,
    "residual_nominal": _residual_nominal,
    "residual_collpen": _residual_collpen,
    "residual_collpen_dense": _residual_collpen_dense,
    "orca": _orca,
    "social_force": _social_force,
    "dwa": _dwa,
    "mppi": _mppi,
    "frenet": _frenet,
    "direct_discrete_rl": _direct_discrete_rl,
    "pure_rl": _pure_rl,
    "pure_rl_safe": _pure_rl_safe,
    "mappo": _marl("mappo"),
    "happo": _marl("happo"),
    "hatrpo": _marl("hatrpo"),
}

DEFAULT_MODELS = [
    "orca",
    "social_force",
    "dwa",
    "mppi",
    "frenet",
    "direct_discrete_rl",
    "pure_rl",
    "mappo",
    "happo",
    "hatrpo",
    "utility_pt",
    "residual_marl",
]

# Human-readable labels for tables and figures.
LABELS = {
    "utility_pt": "Utility prior (PT)",
    "utility_nominal": "Utility prior (nominal)",
    "utility_pt_logit": "Utility prior (logit choice)",
    "residual_marl": "Residual MARL (ours)",
    "residual_param": "Parameter residual",
    "residual_sigma_frozen": "Residual (σ frozen, legacy)",
    "residual_weights_only": "Residual (weights only)",
    "residual_sigma_only": "Residual (sigma only)",
    "residual_nominal": "Residual + nominal prior",
    "residual_collpen": "Residual + coll. penalty",
    "residual_collpen_dense": "Residual + coll. pen. (dense)",
    "orca": "ORCA",
    "social_force": "Social force (SDP)",
    "dwa": "DWA",
    "mppi": "MPPI",
    "frenet": "Frenet planner",
    "direct_discrete_rl": "Direct discrete RL (matched)",
    "pure_rl": "IPPO continuous (legacy)",
    "pure_rl_safe": "IPPO + collision penalty",
    "mappo": "MAPPO",
    "happo": "HAPPO",
    "hatrpo": "HATRPO",
}


# Models whose behavior depends on a training seed, with the checkpoint that a
# single-seed run writes. Seeded runs append "_seed<k>" to the stem.
LEARNED_CHECKPOINTS: dict[str, Path] = {
    "residual_marl": Path("RL/checkpoints/v3/residual_policy.pt"),
    "residual_sigma_frozen": Path("RL/checkpoints/v3/residual_param_policy.pt"),
    "residual_weights_only": Path("RL/checkpoints/v3/residual_param_policy.pt"),
    "residual_sigma_only": Path("RL/checkpoints/v3/residual_param_policy.pt"),
    "residual_nominal": Path("RL/checkpoints/v3/residual_nominal_policy.pt"),
    "residual_param": Path("RL/checkpoints/v3/residual_param_policy.pt"),
    "residual_collpen": Path("RL/checkpoints/v3/residual_collpen_policy.pt"),
    "residual_collpen_dense": Path("RL/checkpoints/v3/residual_collpen_dense_policy.pt"),
    "pure_rl": Path("Baselines/checkpoints/v3/pure_rl_policy.pt"),
    "direct_discrete_rl": Path("Baselines/checkpoints/v3/direct_discrete_policy.pt"),
    "pure_rl_safe": Path("Baselines/checkpoints/v3/pure_rl_safe_policy.pt"),
    "mappo": Path("Baselines/checkpoints/v3/mappo_policy.pt"),
    "happo": Path("Baselines/checkpoints/v3/happo_policy.pt"),
    "hatrpo": Path("Baselines/checkpoints/v3/hatrpo_policy.pt"),
}


def is_learned(name: str) -> bool:
    return name in LEARNED_CHECKPOINTS


def seed_checkpoint(name: str, train_seed: int, base: Path | None = None) -> Path | None:
    """Path a training run with ``train_seed`` writes for this model."""
    root = base if base is not None else LEARNED_CHECKPOINTS.get(name)
    if root is None:
        return None
    return root.with_name(f"{root.stem}_seed{int(train_seed)}{root.suffix}")


def resolve_train_seeds(
    name: str,
    train_seeds: list[int] | None,
    base: Path | None = None,
) -> list[tuple[int, Path | None]]:
    """Require every requested seed; never substitute or silently drop runs."""
    if not train_seeds or not is_learned(name):
        return [(-1, base)]
    pairs, missing = [], []
    for seed in dict.fromkeys(train_seeds):
        path = seed_checkpoint(name, seed, base)
        if path is None or not path.exists():
            missing.append(str(path))
        else:
            pairs.append((int(seed), path))
    if missing:
        raise FileNotFoundError(f"{name}: missing requested seed checkpoints: {', '.join(missing)}")
    return pairs


def build_controller(name: str, **kwargs: Any) -> Controller:
    if name not in REGISTRY:
        raise KeyError(f"Unknown model '{name}'. Available: {sorted(REGISTRY)}")
    return REGISTRY[name](**kwargs)


def controller_kwargs(
    name: str,
    residual_checkpoint: Path | None = None,
    pure_rl_checkpoint: Path | None = None,
    calibration: Path | None = None,
    checkpoint_dir: Path | None = None,
    checkpoint_override: Path | None = None,
) -> dict[str, Any]:
    """CLI-level wiring of checkpoints and calibration files."""
    if checkpoint_override is not None and is_learned(name):
        kwargs: dict[str, Any] = {"checkpoint": checkpoint_override}
        if name.startswith("residual"):
            kwargs["calibration"] = calibration
        return kwargs
    if name in RESIDUAL_INFERENCE_VARIANTS:
        kwargs: dict[str, Any] = {"calibration": calibration}
        if residual_checkpoint is not None:
            kwargs["checkpoint"] = residual_checkpoint
        return kwargs
    if name.startswith("residual"):
        return {"calibration": calibration, "checkpoint": LEARNED_CHECKPOINTS[name]}
    if name in {"utility_pt", "utility_pt_logit"}:
        return {"calibration": calibration}
    if name == "pure_rl":
        return {"checkpoint": pure_rl_checkpoint} if pure_rl_checkpoint is not None else {}
    if name == "direct_discrete_rl":
        return {}
    if name in {"mappo", "happo", "hatrpo"} and checkpoint_dir is not None:
        return {"checkpoint": checkpoint_dir / f"{name}_policy.pt"}
    return {}
