"""Roundabout sparse + ablations without CTG++ (no require-matched-protocol)."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable
CASE = ROOT / "Roundabout Case"
RUN = CASE / "runs" / "paper_2day"
CKPT = RUN / "checkpoints"
OUT = RUN / "results"
LOG_DIR = RUN / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)
OUT.mkdir(parents=True, exist_ok=True)

MODELS = [
    "orca",
    "social_force",
    "dwa",
    "mppi",
    "frenet",
    "utility_pt",
    "direct_discrete_rl",
    "mappo",
    "residual_marl",
    "ctrl_sim",
]


def run(name: str, module_args: list[str]) -> None:
    cmd = [PY, "-u", str(CASE / "run_module.py"), *module_args]
    print("START", name, flush=True)
    out = LOG_DIR / f"{name}_no_ctg.out.log"
    err = LOG_DIR / f"{name}_no_ctg.err.log"
    with out.open("w", encoding="utf-8") as out_f, err.open("w", encoding="utf-8") as err_f:
        code = subprocess.call(cmd, cwd=str(ROOT), stdout=out_f, stderr=err_f)
    if code:
        raise SystemExit(f"{name} failed ({code}); see {err}")
    print("OK", name, flush=True)


def main() -> None:
    run(
        "benchmark",
        [
            "Baselines.benchmark",
            "--models",
            *MODELS,
            "--seed",
            "810000",
            "--scenarios",
            "20",
            "--num-agents",
            "6",
            "--max-steps",
            "80",
            "--calibration",
            str(ROOT / "Calibration" / "utility_calibration_jounieh.json"),
            "--checkpoint-dir",
            str(CKPT),
            "--residual-checkpoint",
            str(CKPT / "residual_policy.pt"),
            "--train-seeds",
            "0",
            "1",
            "2",
            "--n-boot",
            "5000",
            "--output-dir",
            str(OUT),
            "--reference-model",
            "residual_marl",
            "--conflict-lookahead",
            "full",
            "--no-figures",
        ],
    )
    run(
        "param_stress",
        [
            "Baselines.ablation_stress",
            "--mode",
            "both",
            "--lookahead",
            "both",
            "--models",
            "utility_pt",
            "residual_marl",
            "residual_param",
            "residual_weights_only",
            "residual_sigma_only",
            "direct_discrete_rl",
            "mappo",
            "--scenarios",
            "20",
            "--stress-scenarios",
            "20",
            "--num-agents",
            "6",
            "--stress-agents",
            "8",
            "--max-steps",
            "80",
            "--calibration",
            str(ROOT / "Calibration" / "utility_calibration_jounieh.json"),
            "--checkpoint-dir",
            str(CKPT),
            "--residual-checkpoint",
            str(CKPT / "residual_policy.pt"),
            "--train-seeds",
            "0",
            "1",
            "2",
            "--n-boot",
            "5000",
            "--output-dir",
            str(OUT / "param_lookahead_stress"),
            "--no-figures",
        ],
    )
    run(
        "gate_ablation",
        [
            "Baselines.ablation_stress",
            "--mode",
            "gate",
            "--lookahead",
            "full",
            "--scenarios",
            "20",
            "--num-agents",
            "6",
            "--max-steps",
            "80",
            "--calibration",
            str(ROOT / "Calibration" / "utility_calibration_jounieh.json"),
            "--checkpoint-dir",
            str(CKPT),
            "--residual-checkpoint",
            str(CKPT / "residual_policy.pt"),
            "--train-seeds",
            "0",
            "1",
            "2",
            "--n-boot",
            "5000",
            "--output-dir",
            str(OUT / "gate_ablation"),
            "--no-figures",
        ],
    )
    print("Roundabout no-CTG pipeline complete", flush=True)


if __name__ == "__main__":
    main()
