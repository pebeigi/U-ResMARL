"""TGSIM Foggy Bottom closed-loop protocol (isolated from the freeway case)."""
from pathlib import Path
import os

CASE_ROOT = Path(__file__).resolve().parent
REPO_ROOT = CASE_ROOT.parent

CALIBRATION = REPO_ROOT / "Calibration" / "utility_calibration_tgsim.json"
STREET_BOUNDARIES = (
    REPO_ROOT / "data" / "TGSIM FB" / "derived_boundaries" / "street_boundaries.csv"
)
TRAJECTORIES_CSV = REPO_ROOT / "data" / "TGSIM FB" / "prepared" / "trajectories_calibration.csv"
RUN_ROOT = Path(os.environ.get('TGSIM_RUN_DIR', str(CASE_ROOT/'runs'/'paper_2day'))).resolve()
CHECKPOINT_DIR = RUN_ROOT / "checkpoints"
RESULT_DIR = RUN_ROOT / "results"
LOG_DIR = RUN_ROOT / "logs"
NEW_BASELINE_DATA = RUN_ROOT / "data"
NEW_BASELINE_RESULT_DIR = RESULT_DIR / "new_baselines"

# Network curb geometry from TGSIM calibration. Lanes / PCA tubes are unused.
NUM_AGENTS = 6
MAX_STEPS = 80
MIN_INITIAL_SPACING = 8.0
MIN_DEST_REMAINING = 40.0
MAX_DEST_REMAINING = 140.0
BASE_DESIRED_SPEED = 8.0
PERCEPTION_RADIUS = 20.0
DESTINATION_THRESHOLD = 2.0

SEEDS = [0, 1, 2]
# Same 2-day recipe as Baselines._run_2day_pipeline (24 PPO updates, not a 30k-step week).
TRAIN_UPDATES = 24
VAL_EPISODES = 16
VAL_EVERY = 10
TEST_EPISODES = 16
BENCH_SCENARIOS = 20
N_BOOT = 5000
# Site scene size stays TGSIM-specific; density bump is the analogue of freeway 10→16.
STRESS_AGENTS = 8
TRAIN_MODELS = (
    "residual_marl",
    "residual_param",
    "direct_discrete_rl",
    "mappo",
)
BENCH_MODELS = (
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
    "ctg_plus_plus",
)
PARAM_EVAL_MODELS = (
    "utility_pt",
    "residual_marl",
    "residual_param",
    "residual_weights_only",
    "residual_sigma_only",
    "direct_discrete_rl",
    "mappo",
)
NEW_BASELINE_MODELS = ("ctrl_sim", "ctg_plus_plus")
NEW_BASELINE_TRAIN_SCENES = 20
NEW_BASELINE_VAL_SCENES = 6
NEW_BASELINE_STEPS = 10000
