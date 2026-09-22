"""Jounieh roundabout closed-loop protocol (isolated from freeway and TGSIM)."""
from pathlib import Path
import json
import os

CASE_ROOT = Path(__file__).resolve().parent
REPO_ROOT = CASE_ROOT.parent

CALIBRATION = REPO_ROOT / "Calibration" / "utility_calibration_jounieh.json"
STREET_BOUNDARIES = REPO_ROOT / "data" / "Lebanon_Jounieh" / "Jounieh_Road_Boundaries.csv"
TRAJECTORIES_CSV = REPO_ROOT / "data" / "Lebanon_Jounieh" / "prepared" / "trajectories_calibration.csv"
RUN_ROOT = Path(os.environ.get("ROUNDABOUT_RUN_DIR", str(CASE_ROOT / "runs" / "paper_2day"))).resolve()
CHECKPOINT_DIR = RUN_ROOT / "checkpoints"
RESULT_DIR = RUN_ROOT / "results"
LOG_DIR = RUN_ROOT / "logs"
NEW_BASELINE_DATA = RUN_ROOT / "data"
NEW_BASELINE_RESULT_DIR = RESULT_DIR / "new_baselines"

# The user-selected 3x Jounieh coordinate frame is 0.13063734 m/pixel.
# Raw source data remain at the supplied scale; the active curb and prepared
# trajectories are both regenerated at 3x. Absolute scale is still provisional.
VEHICLE_LENGTH = 4.5
VEHICLE_WIDTH = 1.8
VEHICLE_WHEELBASE = 2.8
# Keep the execution cap identical to the validated calibration metadata. The
# calibration derives this from the 99th percentile of valid moving ego rows,
# which avoids letting a single differentiated-track outlier set the dynamics.
MAX_AGENT_SPEED = float(json.loads(CALIBRATION.read_text(encoding="utf-8"))["max_agent_speed"])

# Clearances and distances follow the 3x coordinate transform. Arrival retains
# the shared physical 2 m criterion used at the other sites.
ROUTING_CLEARANCE = 0.90
SPAWN_CLEARANCE = 1.65
NUM_AGENTS = 6
MAX_STEPS = 80
MIN_INITIAL_SPACING = 12.0
MIN_DEST_REMAINING = 36.0
MAX_DEST_REMAINING = 135.0
MIN_RECORDED_GOAL_DISTANCE = 15.0
BASE_DESIRED_SPEED = 12.0
PERCEPTION_RADIUS = 36.0
DESTINATION_THRESHOLD = 2.0

SEEDS = [0, 1, 2]
# Same 2-day recipe as Baselines._run_2day_pipeline (24 PPO updates, not a 30k-step week).
TRAIN_UPDATES = 24
VAL_EPISODES = 16
VAL_EVERY = 10
TEST_EPISODES = 16
BENCH_SCENARIOS = 20
N_BOOT = 5000
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
NEW_BASELINE_MODELS = ("ctrl_sim",)
NEW_BASELINE_TRAIN_SCENES = 20
NEW_BASELINE_VAL_SCENES = 6
NEW_BASELINE_STEPS = 10000
