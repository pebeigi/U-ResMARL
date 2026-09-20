"""Jounieh roundabout closed-loop protocol (isolated from freeway and TGSIM)."""
from pathlib import Path
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

# Retain the supplied 0.04354578 m/pixel frame until a ground distance verifies
# its absolute scale. At that scale, recorded image-axis boxes are compatible
# with roughly 1.74 x 0.63 oriented boxes (see tools/audit_roundabout_scale.py).
# The existing calibration used 4.5 x 1.8: it is a legacy prior, not a refit for
# these footprints. These smaller model dimensions remain provisional.
VEHICLE_LENGTH = 1.6
VEHICLE_WIDTH = 0.64
VEHICLE_WHEELBASE = 0.99
MAX_AGENT_SPEED = 14.813246460520965

# The curb is a ~56 m x 31 m ring with two islands. Median on-road clearance
# is <1 m, so TGSIM's 3.5 m spawn/route buffers would empty the polygon.
ROUTING_CLEARANCE = 0.30
SPAWN_CLEARANCE = 0.55
NUM_AGENTS = 6
MAX_STEPS = 80
MIN_INITIAL_SPACING = 4.0
MIN_DEST_REMAINING = 12.0
MAX_DEST_REMAINING = 45.0
BASE_DESIRED_SPEED = 4.0
PERCEPTION_RADIUS = 12.0
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
