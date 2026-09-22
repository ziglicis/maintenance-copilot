"""Every tunable number in the system. No hidden constants elsewhere."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
ARTIFACT_DIR = ROOT / "artifacts"
MODEL_DIR = ARTIFACT_DIR / "models"
PREDICTIONS = ARTIFACT_DIR / "predictions.parquet"
SIMULATION = ARTIFACT_DIR / "simulation.parquet"
METRICS = ARTIFACT_DIR / "metrics.json"
TELEMETRY_DB = ARTIFACT_DIR / "telemetry.db"

DATA_URL = "https://phm-datasets.s3.amazonaws.com/NASA/6.+Turbofan+Engine+Degradation+Simulation+Data+Set.zip"
DATASET = "FD001"

SEED = 42
RUL_CAP = 125  # piecewise-linear label cap. Degradation is not observable before this.
VAL_ENGINES = 20  # held out from the 100 training engines, by unit id
QUANTILES = (0.1, 0.5, 0.9)
ROLLING_WINDOWS = (5, 20)
SLOPE_WINDOW = 20

# Risk tiers in cycles of predicted RUL.
TIER_RED = 30
TIER_AMBER = 60
LEAD_TARGET_CYCLES = 20  # a warning counts as early only this many cycles before failure

# LightGBM hyperparameters. Objective and alpha are set per model in models.train_gbm.
GBM_PARAMS = {
    "learning_rate": 0.05,
    "num_leaves": 31,
    "min_data_in_leaf": 40,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "seed": SEED,
    "verbosity": -1,
}
GBM_MAX_ROUNDS = 800
GBM_EARLY_STOPPING = 50

# Policy simulator assumptions, all editable in the UI.
COST_UNSCHEDULED_FAILURE = 750_000.0
COST_SCHEDULED_MAINTENANCE = 120_000.0
DOWNTIME_DAYS_UNSCHEDULED = 14.0
DOWNTIME_DAYS_SCHEDULED = 3.0
COST_PER_WASTED_CYCLE = 400.0
FIXED_INTERVAL_CYCLES = 150
PREDICTIVE_THRESHOLD = 30  # maintain when the lower bound of predicted RUL drops below this
PARTS_LEAD_CYCLES = 15  # cycles between deciding to pull an engine and the slot being ready

# Work order generator.
LLM_MODEL = "claude-opus-5"
LLM_MODELS = ("claude-opus-5", "claude-sonnet-5", "claude-haiku-4-5")
LLM_PRICING = {  # USD per million tokens, (input, output)
    "claude-opus-5": (5.0, 25.0),
    "claude-sonnet-5": (2.0, 10.0),
    "claude-haiku-4-5": (1.0, 5.0),
}
LLM_EFFORT = "medium"  # thinking depth and overall token spend
LLM_EFFORTS = ("low", "medium", "high")
# Effort is rejected by models that do not support it, Haiku 4.5 among them.
LLM_EFFORT_MODELS = frozenset({"claude-opus-5", "claude-sonnet-5"})
LLM_MAX_TOKENS = 4000
SENSOR_TOLERANCE = 0.05  # groundedness check, absolute tolerance on cited sensor values
RANGE_TOLERANCE = 0.1  # tolerance on a cited healthy range, which the prompt supplies at 4dp
TREND_WINDOW = 20  # cycles used to decide whether a sensor is rising, falling or stable
TREND_SIGMA = 2.0  # how many standard errors of movement count as a trend rather than noise
LOG_ENTRIES_PER_ENGINE = 5

# Fleet scale used for the projected monthly cost in the Operations tab.
FLEET_SIZE = 100
REFRESHES_PER_MONTH = 30
CYCLES_PER_ENGINE_PER_YEAR = 300  # flights per engine per year, used to annualise simulated savings
