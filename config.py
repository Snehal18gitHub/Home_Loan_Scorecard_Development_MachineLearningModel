from pathlib import Path


# ============================================================
# CONFIGURATION
# ============================================================

RANDOM_STATE = 42

TARGET_COLUMN = "TARGET"


# ============================================================
# PROJECT PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

DATA_DIR = BASE_DIR / "data"

DATA_PATH = DATA_DIR / "application_train.csv"


# ============================================================
# MODEL / OUTPUT PATHS
# ============================================================

MODEL_DIR = BASE_DIR / "models"

OUTPUT_DIR = BASE_DIR / "outputs"


# ============================================================
# SHAP CONFIGURATION
# ============================================================

SHAP_SAMPLE_SIZE = 5000