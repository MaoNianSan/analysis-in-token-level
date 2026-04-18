"""
Global configuration for the AD classification pipeline.
"""

import os
from pathlib import Path
import random
import numpy as np


# =========================
# Runtime seed overrides
# =========================
def _parse_int_env(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None or str(value).strip() == "":
        return default
    try:
        return int(value)
    except ValueError:
        return default


def _parse_int_list_env(name: str, default: list[int]) -> list[int]:
    value = os.getenv(name)
    if value is None or str(value).strip() == "":
        return default

    parsed = []
    for item in str(value).split(","):
        item = item.strip()
        if not item:
            continue
        try:
            parsed.append(int(item))
        except ValueError:
            continue

    return parsed if parsed else default


# =========================
# Reproducibility
# =========================
RANDOM_SEED = _parse_int_env("RANDOM_SEED", 42)
ROBUSTNESS_SEEDS = _parse_int_list_env("ROBUSTNESS_SEEDS", [7, 42, 123, 2026])

# =========================
# Direction / stability thresholds
# =========================
EPSILON = 1e-3
STABILITY_THRESHOLD = 0.9

# =========================
# Data paths
# =========================
DATA_DIR = Path("data_Test AD")
RAW_TRAIN_AD_PATH = DATA_DIR / "ad_s2t_wav2vec.csv"
RAW_TRAIN_CONTROL_PATH = DATA_DIR / "control_s2t_wav2vec.csv"
RAW_TEST_PATH = DATA_DIR / "test_s2t_wav2vec.csv"

# =========================
# Analysis parameters
# =========================
WINDOW_LENGTH = 4
STRIDE = 2
TOP_K = 5
N_SAMPLES = 10
NOISE_SCALE = 0.003
ATTRIBUTION_METHOD = "leave_one_out"


# =========================
# Script-specific output directories
# =========================
def build_script_dirs(script_name: str) -> dict[str, str]:
    """Build standard script directory paths without creating folders."""
    base_dir = DATA_DIR / script_name
    return {
        "base": str(base_dir),
        "csv": str(base_dir / "csv"),
        "figures": str(base_dir / "figures"),
        "logs": str(base_dir / "logs"),
        "model": str(base_dir / "model"),
        "metrics": str(base_dir / "metrics"),
        "predictions": str(base_dir / "predictions"),
    }


def build_dataset_dirs(script_name: str, dataset_name: str) -> dict[str, str]:
    """Build standard dataset-level directory paths without creating folders."""
    base_dir = DATA_DIR / script_name / dataset_name
    return {
        "base": str(base_dir),
        "csv": str(base_dir / "csv"),
        "figures": str(base_dir / "figures"),
        "logs": str(base_dir / "logs"),
        "model": str(base_dir / "model"),
        "metrics": str(base_dir / "metrics"),
        "predictions": str(base_dir / "predictions"),
    }


def ensure_dirs(paths: dict[str, str], keys: list[str] | tuple[str, ...] | None = None) -> None:
    """
    Lazily create only requested folders.

    If keys is None, all entries in paths are created.
    """
    selected_keys = keys if keys is not None else list(paths.keys())
    for key in selected_keys:
        if key not in paths:
            raise KeyError(f"Unknown path key: {key}")
        Path(paths[key]).mkdir(parents=True, exist_ok=True)


OUTPUT_FILES = {
    "classification_results": "classification_results.csv",
    "window_importance": "window_importance.csv",
    "token_attribution_all": "token_attribution_all.csv",
    "token_attribution_aggregated": "token_attribution_aggregated.csv",
    "debugging_stability_summary": "debugging_stability_summary.csv",
    "train_classifier_output": "train_classifier_output.txt",
    "window_extraction_output": "window_extraction_output.txt",
}

# =========================
# Script-specific output paths
# =========================
# Preprocess data outputs
PREPROCESS_DATA_DIRS = build_script_dirs("preprocess_data")
TRAIN_AD_PATH = str(Path(PREPROCESS_DATA_DIRS["csv"]) / "ad.csv")
TRAIN_CONTROL_PATH = str(Path(PREPROCESS_DATA_DIRS["csv"]) / "control.csv")
TEST_PATH = str(Path(PREPROCESS_DATA_DIRS["csv"]) / "test.csv")

# Unified dataset mapping for downstream analysis
DATASET_PATHS = {
    "test": TEST_PATH,
    "ad": TRAIN_AD_PATH,
    "control": TRAIN_CONTROL_PATH,
}

# Train classifier outputs
TRAIN_CLASSIFIER_DIRS = build_script_dirs("train_classifier")
MODEL_DIR = TRAIN_CLASSIFIER_DIRS["model"]
VECTORIZER_PATH = str(Path(TRAIN_CLASSIFIER_DIRS["model"]) / "tfidf_vectorizer.joblib")
MODEL_PATH = str(
    Path(TRAIN_CLASSIFIER_DIRS["model"]) / "logistic_regression_model.joblib"
)
CLASSIFICATION_RESULTS_PATH = str(
    Path(TRAIN_CLASSIFIER_DIRS["predictions"]) / "classification_results.csv"
)
TRAIN_CLASSIFIER_OUTPUT_PATH = str(
    Path(TRAIN_CLASSIFIER_DIRS["logs"]) / "train_classifier_output.txt"
)

# Window extraction outputs (legacy default paths remain for compatibility)
WINDOW_EXTRACTION_DIRS = build_script_dirs("window_extraction")
WINDOW_EXTRACTION_OUTPUT_PATH = str(
    Path(WINDOW_EXTRACTION_DIRS["logs"]) / "window_extraction_output.txt"
)
WINDOW_IMPORTANCE_PATH = str(
    Path(WINDOW_EXTRACTION_DIRS["csv"]) / "window_importance.csv"
)

# Token aggregation outputs
TOKEN_AGGREGATION_DIRS = build_script_dirs("token_aggregation")
TOKEN_ATTRIBUTION_ALL_PATH = str(
    Path(TOKEN_AGGREGATION_DIRS["csv"]) / "token_attribution_all.csv"
)
TOKEN_ATTRIBUTION_AGGREGATED_PATH = str(
    Path(TOKEN_AGGREGATION_DIRS["csv"]) / "token_attribution_aggregated.csv"
)

# Debug stability outputs
DEBUG_STABILITY_DIRS = build_script_dirs("debug_stability")
DEBUGGING_STABILITY_SUMMARY_PATH = str(
    Path(DEBUG_STABILITY_DIRS["csv"]) / "debugging_stability_summary.csv"
)

# Visualization outputs
VISUALIZATION_DIRS = build_script_dirs("visualization")
FIGURE_DIR = VISUALIZATION_DIRS["figures"]

# Main pipeline outputs
MAIN_PIPELINE_DIRS = build_script_dirs("main_pipeline")

LEGACY_OUTPUT_FILES = [
    "window_importance_all.csv",
    "window_importance_top5.csv",
    "token_attribution_top5.csv",
    "debugging_window_stability.csv",
    "debugging_token_stability.csv",
    "debugging_diagnostics.csv",
]


def set_global_seed(seed: int = RANDOM_SEED) -> None:
    """Set process-wide random seed."""
    random.seed(seed)
    np.random.seed(seed)


def compute_direction(delta: float, epsilon: float = EPSILON) -> str:
    """Convert signed delta into direction with a neutral zone around zero."""
    if delta > epsilon:
        return "AD_support"
    if delta < -epsilon:
        return "Control_support"
    return "Neutral"
