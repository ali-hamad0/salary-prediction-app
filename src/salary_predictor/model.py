# src/salary_predictor/model.py
#
# Simple functions to load the model from disk and make predictions.
# We use module-level variables so the model is loaded once
# and reused for every request — not reloaded every time.

import logging
from pathlib import Path

import joblib
import pandas as pd

logger = logging.getLogger(__name__)

# ── Paths ─────────────────────────────────────────────────────────────
# __file__ = src/salary_predictor/model.py
# .parents[2] goes up two levels to the project root
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_MODELS_DIR   = _PROJECT_ROOT / "models"

# ── Module-level variables ─────────────────────────────────────────────
# These are empty at first and filled when load_model() is called.
_model    = None
_encoders: dict = {}
_metadata: dict = {}


def load_model() -> None:
    """
    Load the model, encoders, and metadata from disk.

    We use global variables so the loaded model stays in memory
    and is shared across every API request without reloading.
    """
    global _model, _encoders, _metadata

    model_path    = _MODELS_DIR / "salary_model.pkl"
    encoders_path = _MODELS_DIR / "encoders.pkl"
    metadata_path = _MODELS_DIR / "model_metadata.pkl"

    # If any file is missing, this raises FileNotFoundError immediately
    # which stops the server from starting with a clear error message
    _model    = joblib.load(model_path)
    _encoders = joblib.load(encoders_path)
    _metadata = joblib.load(metadata_path)

    logger.info(
        "Loaded: %s | R2=%.4f | MAE=$%.0f",
        _metadata.get("model_type", "unknown"),
        _metadata.get("metrics", {}).get("r2_test", 0),
        _metadata.get("metrics", {}).get("mae", 0),
    )


def is_model_loaded() -> bool:
    """Return True if the model is ready to make predictions."""
    return _model is not None


def get_valid_values(column: str) -> list:
    """
    Return accepted string values for a categorical column.

    Args:
        column: name of the encoded column e.g. 'job_title'

    Returns:
        list of valid string labels from the encoder
    """
    if column not in _encoders:
        return []
    return _encoders[column].classes_.tolist()


def predict_salary(inputs: dict) -> float:
    """
    Encode the inputs and return the predicted salary in USD.

    Args:
        inputs: dict with raw string/int values from the request

    Returns:
        predicted salary as a float

    Raises:
        ValueError: if a string value is not in the encoder
    """
    feature_names: list = _metadata["feature_names"]

    # Remove the accidental index column from feature list if it exists.
    # This was saved in the model by mistake during training.
    # The permanent fix is to retrain — this handles it in the meantime.

    row: dict = {}

    for feature in feature_names:
        value = inputs[feature]

        if feature in _encoders:
            encoder      = _encoders[feature]
            valid_values = encoder.classes_.tolist()

            if value not in valid_values:
                raise ValueError(
                    f"'{value}' is not valid for '{feature}'. "
                    f"Accepted values: {valid_values}"
                )

            row[feature] = int(encoder.transform([value])[0])

        else:
            # Numeric columns — pass through as-is
            row[feature] = value

    df = pd.DataFrame([row], columns=feature_names)
    salary: float = float(_model.predict(df)[0])
    return round(salary, 2)