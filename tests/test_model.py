# tests/test_model.py
#
# Unit tests for salary_predictor.model
#
# All tests use mocks — no pkl files needed.
# This means tests run fast and don't break if the model file changes.

from unittest.mock import MagicMock

import numpy as np
import pytest

import salary_predictor.model as model_module


# ── Fixtures ───────────────────────────────────────────────────────────

@pytest.fixture(autouse=True)
def reset_model_state() -> None:
    """
    Reset module-level model state before and after every test.

    Without this, a test that sets _model would bleed into the next test.
    autouse=True means this runs automatically for every test in this file.
    """
    model_module._model    = None
    model_module._encoders = {}
    model_module._metadata = {}
    yield
    model_module._model    = None
    model_module._encoders = {}
    model_module._metadata = {}


def _make_encoder(classes: list) -> MagicMock:
    """Helper: return a mock LabelEncoder with the given classes.

    Uses numpy array for classes_ to match sklearn's real LabelEncoder behavior.
    """
    encoder = MagicMock()
    encoder.classes_               = np.array(classes)
    encoder.transform.return_value = [0]
    return encoder


# ── is_model_loaded ────────────────────────────────────────────────────

def test_is_model_loaded_returns_false_before_load() -> None:
    """Model is not loaded on startup — is_model_loaded must return False."""
    # Arrange — _model is None (set by fixture)

    # Act
    result = model_module.is_model_loaded()

    # Assert
    assert result is False


def test_is_model_loaded_returns_true_after_mock_load() -> None:
    """After assigning a mock model, is_model_loaded must return True."""
    # Arrange
    model_module._model = MagicMock()

    # Act
    result = model_module.is_model_loaded()

    # Assert
    assert result is True


# ── get_valid_values ───────────────────────────────────────────────────

def test_get_valid_values_returns_empty_list_when_encoders_not_loaded() -> None:
    """Before load, any column query returns an empty list."""
    # Arrange — _encoders is {} (set by fixture)

    # Act
    result = model_module.get_valid_values("job_title")

    # Assert
    assert result == []


def test_get_valid_values_returns_correct_classes() -> None:
    """Returns the classes list from the encoder for the requested column."""
    # Arrange
    classes = ["Data Analyst", "Data Engineer", "Data Scientist"]
    model_module._encoders = {"job_title": _make_encoder(classes)}

    # Act
    result = model_module.get_valid_values("job_title")

    # Assert
    assert result == classes


def test_get_valid_values_returns_empty_for_unknown_column() -> None:
    """A column not in the encoders dict returns an empty list."""
    # Arrange
    model_module._encoders = {"job_title": _make_encoder(["Data Scientist"])}

    # Act
    result = model_module.get_valid_values("nonexistent_column")

    # Assert
    assert result == []


# ── predict_salary ─────────────────────────────────────────────────────

def test_predict_salary_returns_float_for_valid_inputs() -> None:
    """Valid inputs go through the encoder and return a rounded float."""
    # Arrange
    model_module._model    = MagicMock()
    model_module._model.predict.return_value = [130500.75]
    model_module._encoders = {"job_title": _make_encoder(["Data Scientist"])}
    model_module._metadata = {"feature_names": ["job_title"]}

    inputs = {"job_title": "Data Scientist"}

    # Act
    result = model_module.predict_salary(inputs)

    # Assert
    assert isinstance(result, float)
    assert result == 130500.75


def test_predict_salary_raises_value_error_for_invalid_job_title() -> None:
    """An unrecognised job title raises ValueError with a clear message."""
    # Arrange
    model_module._model    = MagicMock()
    model_module._encoders = {"job_title": _make_encoder(["Data Scientist"])}
    model_module._metadata = {"feature_names": ["job_title"]}

    inputs = {"job_title": "Pizza Chef"}

    # Act & Assert
    with pytest.raises(ValueError, match="not valid"):
        model_module.predict_salary(inputs)


def test_predict_salary_passes_numeric_columns_through() -> None:
    """Numeric columns (no encoder) are passed as-is to the model."""
    # Arrange
    mock_model = MagicMock()
    mock_model.predict.return_value = [95000.0]

    model_module._model    = mock_model
    model_module._encoders = {}          # no encoders — all columns are numeric
    model_module._metadata = {"feature_names": ["remote_ratio", "work_year"]}

    inputs = {"remote_ratio": 100, "work_year": 2023}

    # Act
    result = model_module.predict_salary(inputs)

    # Assert
    assert result == 95000.0
    # Verify the model received a DataFrame with the right columns
    call_args = mock_model.predict.call_args[0][0]
    assert list(call_args.columns) == ["remote_ratio", "work_year"]
