# tests/test_api.py
#
# Integration tests for the FastAPI prediction endpoints.
#
# Uses FastAPI's TestClient (no real HTTP server needed).
# The model is mocked in every test — no pkl files required.

from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from salary_predictor.api.main import app

client = TestClient(app, raise_server_exceptions=True)

# A complete set of valid query parameters for /predict
VALID_PARAMS: dict = {
    "experience_level":   "SE",
    "employment_type":    "FT",
    "company_size":       "L",
    "remote_ratio":       100,
    "work_year":          2023,
    "job_title":          "Data Scientist",
    "employee_residence": "US",
    "company_location":   "US",
}


# ── /health ────────────────────────────────────────────────────────────

def test_health_returns_ok_when_model_is_loaded() -> None:
    """Health endpoint returns status=ok when the model is ready."""
    # Arrange
    with patch("salary_predictor.api.main.is_model_loaded", return_value=True):

        # Act
        response = client.get("/health")

    # Assert
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["model_loaded"] is True


def test_health_returns_model_not_loaded_when_model_is_missing() -> None:
    """Health endpoint reports model_loaded=False when startup failed."""
    # Arrange
    with patch("salary_predictor.api.main.is_model_loaded", return_value=False):

        # Act
        response = client.get("/health")

    # Assert
    assert response.status_code == 200
    data = response.json()
    assert data["model_loaded"] is False
    assert data["status"] != "ok"


# ── /predict ───────────────────────────────────────────────────────────

def test_predict_returns_salary_for_valid_inputs() -> None:
    """Valid inputs return 200 with a predicted_salary_usd field."""
    # Arrange
    with patch("salary_predictor.api.predict.is_model_loaded", return_value=True), \
         patch("salary_predictor.api.predict.predict_salary", return_value=142000.0):

        # Act
        response = client.get("/predict", params=VALID_PARAMS)

    # Assert
    assert response.status_code == 200
    data = response.json()
    assert "predicted_salary_usd" in data
    assert data["predicted_salary_usd"] == 142000.0


def test_predict_response_echoes_inputs() -> None:
    """The response includes the original inputs for traceability."""
    # Arrange
    with patch("salary_predictor.api.predict.is_model_loaded", return_value=True), \
         patch("salary_predictor.api.predict.predict_salary", return_value=100000.0):

        # Act
        response = client.get("/predict", params=VALID_PARAMS)

    # Assert
    data = response.json()
    assert data["inputs"]["job_title"] == "Data Scientist"
    assert data["inputs"]["experience_level"] == "SE"


def test_predict_returns_503_when_model_not_loaded() -> None:
    """If the model failed to load at startup, /predict returns 503."""
    # Arrange
    with patch("salary_predictor.api.predict.is_model_loaded", return_value=False):

        # Act
        response = client.get("/predict", params=VALID_PARAMS)

    # Assert
    assert response.status_code == 503


def test_predict_returns_422_for_invalid_job_title() -> None:
    """An unrecognised job title is rejected with 422 Unprocessable Entity."""
    # Arrange
    bad_params = {**VALID_PARAMS, "job_title": "Pizza Chef"}

    with patch("salary_predictor.api.predict.is_model_loaded", return_value=True), \
         patch(
             "salary_predictor.api.predict.predict_salary",
             side_effect=ValueError("'Pizza Chef' is not valid for 'job_title'"),
         ):

        # Act
        response = client.get("/predict", params=bad_params)

    # Assert
    assert response.status_code == 422
    assert "not valid" in response.json()["detail"]


def test_predict_returns_500_on_unexpected_error() -> None:
    """An unexpected internal error returns 500 with a safe generic message."""
    # Arrange
    with patch("salary_predictor.api.predict.is_model_loaded", return_value=True), \
         patch(
             "salary_predictor.api.predict.predict_salary",
             side_effect=RuntimeError("unexpected internal failure"),
         ):

        # Act
        response = client.get("/predict", params=VALID_PARAMS)

    # Assert
    assert response.status_code == 500
    # The raw error must NOT leak to the client
    assert "unexpected internal failure" not in response.json()["detail"]


# ── /valid-values ──────────────────────────────────────────────────────

def test_valid_values_returns_three_keys() -> None:
    """/valid-values always returns job_title, employee_residence, company_location."""
    # Arrange
    mock_values = ["Data Scientist", "Data Engineer"]

    with patch(
        "salary_predictor.api.predict.get_valid_values",
        return_value=mock_values,
    ):
        # Act
        response = client.get("/valid-values")

    # Assert
    assert response.status_code == 200
    data = response.json()
    assert "job_title" in data
    assert "employee_residence" in data
    assert "company_location" in data


def test_valid_values_returns_lists_not_empty() -> None:
    """Each field in /valid-values is a non-empty list."""
    # Arrange
    with patch(
        "salary_predictor.api.predict.get_valid_values",
        return_value=["Data Scientist", "Data Engineer"],
    ):
        # Act
        response = client.get("/valid-values")

    # Assert
    data = response.json()
    for key in ("job_title", "employee_residence", "company_location"):
        assert isinstance(data[key], list)
        assert len(data[key]) > 0
