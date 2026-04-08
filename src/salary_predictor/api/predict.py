# src/salary_predictor/api/predict.py
#
# This file handles one job: receive a request, call the model,
# return the prediction. Each step is separated clearly.

import logging

from fastapi import APIRouter, HTTPException

from salary_predictor.api.schemas import PredictionRequest, PredictionResponse
from salary_predictor.model import predict_salary, get_valid_values, is_model_loaded

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get("/predict", response_model=PredictionResponse)
def predict(
    experience_level: str,
    employment_type: str,
    company_size: str,
    remote_ratio: int,
    work_year: int,
    job_title: str,
    employee_residence: str,
    company_location: str,
) -> PredictionResponse:
    """
    Predict the expected salary in USD for a data science position.

    All string inputs are validated against the trained encoders.
    If an invalid value is sent, a 422 error is returned with a
    clear message showing what values are accepted.
    """
    # Safety check — reject requests if model failed to load at startup
    if not is_model_loaded():
        raise HTTPException(
            status_code=503,
            detail="Model is not ready yet. Please try again in a moment.",
        )

    # Bundle all inputs into one dict to pass to the model
    inputs = {
        "experience_level":   experience_level,
        "employment_type":    employment_type,
        "company_size":       company_size,
        "remote_ratio":       remote_ratio,
        "work_year":          work_year,
        "job_title":          job_title,
        "employee_residence": employee_residence,
        "company_location":   company_location,
    }

    try:
        salary = predict_salary(inputs)

    except ValueError as error:
        # This happens when a string value is not in the encoder
        # e.g. job_title="Pizza Chef" — not in training data
        logger.warning("Invalid input received: %s", error)
        raise HTTPException(status_code=422, detail=str(error)) from error

    except Exception as error:
        # Something unexpected went wrong — log full details, hide from user
        logger.error("Prediction failed: %s", error, exc_info=True)
        raise HTTPException(
            status_code=500,
            detail="Something went wrong. Please check your inputs and try again.",
        ) from error

    logger.info("Predicted salary: $%.0f for %s / %s", salary, job_title, experience_level)

    return PredictionResponse(
        predicted_salary_usd=salary,
        inputs=inputs,
    )

@router.get("/debug")
def debug() -> dict:
    """Temporary endpoint — shows what features the model expects."""
    from salary_predictor.model import _metadata, _encoders
    return {
        "feature_names": _metadata.get("feature_names", []),
        "encoded_columns": list(_encoders.keys()),
    }

@router.get("/valid-values")
def valid_values() -> dict:
    """
    Return all accepted values for the free-text fields.

    Call this first to know which job_title, employee_residence,
    and company_location values the model will accept.
    """
    return {
        "job_title":          get_valid_values("job_title"),
        "employee_residence": get_valid_values("employee_residence"),
        "company_location":   get_valid_values("company_location"),
    }