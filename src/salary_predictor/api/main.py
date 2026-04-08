# src/salary_predictor/api/main.py
#
# This is where everything comes together.
# FastAPI starts here, the model loads here, routes are registered here.

import logging
import sys

from fastapi import FastAPI

from salary_predictor.api.predict import router
from salary_predictor.api.schemas import HealthResponse
from salary_predictor.model import load_model, is_model_loaded

# ── Logging ───────────────────────────────────────────────────────────
# We use logging instead of print() so messages have timestamps and levels.
# This follows our coding rules — never use print() in production code.
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)

# ── App ───────────────────────────────────────────────────────────────
app = FastAPI(
    title="Salary Predictor API",
    description="Predict data science salaries using a trained Random Forest model.",
    version="1.0.0",
)

app.include_router(router)


@app.on_event("startup")
def startup() -> None:
    """
    Runs once when the server starts.
    We load the model here so every request after this is instant.
    """
    logger.info("Server starting — loading model...")
    load_model()
    logger.info("Model loaded. API is ready.")


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    """Quick check — is the API alive and is the model loaded?"""
    return HealthResponse(
        status="ok" if is_model_loaded() else "model not loaded",
        model_loaded=is_model_loaded(),
    )