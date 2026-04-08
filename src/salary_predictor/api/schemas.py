# src/salary_predictor/api/schemas.py
#
# We define what the API accepts and what it returns.
# Pydantic checks every value automatically — if someone sends
# "XL" for company_size, FastAPI rejects it before our code runs.

from typing import Literal
from pydantic import BaseModel, Field


class PredictionRequest(BaseModel):
    """Every field the API needs to make a salary prediction."""

    experience_level: Literal["EN", "MI", "SE", "EX"] = Field(
        description="EN=Entry  MI=Mid  SE=Senior  EX=Executive"
    )
    employment_type: Literal["FT", "PT", "CT", "FL"] = Field(
        description="FT=Full-time  PT=Part-time  CT=Contract  FL=Freelance"
    )
    company_size: Literal["S", "M", "L"] = Field(
        description="S=Small  M=Medium  L=Large"
    )
    remote_ratio: Literal[0, 50, 100] = Field(
        description="0=On-site  50=Hybrid  100=Fully remote"
    )
    work_year: int = Field(
        ge=2020,
        le=2023,
        description="Year the salary was paid — between 2020 and 2023",
    )
    job_title: str = Field(
        description="Job title — call /valid-values to see all accepted titles"
    )
    employee_residence: str = Field(
        description="Employee country code e.g. US, GB, IN"
    )
    company_location: str = Field(
        description="Company country code e.g. US, GB, IN"
    )


class PredictionResponse(BaseModel):
    """What the API sends back after a successful prediction."""

    predicted_salary_usd: float
    inputs: dict


class HealthResponse(BaseModel):
    """Simple check that the API is alive and the model is ready."""

    status: str
    model_loaded: bool