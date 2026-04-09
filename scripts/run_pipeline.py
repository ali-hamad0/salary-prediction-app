# scripts/run_pipeline.py
#
# This script does three things in order:
#   1. Generates a representative set of input combinations
#   2. Calls the prediction API for each combination
#   3. Collects and returns all results
#
# "Covering the input space" means we pick the axes that matter
# most for salary and combine them using itertools.product.
# 4 experience × 4 employment × 3 company size × 3 remote × 8 titles
# = 1,152 combinations. Every meaningful scenario is covered.

import itertools
import logging
import os
import sys
import time

import requests
from dotenv import load_dotenv

load_dotenv()

# ── Logging ────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)

# ── Config ─────────────────────────────────────────────────────
API_BASE_URL     = os.getenv("API_BASE_URL", "http://127.0.0.1:8000")
REQUEST_TIMEOUT  = 10     # seconds per request before giving up
PAUSE_BETWEEN    = 0.02   # small pause so we don't hammer the API


# ── Input space ────────────────────────────────────────────────
# We fix residence and location to US — most data points are US-based.
# We combine everything else that meaningfully affects salary.

EXPERIENCE_LEVELS  = ["EN", "MI", "SE", "EX"]
EMPLOYMENT_TYPES   = ["FT", "PT", "CT", "FL"]
COMPANY_SIZES      = ["S", "M", "L"]
REMOTE_RATIOS      = [0, 50, 100]
WORK_YEAR          = 2023
EMPLOYEE_RESIDENCE = "US"
COMPANY_LOCATION   = "US"


def check_api_health() -> None:
    """
    Check the API is alive before we start sending requests.
    Exits immediately with a clear message if it is not reachable.
    """
    try:
        response = requests.get(
            f"{API_BASE_URL}/health",
            timeout=REQUEST_TIMEOUT,
        )
        response.raise_for_status()
        data = response.json()

        if not data.get("model_loaded"):
            logger.critical("API is running but model is not loaded. Start the server first.")
            sys.exit(1)

        logger.info("API health check passed — model is loaded and ready.")

    except requests.exceptions.ConnectionError:
        logger.critical(
            "Cannot connect to API at %s\n"
            "Make sure the server is running:\n"
            "  uv run uvicorn salary_predictor.api.main:app --reload --port 8000",
            API_BASE_URL,
        )
        sys.exit(1)


def fetch_job_titles() -> list:
    """
    Fetch the list of valid job titles directly from the API.

    We do this dynamically so we never hardcode titles that might
    not exist in the model's encoders.

    Returns:
        list of valid job title strings

    Raises:
        SystemExit: if the request fails
    """
    try:
        response = requests.get(
            f"{API_BASE_URL}/valid-values",
            timeout=REQUEST_TIMEOUT,
        )
        response.raise_for_status()
        titles = response.json().get("job_title", [])
        logger.info("Fetched %d valid job titles from API.", len(titles))
        return titles

    except requests.exceptions.RequestException as error:
        logger.critical("Could not fetch valid values: %s", error)
        sys.exit(1)


def generate_combinations(job_titles: list) -> list:
    """
    Build every meaningful combination of inputs.

    We use itertools.product which gives us the cartesian product —
    every possible pairing across all the axes we care about.

    Args:
        job_titles: list of valid job titles from the API

    Returns:
        list of dicts — each dict is one full set of inputs
    """
    combos = []

    for experience, employment, size, remote, title in itertools.product(
        EXPERIENCE_LEVELS,
        EMPLOYMENT_TYPES,
        COMPANY_SIZES,
        REMOTE_RATIOS,
        job_titles,
    ):
        combos.append({
            "experience_level":   experience,
            "employment_type":    employment,
            "company_size":       size,
            "remote_ratio":       remote,
            "work_year":          WORK_YEAR,
            "job_title":          title,
            "employee_residence": EMPLOYEE_RESIDENCE,
            "company_location":   COMPANY_LOCATION,
        })

    logger.info("Generated %d input combinations.", len(combos))
    return combos


def call_predict(inputs: dict) -> dict | None:
    """
    Send one prediction request to the API.

    We never let a single failed request crash the whole script.
    If a request fails for any reason we log a warning and return None.

    Args:
        inputs: dict with all required fields for the /predict endpoint

    Returns:
        dict with the API response, or None if the request failed
    """
    try:
        response = requests.get(
            f"{API_BASE_URL}/predict",
            params=inputs,
            timeout=REQUEST_TIMEOUT,
        )
        response.raise_for_status()
        return response.json()

    except requests.exceptions.Timeout:
        logger.warning(
            "Request timed out: %s / %s",
            inputs.get("job_title"), inputs.get("experience_level"),
        )
        return None

    except requests.exceptions.RequestException as error:
        logger.warning(
            "Request failed for %s / %s: %s",
            inputs.get("job_title"), inputs.get("experience_level"), error,
        )
        return None


def run_pipeline() -> list:
    """
    Full pipeline: health check → fetch titles → generate
    combinations → call API for each → return results.

    Returns:
        list of successful prediction result dicts
    """
    logger.info("=" * 50)
    logger.info("Pipeline starting")
    logger.info("=" * 50)

    # Step 1 — make sure the API is up
    check_api_health()

    # Step 2 — get valid job titles from the API
    job_titles = fetch_job_titles()

    # Step 3 — generate all combinations
    combinations = generate_combinations(job_titles)
    total = len(combinations)

    # Step 4 — call the API for each combination
    results = []
    failed  = 0

    logger.info("Sending %d requests — this will take ~%d seconds...",
                total, int(total * PAUSE_BETWEEN))

    for index, inputs in enumerate(combinations, start=1):

        # Log progress every 100 requests so we know it's working
        if index % 100 == 0 or index == 1:
            logger.info("Progress: %d / %d", index, total)

        result = call_predict(inputs)

        if result is not None:
            # Flatten inputs and prediction into one clean dict
            results.append({
                "experience_level":    inputs["experience_level"],
                "employment_type":     inputs["employment_type"],
                "company_size":        inputs["company_size"],
                "remote_ratio":        inputs["remote_ratio"],
                "work_year":           inputs["work_year"],
                "job_title":           inputs["job_title"],
                "employee_residence":  inputs["employee_residence"],
                "company_location":    inputs["company_location"],
                "predicted_salary_usd": result["predicted_salary_usd"],
            })
        else:
            failed += 1

        time.sleep(PAUSE_BETWEEN)

    # Step 5 — summary
    logger.info("=" * 50)
    logger.info("Pipeline complete")
    logger.info("  Total combinations : %d", total)
    logger.info("  Successful         : %d", len(results))
    logger.info("  Failed             : %d", failed)

    if results:
        salaries = [r["predicted_salary_usd"] for r in results]
        min_sal = min(salaries)
        max_sal = max(salaries)
        avg_sal = sum(salaries) / len(salaries)
        logger.info("  Salary range       : $%s — $%s",
                    f"{min_sal:,.0f}", f"{max_sal:,.0f}")
        logger.info("  Average salary     : $%s",
            f"{avg_sal:,.0f}")

    logger.info("=" * 50)

    return results


# ── Entry point ────────────────────────────────────────────────
if __name__ == "__main__":
    predictions = run_pipeline()

    # Print first 3 results so we can visually verify the output
    print("\nFirst 3 results:")
    for item in predictions[:3]:
        print(
            f"  {item['job_title']:<30} "
            f"{item['experience_level']} / {item['company_size']} / "
            f"remote {item['remote_ratio']}%  →  "
            f"${item['predicted_salary_usd']:,.0f}"
        )