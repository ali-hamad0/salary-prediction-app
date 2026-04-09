# src/salary_predictor/db/persist.py
#
# Saves bulk predictions to Supabase.
# Each row stores the full input set, predicted salary,
# and optionally the LLM narrative and chart (base64).
#
# Used by run_analysis.py (bulk pipeline) and the dashboard
# (individual predictions).

import logging

from salary_predictor.db.supabase_client import get_client

logger = logging.getLogger(__name__)

# How many predictions to insert at once.
# Supabase has a request size limit so we insert in batches.
BATCH_SIZE = 100


def save_predictions(predictions: list, run_id: str) -> int:
    """
    Save all predictions to the predictions table.

    We insert in batches of 100 to avoid hitting Supabase
    request size limits.

    Args:
        predictions: list of prediction dicts from run_pipeline()
        run_id:      unique ID linking this batch to its analysis

    Returns:
        number of rows successfully inserted
    """
    client = get_client()
    total_saved = 0

    # Split predictions into batches
    batches = [
        predictions[i : i + BATCH_SIZE]
        for i in range(0, len(predictions), BATCH_SIZE)
    ]

    logger.info(
        "Saving %d predictions in %d batches...",
        len(predictions), len(batches),
    )

    for batch_number, batch in enumerate(batches, start=1):
        # Build the rows to insert
        rows = []
        for item in batch:
            rows.append({
                "run_id":               run_id,
                "experience_level":     item["experience_level"],
                "employment_type":      item["employment_type"],
                "company_size":         item["company_size"],
                "remote_ratio":         item["remote_ratio"],
                "work_year":            item["work_year"],
                "job_title":            item["job_title"],
                "employee_residence":   item["employee_residence"],
                "company_location":     item["company_location"],
                "predicted_salary_usd": item["predicted_salary_usd"],
            })

        try:
            client.table("predictions").insert(rows).execute()
            total_saved += len(rows)
            logger.info(
                "Batch %d / %d saved (%d rows)",
                batch_number, len(batches), len(rows),
            )

        except Exception as error:
            logger.error("Batch %d failed: %s", batch_number, error)

    logger.info("Predictions saved: %d / %d", total_saved, len(predictions))
    return total_saved


