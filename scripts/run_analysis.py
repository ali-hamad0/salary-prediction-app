# scripts/run_analysis.py
#
# Runs the full analysis pipeline:
#   1. Calls run_pipeline() to get predictions from the API
#   2. Sends predictions to Ollama for narrative generation
#   3. Generates a salary chart
#   4. Saves everything to Supabase

import importlib.util
import logging
import os
import sys
import uuid

from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)


def load_run_pipeline():
    """
    Load run_pipeline.py directly from its file path.

    We do this because scripts/ is not a Python package
    so we cannot use a normal import statement on it.

    Returns:
        the run_pipeline function
    """
    pipeline_path = os.path.join(os.path.dirname(__file__), "run_pipeline.py")
    spec   = importlib.util.spec_from_file_location("run_pipeline", pipeline_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.run_pipeline


def main() -> None:
    """Run predictions → LLM analysis → chart → Supabase."""

    from salary_predictor.llm.analyst import get_narrative
    from salary_predictor.llm.charts import make_experience_chart
    from salary_predictor.db.persist import save_predictions, save_analysis

    # Step 1 — get predictions from the API
    logger.info("Step 1 — running prediction pipeline...")
    run_pipeline = load_run_pipeline()
    predictions  = run_pipeline()

    if not predictions:
        logger.error("No predictions returned — is the API running?")
        sys.exit(1)

    logger.info("Got %d predictions", len(predictions))

    # Step 2 — generate LLM narrative
    logger.info("Step 2 — generating LLM narrative...")
    result    = get_narrative(predictions)
    narrative = result["narrative"]
    stats     = result["stats"]

    # Step 3 — generate chart
    logger.info("Step 3 — generating chart...")
    chart_base64 = make_experience_chart(stats)

    # Step 4 — print so we can verify
    logger.info("=" * 50)
    logger.info("Analysis complete")
    logger.info("=" * 50)

    print("\n--- NARRATIVE ---\n")
    print(narrative)

    print("\n--- STATS SUMMARY ---")
    print(f"  Total predictions : {stats['total_predictions']}")
    print(f"  Overall average   : ${stats['overall_mean']:,.0f}")
    print(f"  Salary range      : ${stats['overall_min']:,.0f} — ${stats['overall_max']:,.0f}")

    print("\n--- CHART ---")
    print(f"  Base64 length     : {len(chart_base64)} characters")
    print(f"  First 60 chars    : {chart_base64[:60]}...")

    # Step 5 — save to Supabase
    logger.info("Step 5 — saving to Supabase...")

    # One run_id links all predictions to their analysis
    run_id = str(uuid.uuid4())
    logger.info("Run ID: %s", run_id)

    rows_saved     = save_predictions(predictions, run_id)
    analysis_saved = save_analysis(run_id, narrative, chart_base64, stats)

    print(f"\n--- SUPABASE ---")
    print(f"  Run ID           : {run_id}")
    print(f"  Predictions saved: {rows_saved}")
    print(f"  Analysis saved   : {analysis_saved}")
    print("\nPhase 6 complete. Ready for Phase 7 — Streamlit dashboard.")


if __name__ == "__main__":
    main()