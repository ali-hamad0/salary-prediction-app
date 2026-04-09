# scripts/run_analysis.py
#
# Runs the full analysis pipeline:
#   1. Calls run_pipeline() to get predictions from the API
#   2. Sends predictions to Ollama for narrative generation
#   3. Generates a salary chart
#   4. Prints results so we can verify before Phase 6
#
# Phase 6 will add Supabase saving to this same flow.

import importlib.util
import logging
import os
import sys

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
    """Run predictions → LLM analysis → chart generation."""

    from salary_predictor.llm.analyst import get_narrative
    from salary_predictor.llm.charts import make_experience_chart

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

    # Step 4 — print everything so we can verify
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

    print("\nPhase 5 complete. Ready for Phase 6 — Supabase persistence.")

    return predictions, narrative, stats, chart_base64


if __name__ == "__main__":
    main()