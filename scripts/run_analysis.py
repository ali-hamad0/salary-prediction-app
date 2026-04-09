# scripts/run_analysis.py
#
# Full local pipeline: generates bulk predictions by calling the API,
# then saves every result to the Supabase predictions table.
# LLM narrative and chart are generated per prediction in the dashboard
# when a user submits the form.

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
    """Load run_pipeline.py from its file path."""
    pipeline_path = os.path.join(os.path.dirname(__file__), "run_pipeline.py")
    spec   = importlib.util.spec_from_file_location("run_pipeline", pipeline_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.run_pipeline


def main() -> None:
    """Run bulk predictions and save to Supabase."""
    from salary_predictor.db.persist import save_predictions

    logger.info("Running bulk prediction pipeline...")
    run_pipeline = load_run_pipeline()
    predictions  = run_pipeline()

    if not predictions:
        logger.error("No predictions returned — is the API running?")
        sys.exit(1)

    rows_saved = save_predictions(predictions, run_id="bulk")

    print(f"\n  Predictions saved: {rows_saved}")
    print("\nDone. Open the dashboard to make individual predictions with LLM analysis.")


if __name__ == "__main__":
    main()