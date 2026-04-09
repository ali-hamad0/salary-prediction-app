# src/salary_predictor/llm/analyst.py
#
# Computes salary stats from predictions and sends them
# to Ollama to generate a written narrative.
#
# We compute the stats first and put real numbers in the prompt.
# This is the difference between a useful narrative and generic filler.

import logging
import os

import requests
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL    = os.getenv("OLLAMA_MODEL", "llama3.1:latest")


def compute_stats(predictions: list) -> dict:
    """
    Compute summary statistics from the prediction results.

    Args:
        predictions: list of dicts from run_pipeline()

    Returns:
        dict of grouped salary averages and overall stats
    """
    # Group salaries by experience level
    by_experience = {}
    for item in predictions:
        level = item["experience_level"]
        if level not in by_experience:
            by_experience[level] = []
        by_experience[level].append(item["predicted_salary_usd"])

    # Group salaries by job title
    by_title = {}
    for item in predictions:
        title = item["job_title"]
        if title not in by_title:
            by_title[title] = []
        by_title[title].append(item["predicted_salary_usd"])

    # Group salaries by remote ratio
    by_remote = {}
    for item in predictions:
        remote = item["remote_ratio"]
        if remote not in by_remote:
            by_remote[remote] = []
        by_remote[remote].append(item["predicted_salary_usd"])

    # Group salaries by company size
    by_size = {}
    for item in predictions:
        size = item["company_size"]
        if size not in by_size:
            by_size[size] = []
        by_size[size].append(item["predicted_salary_usd"])

    # Compute mean salary for each group
    def mean(values):
        return round(sum(values) / len(values), 2)

    all_salaries = [item["predicted_salary_usd"] for item in predictions]

    return {
        "total_predictions": len(predictions),
        "overall_mean":      round(mean(all_salaries), 2),
        "overall_min":       round(min(all_salaries), 2),
        "overall_max":       round(max(all_salaries), 2),
        "by_experience":     {k: mean(v) for k, v in by_experience.items()},
        "by_job_title":      {k: mean(v) for k, v in by_title.items()},
        "by_remote_ratio":   {k: mean(v) for k, v in by_remote.items()},
        "by_company_size":   {k: mean(v) for k, v in by_size.items()},
    }


def build_prompt(stats: dict) -> str:
    """
    Build the prompt with real numbers from the stats.

    Giving the LLM exact dollar figures prevents it from
    making up numbers or writing vague generic sentences.

    Args:
        stats: dict from compute_stats()

    Returns:
        full prompt string to send to Ollama
    """
    exp_labels  = {"EN": "Entry", "MI": "Mid", "SE": "Senior", "EX": "Executive"}
    size_labels = {"S": "Small", "M": "Medium", "L": "Large"}
    remote_labels = {0: "On-site", 50: "Hybrid", 100: "Fully remote"}

    # Format each group as readable lines
    exp_lines = "\n".join(
        f"  - {exp_labels.get(k, k)}: ${v:,.0f}"
        for k, v in sorted(stats["by_experience"].items())
    )
    title_lines = "\n".join(
        f"  - {k}: ${v:,.0f}"
        for k, v in sorted(
            stats["by_job_title"].items(),
            key=lambda x: x[1],
            reverse=True,
        )
    )
    remote_lines = "\n".join(
        f"  - {remote_labels.get(k, k)}: ${v:,.0f}"
        for k, v in sorted(stats["by_remote_ratio"].items())
    )
    size_lines = "\n".join(
        f"  - {size_labels.get(k, k)}: ${v:,.0f}"
        for k, v in sorted(stats["by_company_size"].items())
    )

    return f"""You are a data analyst writing a salary insights report.

Use ONLY the numbers below. Do not invent any figures.

=== DATA ===
Total predictions: {stats["total_predictions"]}
Salary range: ${stats["overall_min"]:,.0f} to ${stats["overall_max"]:,.0f}
Overall average: ${stats["overall_mean"]:,.0f}

By experience level:
{exp_lines}

By job title (highest to lowest):
{title_lines}

By remote work type:
{remote_lines}

By company size:
{size_lines}

=== INSTRUCTIONS ===
Write exactly 3 paragraphs. No bullet points. No title or heading.

Paragraph 1 — FINDING: The single most important pattern in the data.
Use specific dollar figures. Start with "The data reveals..."

Paragraph 2 — EXPLANATION: Why this pattern likely exists.
Connect it to how the job market actually works.

Paragraph 3 — IMPLICATION: What this means for a data science professional.
Give one concrete actionable recommendation.
Start with "For data science professionals..."

Total length: 150 to 200 words. Professional but conversational tone."""


def get_narrative(predictions: list) -> dict:
    """
    Compute stats and call Ollama to generate a narrative.

    Args:
        predictions: list of dicts from run_pipeline()

    Returns:
        dict with keys: narrative (str), stats (dict)

    Raises:
        RuntimeError: if Ollama is unreachable or returns empty
    """
    # Step 1 — compute stats
    stats = compute_stats(predictions)
    logger.info("Stats computed from %d predictions", len(predictions))

    # Step 2 — build prompt with real numbers
    prompt = build_prompt(stats)

    # Step 3 — call Ollama
    logger.info("Calling Ollama (%s) — this takes ~30 seconds...", OLLAMA_MODEL)

    try:
        response = requests.post(
            f"{OLLAMA_BASE_URL}/api/generate",
            json={
                "model":  OLLAMA_MODEL,
                "prompt": prompt,
                "stream": False,
            },
            timeout=120,
        )
        response.raise_for_status()

    except requests.exceptions.ConnectionError:
        raise RuntimeError(
            f"Cannot connect to Ollama at {OLLAMA_BASE_URL}. "
            "Is Ollama running? Try: ollama serve"
        )
    except requests.exceptions.Timeout:
        raise RuntimeError(
            "Ollama took too long to respond. "
            "Try again or check your machine resources."
        )
    except requests.exceptions.RequestException as error:
        raise RuntimeError(f"Ollama request failed: {error}") from error

    narrative = response.json().get("response", "").strip()

    if not narrative:
        raise RuntimeError("Ollama returned an empty response.")

    logger.info("Narrative generated — %d characters", len(narrative))

    return {
        "narrative": narrative,
        "stats":     stats,
    }