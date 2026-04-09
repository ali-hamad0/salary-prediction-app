# src/salary_predictor/llm/analyst.py
#
# Sends ONE prediction's inputs + salary to Ollama
# and returns a context-aware narrative.
#
# The prompt changes based on the actual inputs —
# experience level, job title, company size, salary.
# This makes every narrative unique to that prediction.
#
# When a Supabase DataFrame is provided, we compute the real
# market average for that job title + experience level and
# inject the comparison (above/below by X%) directly into the
# prompt so the LLM never has to guess.

import logging
import os

import pandas as pd
import requests
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL    = os.getenv("OLLAMA_MODEL", "llama3.1:latest")

# Human-readable labels for the coded values
EXPERIENCE_LABELS = {
    "EN": "Entry-level",
    "MI": "Mid-level",
    "SE": "Senior",
    "EX": "Executive",
}
EMPLOYMENT_LABELS = {
    "FT": "Full-time",
    "PT": "Part-time",
    "CT": "Contract",
    "FL": "Freelance",
}
SIZE_LABELS = {
    "S": "Small",
    "M": "Medium",
    "L": "Large",
}
REMOTE_LABELS = {
    0:   "fully on-site",
    50:  "hybrid",
    100: "fully remote",
}


def _market_comparison(
    inputs: dict,
    predicted_salary: float,
    df: pd.DataFrame | None,
) -> str:
    """
    Compute how this prediction compares to the real market average
    from Supabase data. Returns a plain-English sentence to inject
    into the LLM prompt.

    We look for rows matching the same job_title + experience_level.
    If there are fewer than 5, we widen to just experience_level.
    If still fewer than 5, we return a neutral fallback string.

    Args:
        inputs:           user inputs dict
        predicted_salary: the salary predicted by the model
        df:               optional Supabase predictions DataFrame

    Returns:
        comparison string, e.g. "above the market average by 12%"
    """
    if df is None or df.empty:
        return "within the expected range for this role"

    title      = inputs["job_title"]
    experience = inputs["experience_level"]

    # Try narrow slice first: same title + same experience level
    narrow = df[
        (df["job_title"] == title) &
        (df["experience_level"] == experience)
    ]["predicted_salary_usd"]

    # Widen to just experience level if not enough data
    if len(narrow) < 5:
        narrow = df[
            df["experience_level"] == experience
        ]["predicted_salary_usd"]

    if len(narrow) < 5:
        return "within the expected range for this role"

    market_avg = float(narrow.mean())
    diff_pct   = ((predicted_salary - market_avg) / market_avg) * 100

    if abs(diff_pct) < 3:
        return f"right at the market average of ${market_avg:,.0f}"
    elif diff_pct > 0:
        return f"above the market average of ${market_avg:,.0f} by {diff_pct:.0f}%"
    else:
        return f"below the market average of ${market_avg:,.0f} by {abs(diff_pct):.0f}%"


def build_prompt(
    inputs: dict,
    predicted_salary: float,
    market_context: str,
) -> str:
    """
    Build a context-aware prompt using the actual prediction inputs
    and a pre-computed market comparison string.

    Every field from the user's input is included so the LLM
    can write a narrative specific to this exact scenario.

    Args:
        inputs:           dict of user inputs (experience, title, etc.)
        predicted_salary: the salary the model predicted
        market_context:   plain-English comparison vs. real market average

    Returns:
        prompt string to send to Ollama
    """
    experience = EXPERIENCE_LABELS.get(inputs["experience_level"], inputs["experience_level"])
    employment = EMPLOYMENT_LABELS.get(inputs["employment_type"], inputs["employment_type"])
    size       = SIZE_LABELS.get(inputs["company_size"], inputs["company_size"])
    remote     = REMOTE_LABELS.get(inputs["remote_ratio"], f"{inputs['remote_ratio']}% remote")
    title      = inputs["job_title"]
    salary     = f"${predicted_salary:,.0f}"

    return f"""You are a senior career strategist writing a salary intelligence report for a data science professional.

PREDICTION DATA:
- Role            : {title}
- Experience      : {experience}
- Employment      : {employment}
- Company size    : {size}
- Work arrangement: {remote}
- Predicted salary: {salary} per year
- Market position : {market_context}

Write exactly 3 paragraphs. No bullet points. No headers. No filler phrases like "In conclusion" or "It is worth noting".

Paragraph 1 — THE SALARY IN CONTEXT:
Open with a strong, specific sentence that anchors the salary to reality for this exact role and level.
State that this salary is {market_context} — treat this as fact, not speculation.
Explain what {salary} per year actually means at the {experience} stage of a {title}'s career —
is this a comfortable starting point, a competitive offer, or a ceiling they need to break through?
Make the reader feel the weight of the number.

Paragraph 2 — WHAT IS DRIVING THIS NUMBER:
Identify the two most influential factors shaping this salary from the inputs provided.
Do not list them — weave them into a connected explanation.
Be specific: "{size} companies" behave differently from large ones in how they price talent.
"{remote}" roles carry different market dynamics than on-site positions.
"{employment}" status affects total compensation beyond base salary.
Explain the cause-and-effect — why do these factors push the salary in the direction they do?

Paragraph 3 — THE MOVE THAT CHANGES EVERYTHING:
Give one strategic, high-leverage move this specific professional can make to meaningfully grow their salary.
Not generic advice. Think about the combination of their level ({experience}), their setup ({size} company, {remote}),
and their role ({title}).
What is the single lever that would move their number the most — a transition, a negotiation angle, a skill bet?
Make the advice feel urgent and actionable, not theoretical.

Write in a confident, direct voice. No hedging. Each paragraph should feel like it was written specifically for this person."""


def get_narrative(
    inputs: dict,
    predicted_salary: float,
    df: pd.DataFrame | None = None,
) -> str:
    """
    Call Ollama and return a narrative for one prediction.

    Args:
        inputs:           dict of user inputs from the dashboard form
        predicted_salary: the salary predicted by the model
        df:               optional Supabase DataFrame used to compute
                          real market comparison numbers

    Returns:
        narrative string from Ollama

    Raises:
        RuntimeError: if Ollama is unreachable or returns empty
    """
    market_context = _market_comparison(inputs, predicted_salary, df)
    prompt         = build_prompt(inputs, predicted_salary, market_context)

    logger.info(
        "Calling Ollama for %s / %s — predicted $%,.0f — %s",
        inputs.get("job_title"),
        inputs.get("experience_level"),
        predicted_salary,
        market_context,
    )

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
            "Make sure Ollama is running: ollama serve"
        )
    except requests.exceptions.Timeout:
        raise RuntimeError("Ollama took too long. Try again.")
    except requests.exceptions.RequestException as error:
        raise RuntimeError(f"Ollama request failed: {error}") from error

    narrative = response.json().get("response", "").strip()

    if not narrative:
        raise RuntimeError("Ollama returned an empty response.")

    logger.info("Narrative generated — %d characters", len(narrative))
    return narrative
