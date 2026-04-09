# src/salary_predictor/llm/charts.py
#
# Generates a context-aware chart for ONE prediction.
#
# The chart is NOT the same every time.
# It adapts based on the prediction:
#
#   - If the Supabase data has enough rows for this job title (>=8),
#     the bars show REAL average salaries for that title by experience level.
#     Title: "Salary by Experience — {Job Title}"
#
#   - If there's not enough data for that title but enough for the company size,
#     the bars show averages for that company size.
#     Title: "Salary by Experience — {Size} Companies"
#
#   - Fallback: reference averages from the full dataset.
#     Title: "Salary by Experience — Market Reference"
#
# In all cases:
#   - The predicted experience level bar is highlighted in blue
#   - A dashed red line marks this specific prediction's salary
#   - The chart title and subtitle describe exactly what data is shown

import base64
import logging
from io import BytesIO

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import pandas as pd

logger = logging.getLogger(__name__)

EXPERIENCE_ORDER = ["EN", "MI", "SE", "EX"]

EXPERIENCE_LABELS = {
    "EN": "Entry",
    "MI": "Mid",
    "SE": "Senior",
    "EX": "Executive",
}

SIZE_LABELS = {
    "S": "Small",
    "M": "Medium",
    "L": "Large",
}

# Fallback reference averages from the full training dataset.
# Used only when Supabase has no useful data to slice from.
REFERENCE_AVERAGES = {
    "EN": 60000,
    "MI": 95000,
    "SE": 140000,
    "EX": 165000,
}


def _compute_averages(df: pd.DataFrame) -> dict:
    """
    Given a filtered DataFrame, compute the mean predicted salary
    for each experience level that has at least one row.

    Args:
        df: filtered DataFrame with 'experience_level' and
            'predicted_salary_usd' columns

    Returns:
        dict mapping experience code → average salary (float)
    """
    result = {}
    for level in EXPERIENCE_ORDER:
        subset = df[df["experience_level"] == level]["predicted_salary_usd"]
        if not subset.empty:
            result[level] = float(subset.mean())
    return result


def _resolve_chart_data(inputs: dict, df: pd.DataFrame | None) -> tuple[dict, str]:
    """
    Decide what data to show and what to call the chart.

    Strategy (in order):
      1. Filter df to this job_title — use if >= 8 rows
      2. Filter df to this company_size — use if >= 8 rows
      3. Fall back to REFERENCE_AVERAGES

    Returns:
        (averages_dict, chart_subtitle)
        averages_dict: {experience_level: avg_salary} for all 4 levels
    """
    job_title    = inputs["job_title"]
    company_size = inputs["company_size"]

    if df is not None and not df.empty:
        # Strategy 1 — job title slice
        title_df = df[df["job_title"] == job_title]
        if len(title_df) >= 8:
            averages = _compute_averages(title_df)
            # Fill any missing levels with reference so all 4 bars appear
            for lvl in EXPERIENCE_ORDER:
                if lvl not in averages:
                    averages[lvl] = REFERENCE_AVERAGES[lvl]
            subtitle = f"Real data — {job_title}"
            logger.info(
                "Chart: using %d rows for job title '%s'", len(title_df), job_title
            )
            return averages, subtitle

        # Strategy 2 — company size slice
        size_df = df[df["company_size"] == company_size]
        if len(size_df) >= 8:
            averages = _compute_averages(size_df)
            for lvl in EXPERIENCE_ORDER:
                if lvl not in averages:
                    averages[lvl] = REFERENCE_AVERAGES[lvl]
            size_label = SIZE_LABELS.get(company_size, company_size)
            subtitle   = f"Real data — {size_label} Companies"
            logger.info(
                "Chart: using %d rows for company size '%s'", len(size_df), company_size
            )
            return averages, subtitle

    # Strategy 3 — reference fallback
    logger.info("Chart: not enough Supabase data — using reference averages")
    return dict(REFERENCE_AVERAGES), "Market reference averages"


def make_prediction_chart(
    inputs: dict,
    predicted_salary: float,
    df: pd.DataFrame | None = None,
) -> str:
    """
    Bar chart comparing this prediction against average salaries
    by experience level. The chart data adapts to the prediction context.

    Args:
        inputs:           dict of user inputs (experience_level, job_title, etc.)
        predicted_salary: the salary predicted by the model
        df:               optional DataFrame from Supabase; used to compute
                          real averages instead of hardcoded references

    Returns:
        base64-encoded PNG string
    """
    experience = inputs["experience_level"]
    title      = inputs["job_title"]

    averages, subtitle = _resolve_chart_data(inputs, df)

    levels   = EXPERIENCE_ORDER
    labels   = [EXPERIENCE_LABELS[k] for k in levels]
    heights  = [averages.get(k, 0) for k in levels]

    # Highlight the predicted experience level, mute the rest
    colors = [
        "#1F77B4" if lvl == experience else "#C8D8E8"
        for lvl in levels
    ]

    fig, ax = plt.subplots(figsize=(9, 5))
    fig.patch.set_facecolor("white")

    bars = ax.bar(
        labels, heights,
        color=colors,
        edgecolor="white",
        linewidth=0.8,
    )

    # Label on top of each bar
    for bar, val in zip(bars, heights):
        if val > 0:
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                val + 500,
                f"${val:,.0f}",
                ha="center", va="bottom",
                fontsize=10, color="#444444",
            )

    # Dashed line — this prediction's salary
    ax.axhline(
        y=predicted_salary,
        color="#DD4444",
        linestyle="--",
        linewidth=1.8,
        label=f"This prediction: ${predicted_salary:,.0f}",
    )

    ax.set_title(
        f"Salary by Experience Level\n{subtitle}",
        fontsize=13, fontweight="bold", pad=14,
    )
    ax.set_xlabel("Experience Level", fontsize=11)
    ax.set_ylabel("Annual Salary (USD)", fontsize=11)
    ax.yaxis.set_major_formatter(
        mticker.FuncFormatter(lambda y, _: f"${y/1000:.0f}k")
    )
    max_val = max(max(heights), predicted_salary) if heights else predicted_salary
    ax.set_ylim(0, max_val * 1.25)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(fontsize=10)

    plt.tight_layout()

    buffer = BytesIO()
    plt.savefig(buffer, format="png", dpi=150, bbox_inches="tight")
    plt.close()
    buffer.seek(0)

    chart_base64 = base64.b64encode(buffer.read()).decode("utf-8")
    logger.info(
        "Chart generated for %s / %s — source: %s",
        title, experience, subtitle,
    )
    return chart_base64
