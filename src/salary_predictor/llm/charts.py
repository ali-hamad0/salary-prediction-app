# src/salary_predictor/llm/charts.py
#
# Generates a salary chart from prediction stats and returns
# it as a base64 string so it can be stored in Supabase
# and displayed directly in the Streamlit dashboard.

import base64
import logging
from io import BytesIO

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

logger = logging.getLogger(__name__)


def make_experience_chart(stats: dict) -> str:
    """
    Bar chart showing mean salary by experience level.

    We return base64 so the chart can be stored as text in
    Supabase — no file storage or external URLs needed.

    Args:
        stats: dict from compute_stats()

    Returns:
        base64-encoded PNG string
    """
    exp_labels = {"EN": "Entry", "MI": "Mid", "SE": "Senior", "EX": "Executive"}
    order      = ["EN", "MI", "SE", "EX"]

    # Only keep levels that exist in the stats
    levels   = [exp_labels[k] for k in order if k in stats["by_experience"]]
    salaries = [stats["by_experience"][k] for k in order if k in stats["by_experience"]]

    fig, ax = plt.subplots(figsize=(9, 5))
    fig.patch.set_facecolor("white")

    colors = ["#AEC6E8", "#5A9FD4", "#1F77B4", "#0D4F8B"]
    bars   = ax.bar(
        levels, salaries,
        color=colors[:len(levels)],
        edgecolor="white",
        linewidth=0.8,
    )

    # Dollar label on top of each bar
    for bar, val in zip(bars, salaries):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 500,
            f"${val:,.0f}",
            ha="center", va="bottom",
            fontsize=11, fontweight="bold",
        )

    ax.set_title(
        "Mean Predicted Salary by Experience Level",
        fontsize=14, fontweight="bold", pad=16,
    )
    ax.set_xlabel("Experience Level", fontsize=11)
    ax.set_ylabel("Mean Annual Salary (USD)", fontsize=11)
    ax.yaxis.set_major_formatter(
        mticker.FuncFormatter(lambda y, _: f"${y/1000:.0f}k")
    )
    ax.set_ylim(0, max(salaries) * 1.2)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()

    # Save to memory buffer and encode as base64
    buffer = BytesIO()
    plt.savefig(buffer, format="png", dpi=150, bbox_inches="tight")
    plt.close()
    buffer.seek(0)

    chart_base64 = base64.b64encode(buffer.read()).decode("utf-8")
    logger.info("Chart generated and encoded as base64")

    return chart_base64