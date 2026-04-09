# src/salary_predictor/dashboard/app.py
#
# Streamlit dashboard with two tabs:
#   Tab 1 — Make a prediction: user fills form, gets salary + LLM analysis + chart
#   Tab 2 — All predictions: filterable table of everything in Supabase
#
# Reads from Supabase only — never calls the model directly.

import base64
import logging
import os
import sys
from io import BytesIO

import pandas as pd
import requests
import streamlit as st
from dotenv import load_dotenv
from PIL import Image

from salary_predictor.db.supabase_client import get_client
from salary_predictor.llm.analyst import get_narrative
from salary_predictor.llm.charts import make_prediction_chart

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)

API_BASE_URL = os.getenv("API_BASE_URL", "http://127.0.0.1:8000")

st.set_page_config(
    page_title="Salary Predictor",
    page_icon="💼",
    layout="wide",
)


# ── Helpers ───────────────────────────────────────────────────

def decode_chart(chart_base64: str) -> Image.Image | None:
    """Decode a base64 string back into a PIL image."""
    try:
        image_bytes = base64.b64decode(chart_base64)
        return Image.open(BytesIO(image_bytes))
    except Exception as error:
        logger.error("Failed to decode chart: %s", error)
        return None


def call_predict_api(inputs: dict) -> float | None:
    """
    Call the FastAPI /predict endpoint with the user's inputs.

    Args:
        inputs: dict of form values

    Returns:
        predicted salary as float, or None if the request failed
    """
    try:
        response = requests.get(
            f"{API_BASE_URL}/predict",
            params=inputs,
            timeout=15,
        )
        response.raise_for_status()
        return response.json()["predicted_salary_usd"]

    except requests.exceptions.ConnectionError:
        st.error("Cannot connect to the API. Make sure the server is running.")
        return None
    except requests.exceptions.RequestException as error:
        st.error(f"API request failed: {error}")
        return None


def save_prediction(inputs: dict, salary: float, narrative: str, chart_base64: str) -> None:
    """
    Save the prediction, narrative, and chart to Supabase.

    Args:
        inputs:       dict of user inputs
        salary:       predicted salary
        narrative:    LLM narrative text
        chart_base64: base64 chart string
    """
    try:
        client = get_client()
        client.table("predictions").insert({
            "experience_level":     inputs["experience_level"],
            "employment_type":      inputs["employment_type"],
            "company_size":         inputs["company_size"],
            "remote_ratio":         inputs["remote_ratio"],
            "work_year":            inputs["work_year"],
            "job_title":            inputs["job_title"],
            "employee_residence":   inputs["employee_residence"],
            "company_location":     inputs["company_location"],
            "predicted_salary_usd": salary,
            "narrative":            narrative,
            "chart_base64":         chart_base64,
        }).execute()
        logger.info("Prediction saved to Supabase")
    except Exception as error:
        logger.error("Failed to save to Supabase: %s", error)


@st.cache_data(ttl=60)
def load_all_predictions() -> pd.DataFrame:
    """Load all predictions from Supabase for the table tab."""
    try:
        client   = get_client()
        response = client.table("predictions").select("*").order(
            "created_at", desc=True
        ).execute()
        if not response.data:
            return pd.DataFrame()
        return pd.DataFrame(response.data)
    except Exception as error:
        logger.error("Failed to load predictions: %s", error)
        return pd.DataFrame()


# ── Valid values for form dropdowns ───────────────────────────

@st.cache_data(ttl=3600)
def fetch_valid_values() -> dict:
    """Fetch valid job titles and country codes from the API."""
    try:
        response = requests.get(f"{API_BASE_URL}/valid-values", timeout=10)
        response.raise_for_status()
        return response.json()
    except Exception:
        # Fallback values if API is unreachable
        return {
            "job_title":          ["Data Scientist", "Data Engineer",
                                   "Data Analyst", "Machine Learning Engineer",
                                   "Research Scientist", "Other"],
            "employee_residence": ["US", "GB", "IN", "DE", "CA"],
            "company_location":   ["US", "GB", "IN", "DE", "CA"],
        }


# ── Main ──────────────────────────────────────────────────────

def main() -> None:
    """Render the full dashboard."""

    st.title("💼 Data Science Salary Predictor")
    st.caption("Powered by a Random Forest model · Analysis by Ollama llama3.1")

    tab1, tab2 = st.tabs(["🔮 Make a Prediction", "📊 All Predictions"])

    # ── Tab 1: Make a prediction ──────────────────────────────
    with tab1:

        st.subheader("Enter Job Details")
        valid = fetch_valid_values()

        # Form inputs — two columns for cleaner layout
        col1, col2 = st.columns(2)

        with col1:
            job_title = st.selectbox(
                "Job title",
                options=valid.get("job_title", []),
            )
            experience_level = st.selectbox(
                "Experience level",
                options=["EN", "MI", "SE", "EX"],
                format_func=lambda x: {
                    "EN": "EN — Entry",
                    "MI": "MI — Mid",
                    "SE": "SE — Senior",
                    "EX": "EX — Executive",
                }[x],
            )
            employment_type = st.selectbox(
                "Employment type",
                options=["FT", "PT", "CT", "FL"],
                format_func=lambda x: {
                    "FT": "FT — Full-time",
                    "PT": "PT — Part-time",
                    "CT": "CT — Contract",
                    "FL": "FL — Freelance",
                }[x],
            )
            company_size = st.selectbox(
                "Company size",
                options=["S", "M", "L"],
                format_func=lambda x: {
                    "S": "S — Small",
                    "M": "M — Medium",
                    "L": "L — Large",
                }[x],
            )

        with col2:
            remote_ratio = st.selectbox(
                "Remote ratio",
                options=[0, 50, 100],
                format_func=lambda x: {
                    0:   "0% — On-site",
                    50:  "50% — Hybrid",
                    100: "100% — Fully remote",
                }[x],
            )
            work_year = st.selectbox(
                "Work year",
                options=[2020, 2021, 2022, 2023],
                index=3,
            )
            employee_residence = st.selectbox(
                "Employee residence",
                options=valid.get("employee_residence", ["US"]),
            )
            company_location = st.selectbox(
                "Company location",
                options=valid.get("company_location", ["US"]),
            )

        st.divider()

        # Predict button
        if st.button("🔮 Predict Salary", type="primary", use_container_width=True):

            inputs = {
                "experience_level":   experience_level,
                "employment_type":    employment_type,
                "company_size":       company_size,
                "remote_ratio":       remote_ratio,
                "work_year":          work_year,
                "job_title":          job_title,
                "employee_residence": employee_residence,
                "company_location":   company_location,
            }

            # Step 1 — call the API
            with st.spinner("Predicting salary..."):
                salary = call_predict_api(inputs)

            if salary is None:
                st.stop()

            # Show the predicted salary prominently
            st.metric("Predicted Annual Salary", f"${salary:,.0f}")

            # Step 2 — load Supabase data so LLM + chart use real market numbers
            context_df = load_all_predictions()
            context_df = context_df if not context_df.empty else None

            # Step 3 — generate LLM narrative (uses real market averages if available)
            with st.spinner("Analysing with Ollama — ~30 seconds..."):
                try:
                    narrative = get_narrative(inputs, salary, df=context_df)
                except RuntimeError as error:
                    st.error(f"LLM analysis failed: {error}")
                    narrative = None

            # Step 4 — generate chart (adapts to real data for this job title / company size)
            with st.spinner("Generating chart..."):
                chart_base64 = make_prediction_chart(inputs, salary, df=context_df)

            # Step 5 — display results
            col_left, col_right = st.columns([1, 1])

            with col_left:
                st.subheader("📝 Analysis")
                if narrative:
                    st.write(narrative)
                else:
                    st.info("Narrative unavailable — Ollama may not be running.")

            with col_right:
                st.subheader("📊 Salary Context")
                image = decode_chart(chart_base64)
                if image:
                    st.image(image, use_column_width=True)

            # Step 6 — save to Supabase (always, even if LLM failed)
            save_prediction(
                inputs,
                salary,
                narrative or "",
                chart_base64 or "",
            )
            st.caption("✅ Saved to Supabase")

    # ── Tab 2: All predictions table ─────────────────────────
    with tab2:

        st.subheader("All Predictions")
        st.caption("Every prediction made through the dashboard.")

        # Reload button — clears cache so latest data shows
        if st.button("🔄 Refresh data"):
            st.cache_data.clear()

        df = load_all_predictions()

        if df.empty:
            st.info(
                "No predictions yet. "
                "Go to the 'Make a Prediction' tab to generate your first one."
            )
        else:
            # Summary metrics
            sal = df["predicted_salary_usd"].dropna()
            m1, m2, m3 = st.columns(3)
            m1.metric("Min Salary",  f"${sal.min():,.0f}")
            m2.metric("Avg Salary",  f"${sal.mean():,.0f}")
            m3.metric("Max Salary",  f"${sal.max():,.0f}")

            st.divider()

            # Filters
            fc1, fc2, fc3 = st.columns(3)

            with fc1:
                exp_opts = ["All"] + sorted(df["experience_level"].dropna().unique().tolist())
                sel_exp  = st.selectbox("Experience", exp_opts, key="t2_exp")

            with fc2:
                title_opts = ["All"] + sorted(df["job_title"].dropna().unique().tolist())
                sel_title  = st.selectbox("Job title", title_opts, key="t2_title")

            with fc3:
                size_opts = ["All"] + sorted(df["company_size"].dropna().unique().tolist())
                sel_size  = st.selectbox("Company size", size_opts, key="t2_size")

            # Apply filters
            filtered = df.copy()
            if sel_exp   != "All":
                filtered = filtered[filtered["experience_level"] == sel_exp]
            if sel_title != "All":
                filtered = filtered[filtered["job_title"] == sel_title]
            if sel_size  != "All":
                filtered = filtered[filtered["company_size"] == sel_size]

            if filtered.empty:
                st.info("No predictions match the selected filters.")
            else:
                st.caption(f"Showing {len(filtered):,} predictions")

                # Pick columns to display
                show_cols = [
                    "job_title", "experience_level", "employment_type",
                    "company_size", "remote_ratio", "predicted_salary_usd",
                    "created_at",
                ]
                show_cols = [c for c in show_cols if c in filtered.columns]
                display   = filtered[show_cols].copy()

                display = display.rename(columns={
                    "job_title":             "Job Title",
                    "experience_level":      "Experience",
                    "employment_type":       "Employment",
                    "company_size":          "Company Size",
                    "remote_ratio":          "Remote %",
                    "predicted_salary_usd":  "Predicted Salary",
                    "created_at":            "Date",
                })

                display["Predicted Salary"] = display["Predicted Salary"].apply(
                    lambda x: f"${x:,.0f}" if pd.notna(x) else "N/A"
                )

                st.dataframe(display, use_container_width=True, hide_index=True)


if __name__ == "__main__":
    main()