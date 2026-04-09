# src/salary_predictor/dashboard/app.py
#
# Streamlit dashboard — two tabs:
#   Tab 1 — Prediction results (salary + LLM narrative + two charts)
#   Tab 2 — All predictions from Supabase with metrics and a bar chart
#
# Form inputs live in the sidebar so the main area is always the results canvas.
# Results are stored in st.session_state so they survive Streamlit reruns.

import base64
import logging
import os
import sys
from io import BytesIO

import pandas as pd
import plotly.express as px
import requests
import streamlit as st
from dotenv import load_dotenv
from PIL import Image

from salary_predictor.db.supabase_client import get_client
from salary_predictor.llm.analyst import get_narrative
from salary_predictor.llm.charts import make_prediction_chart, make_remote_chart

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

# ── Custom CSS ─────────────────────────────────────────────────
st.markdown("""
<style>
/* Salary metric card */
[data-testid="metric-container"] {
    background: linear-gradient(135deg, #0f2027, #203a43, #2c5364);
    border-radius: 12px;
    padding: 1.2rem 1.5rem;
    color: white !important;
}
[data-testid="metric-container"] label {
    color: #a0c4d8 !important;
    font-size: 0.85rem !important;
    letter-spacing: 0.05em;
    text-transform: uppercase;
}
[data-testid="stMetricValue"] {
    color: white !important;
    font-size: 2.4rem !important;
    font-weight: 700 !important;
}

/* Narrative box */
.narrative-box {
    background: #f8fafc;
    border-left: 4px solid #2c5364;
    border-radius: 0 8px 8px 0;
    padding: 1.2rem 1.5rem;
    line-height: 1.75;
    color: #1e293b;
    font-size: 0.97rem;
}

/* Instruction placeholder */
.placeholder-box {
    background: #f1f5f9;
    border-radius: 12px;
    padding: 3rem 2rem;
    text-align: center;
    color: #64748b;
}

/* Sidebar label */
section[data-testid="stSidebar"] .stSelectbox label,
section[data-testid="stSidebar"] .stSlider label {
    font-size: 0.82rem;
    color: #475569;
    font-weight: 600;
    letter-spacing: 0.03em;
    text-transform: uppercase;
}
</style>
""", unsafe_allow_html=True)


# ── Helpers ────────────────────────────────────────────────────

def decode_chart(chart_base64: str) -> Image.Image | None:
    """Decode a base64 string back into a PIL image."""
    try:
        return Image.open(BytesIO(base64.b64decode(chart_base64)))
    except Exception as error:
        logger.error("Failed to decode chart: %s", error)
        return None


def call_predict_api(inputs: dict) -> float | None:
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


def save_prediction(
    inputs: dict,
    salary: float,
    narrative: str,
    chart_base64: str,
    remote_chart_base64: str,
) -> None:
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
            "remote_chart_base64":  remote_chart_base64,
        }).execute()
        logger.info("Prediction saved to Supabase")
    except Exception as error:
        logger.error("Failed to save to Supabase: %s", error)


@st.cache_data(ttl=60)
def load_all_predictions() -> pd.DataFrame:
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


@st.cache_data(ttl=3600)
def fetch_valid_values() -> dict:
    try:
        response = requests.get(f"{API_BASE_URL}/valid-values", timeout=10)
        response.raise_for_status()
        return response.json()
    except Exception:
        return {
            "job_title":          ["Data Scientist", "Data Engineer",
                                   "Data Analyst", "Machine Learning Engineer",
                                   "Research Scientist"],
            "employee_residence": ["US", "GB", "IN", "DE", "CA"],
            "company_location":   ["US", "GB", "IN", "DE", "CA"],
        }


# ── Sidebar — form inputs ──────────────────────────────────────

def render_sidebar() -> tuple[dict, bool]:
    """Render form in sidebar. Returns (inputs dict, predict_clicked bool)."""
    with st.sidebar:
        st.markdown("## ⚙️ Job Details")
        st.caption("Fill in the details and click Predict.")
        st.divider()

        valid = fetch_valid_values()

        job_title = st.selectbox(
            "Job Title",
            options=valid.get("job_title", []),
        )
        experience_level = st.selectbox(
            "Experience Level",
            options=["EN", "MI", "SE", "EX"],
            format_func=lambda x: {
                "EN": "Entry-level",
                "MI": "Mid-level",
                "SE": "Senior",
                "EX": "Executive",
            }[x],
        )
        employment_type = st.selectbox(
            "Employment Type",
            options=["FT", "PT", "CT", "FL"],
            format_func=lambda x: {
                "FT": "Full-time",
                "PT": "Part-time",
                "CT": "Contract",
                "FL": "Freelance",
            }[x],
        )
        company_size = st.selectbox(
            "Company Size",
            options=["S", "M", "L"],
            format_func=lambda x: {
                "S": "Small  (< 50 employees)",
                "M": "Medium (50–250 employees)",
                "L": "Large  (250+ employees)",
            }[x],
        )
        remote_ratio = st.selectbox(
            "Work Arrangement",
            options=[0, 50, 100],
            format_func=lambda x: {
                0:   "On-site",
                50:  "Hybrid",
                100: "Fully Remote",
            }[x],
        )
        work_year = st.selectbox(
            "Work Year",
            options=[2020, 2021, 2022, 2023],
            index=3,
        )
        employee_residence = st.selectbox(
            "Employee Residence",
            options=valid.get("employee_residence", ["US"]),
        )
        company_location = st.selectbox(
            "Company Location",
            options=valid.get("company_location", ["US"]),
        )

        st.divider()
        clicked = st.button(
            "🔮 Predict Salary",
            type="primary",
            use_container_width=True,
        )

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
    return inputs, clicked


# ── Main ──────────────────────────────────────────────────────

def main() -> None:

    inputs, predict_clicked = render_sidebar()

    # Page header
    st.title("💼 Data Science Salary Predictor")
    st.caption("Random Forest model · LLM analysis by Ollama · Data stored in Supabase")

    tab1, tab2 = st.tabs(["🔮 Prediction", "📊 All Predictions"])

    # ── Tab 1: Prediction ─────────────────────────────────────
    with tab1:

        if predict_clicked:

            # Step 1 — call the API
            with st.spinner("Predicting salary..."):
                salary = call_predict_api(inputs)

            if salary is None:
                st.stop()

            # Step 2 — load Supabase data for real market context
            context_df = load_all_predictions()
            context_df = context_df if not context_df.empty else None

            # Step 3 — LLM narrative
            with st.spinner("Generating analysis — this takes ~30 seconds..."):
                try:
                    narrative = get_narrative(inputs, salary, df=context_df)
                except RuntimeError as error:
                    st.error(f"LLM analysis failed: {error}")
                    narrative = None

            # Step 4 — charts
            with st.spinner("Building charts..."):
                chart_b64  = make_prediction_chart(inputs, salary, df=context_df)
                remote_b64 = make_remote_chart(inputs, salary, df=context_df)

            # Step 5 — store in session_state so results survive reruns
            st.session_state["result"] = {
                "salary":      salary,
                "narrative":   narrative,
                "chart_b64":   chart_b64,
                "remote_b64":  remote_b64,
                "inputs":      inputs,
            }

            # Step 6 — save to Supabase
            save_prediction(inputs, salary, narrative or "", chart_b64 or "", remote_b64 or "")
            st.cache_data.clear()   # refresh Tab 2 metrics after saving

        # ── Display results (from session_state) ──────────────
        if "result" in st.session_state:
            res = st.session_state["result"]

            # Salary metric — full width, prominent
            st.metric("Predicted Annual Salary", f"${res['salary']:,.0f}")
            st.divider()

            # Narrative left | Charts right
            col_left, col_right = st.columns([1, 1], gap="large")

            with col_left:
                st.markdown("#### 📝 Salary Analysis")
                if res["narrative"]:
                    st.markdown(
                        f"<div class='narrative-box'>{res['narrative']}</div>",
                        unsafe_allow_html=True,
                    )
                else:
                    st.info("Narrative unavailable — start Ollama and predict again.")

            with col_right:
                st.markdown("#### 📊 Salary by Experience Level")
                img1 = decode_chart(res["chart_b64"])
                if img1:
                    st.image(img1, use_container_width=True)

                st.markdown("#### 🌍 Salary by Work Arrangement")
                img2 = decode_chart(res["remote_b64"])
                if img2:
                    st.image(img2, use_container_width=True)

        else:
            # No prediction yet — show instructions
            st.markdown("""
<div class='placeholder-box'>
    <h3 style='margin-bottom:0.5rem;'>👈 Fill in the job details on the left</h3>
    <p style='margin:0; font-size:1rem;'>
        Select a job title, experience level, company setup, and work arrangement —
        then click <strong>Predict Salary</strong> to get an instant estimate
        with AI-powered analysis and charts.
    </p>
</div>
""", unsafe_allow_html=True)

    # ── Tab 2: All Predictions ────────────────────────────────
    with tab2:

        col_head, col_btn = st.columns([5, 1])
        with col_head:
            st.markdown("#### All Predictions")
            st.caption("Every prediction made through the dashboard, stored in Supabase.")
        with col_btn:
            if st.button("🔄 Refresh", use_container_width=True):
                st.cache_data.clear()

        df = load_all_predictions()

        if df.empty:
            st.info("No predictions yet — make your first prediction using the sidebar.")
        else:
            # Summary metrics
            sal = df["predicted_salary_usd"].dropna()
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Total Predictions", f"{len(df):,}")
            m2.metric("Min Salary",  f"${sal.min():,.0f}")
            m3.metric("Avg Salary",  f"${sal.mean():,.0f}")
            m4.metric("Max Salary",  f"${sal.max():,.0f}")

            st.divider()

            # Bar chart — avg salary by experience level
            exp_order  = {"EN": "Entry", "MI": "Mid", "SE": "Senior", "EX": "Executive"}
            chart_data = (
                df.dropna(subset=["experience_level", "predicted_salary_usd"])
                  .assign(Level=lambda d: d["experience_level"].map(exp_order))
                  .groupby("Level", as_index=False)["predicted_salary_usd"]
                  .mean()
                  .rename(columns={"predicted_salary_usd": "Avg Salary"})
            )
            level_order = ["Entry", "Mid", "Senior", "Executive"]
            chart_data["Level"] = pd.Categorical(chart_data["Level"], categories=level_order, ordered=True)
            chart_data = chart_data.sort_values("Level")

            fig = px.bar(
                chart_data,
                x="Level",
                y="Avg Salary",
                text=chart_data["Avg Salary"].apply(lambda v: f"${v:,.0f}"),
                color="Avg Salary",
                color_continuous_scale="Blues",
                labels={"Avg Salary": "Average Predicted Salary (USD)"},
                title="Average Predicted Salary by Experience Level",
            )
            fig.update_traces(textposition="outside")
            fig.update_layout(
                coloraxis_showscale=False,
                plot_bgcolor="white",
                yaxis=dict(tickprefix="$", tickformat=","),
                title_font_size=15,
                margin=dict(t=50, b=30),
            )
            st.plotly_chart(fig, use_container_width=True)

            st.divider()

            # Filters
            fc1, fc2, fc3 = st.columns(3)
            with fc1:
                exp_opts = ["All"] + sorted(df["experience_level"].dropna().unique().tolist())
                sel_exp  = st.selectbox("Filter by Experience", exp_opts, key="t2_exp")
            with fc2:
                title_opts = ["All"] + sorted(df["job_title"].dropna().unique().tolist())
                sel_title  = st.selectbox("Filter by Job Title", title_opts, key="t2_title")
            with fc3:
                size_opts = ["All"] + sorted(df["company_size"].dropna().unique().tolist())
                sel_size  = st.selectbox("Filter by Company Size", size_opts, key="t2_size")

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
                st.caption(f"Showing {len(filtered):,} of {len(df):,} predictions")

                show_cols = [
                    "job_title", "experience_level", "employment_type",
                    "company_size", "remote_ratio", "predicted_salary_usd",
                    "created_at",
                ]
                show_cols = [c for c in show_cols if c in filtered.columns]
                display   = filtered[show_cols].copy()

                if "created_at" in display.columns:
                    display["created_at"] = pd.to_datetime(
                        display["created_at"], errors="coerce"
                    ).dt.strftime("%b %d, %Y %H:%M")

                display = display.rename(columns={
                    "job_title":            "Job Title",
                    "experience_level":     "Experience",
                    "employment_type":      "Employment",
                    "company_size":         "Company Size",
                    "remote_ratio":         "Remote %",
                    "predicted_salary_usd": "Predicted Salary",
                    "created_at":           "Date",
                })
                display["Predicted Salary"] = display["Predicted Salary"].apply(
                    lambda x: f"${x:,.0f}" if pd.notna(x) else "N/A"
                )

                st.dataframe(display, use_container_width=True, hide_index=True)


if __name__ == "__main__":
    main()
