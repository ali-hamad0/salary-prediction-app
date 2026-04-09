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
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
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

/* ── Prediction Tab ─────────────────────────────────── */

/* Hero salary card */
.pred-hero {
    background: linear-gradient(135deg, #0f2027 0%, #203a43 55%, #2c5364 100%);
    border-radius: 18px;
    padding: 2.2rem 2.5rem 1.8rem;
    text-align: center;
    margin-bottom: 0.6rem;
}
.pred-hero-label {
    font-size: 0.72rem;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    color: #94a3b8;
    margin-bottom: 0.45rem;
}
.pred-hero-salary {
    font-size: 4rem;
    font-weight: 900;
    color: #38bdf8;
    line-height: 1.05;
    margin-bottom: 0.5rem;
    letter-spacing: -0.02em;
}
.pred-hero-role {
    font-size: 0.92rem;
    color: #cbd5e1;
    margin-bottom: 0.2rem;
}

/* Market position bar */
.mkt-wrap { margin: 0.6rem 0 1.2rem; }
.mkt-labels {
    display: flex;
    justify-content: space-between;
    font-size: 0.7rem;
    color: #94a3b8;
    margin-bottom: 0.3rem;
    padding: 0 2px;
}
.mkt-track {
    height: 8px;
    background: #e2e8f0;
    border-radius: 99px;
    position: relative;
    overflow: visible;
}
.mkt-fill {
    height: 100%;
    background: linear-gradient(90deg, #bae6fd, #0284c7);
    border-radius: 99px;
}
.mkt-dot {
    position: absolute;
    top: -5px;
    width: 18px;
    height: 18px;
    background: #ef4444;
    border: 2.5px solid white;
    border-radius: 50%;
    transform: translateX(-50%);
    box-shadow: 0 2px 6px rgba(239,68,68,0.45);
}
.mkt-dot-label {
    position: absolute;
    top: -26px;
    transform: translateX(-50%);
    font-size: 0.7rem;
    font-weight: 700;
    color: #ef4444;
    white-space: nowrap;
}
.mkt-ticks {
    display: flex;
    justify-content: space-between;
    font-size: 0.68rem;
    color: #94a3b8;
    margin-top: 0.35rem;
    padding: 0 2px;
}

/* Input chips */
.chips-row { display: flex; flex-wrap: wrap; gap: 0.35rem; margin: 0.5rem 0 1rem; }
.chip {
    background: rgba(255,255,255,0.08);
    border: 1px solid rgba(255,255,255,0.15);
    border-radius: 99px;
    padding: 0.28rem 0.8rem;
    font-size: 0.76rem;
    color: #cbd5e1;
    font-weight: 500;
}

/* Narrative box — upgraded */
.narrative-box {
    background: #f8fafc;
    border-left: 4px solid #2c5364;
    border-radius: 0 12px 12px 0;
    padding: 1.3rem 1.5rem;
    line-height: 1.8;
    color: #1e293b;
    font-size: 0.94rem;
}
.narrative-header {
    font-size: 0.72rem;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    color: #64748b;
    font-weight: 700;
    margin-bottom: 0.6rem;
}

/* Step cards — how-it-works placeholder */
.steps-row { display: flex; gap: 1rem; margin: 1.2rem 0 1.5rem; }
.step-card {
    flex: 1;
    background: white;
    border: 1px solid #e2e8f0;
    border-radius: 14px;
    padding: 1.4rem 1.1rem 1.2rem;
    text-align: center;
    box-shadow: 0 1px 4px rgba(0,0,0,0.05);
}
.step-num {
    width: 26px; height: 26px;
    background: linear-gradient(135deg,#203a43,#2c5364);
    color: #38bdf8;
    border-radius: 50%;
    font-size: 0.75rem;
    font-weight: 800;
    display: flex;
    align-items: center;
    justify-content: center;
    margin: 0 auto 0.6rem;
}
.step-icon { font-size: 1.7rem; margin-bottom: 0.45rem; line-height: 1; }
.step-title { font-weight: 700; color: #1e293b; font-size: 0.88rem; margin-bottom: 0.25rem; }
.step-desc { font-size: 0.77rem; color: #64748b; line-height: 1.55; }
.powered-strip {
    background: linear-gradient(135deg,#eff6ff,#dbeafe);
    border-radius: 10px;
    padding: 0.85rem 1.5rem;
    text-align: center;
    color: #1e40af;
    font-size: 0.85rem;
    margin-top: 0.5rem;
}

/* ── EDA Tab ─────────────────────────────────────────── */

/* Dark gradient section banner */
.eda-banner {
    background: linear-gradient(135deg, #0f2027 0%, #203a43 55%, #2c5364 100%);
    border-radius: 12px;
    padding: 0.95rem 1.4rem;
    margin: 1.4rem 0 0.5rem;
    display: flex;
    align-items: flex-start;
    gap: 0.75rem;
}
.eda-icon  { font-size: 1.5rem; line-height: 1.1; }
.eda-title { margin: 0; font-size: 1rem; font-weight: 700; color: #fff; letter-spacing: -0.01em; }
.eda-sub   { margin: 0.12rem 0 0; font-size: 0.8rem; color: #94a3b8; }

/* Blue insight callout */
.callout {
    background: linear-gradient(135deg, #eff6ff, #dbeafe);
    border-left: 4px solid #2563eb;
    border-radius: 0 10px 10px 0;
    padding: 0.8rem 1.15rem;
    margin-bottom: 0.35rem;
    color: #1e3a5f;
    font-size: 0.88rem;
    line-height: 1.65;
}
.callout b { color: #1d4ed8; }

/* Hero stat chips */
.hstat {
    background: rgba(255,255,255,0.1);
    border-radius: 9px;
    padding: 0.65rem 1rem;
    display: inline-block;
    margin: 0 0.45rem 0.45rem 0;
}
.hstat-v { font-size: 1.45rem; font-weight: 800; color: #38bdf8; line-height: 1.1; }
.hstat-l { font-size: 0.68rem; color: #94a3b8; text-transform: uppercase; letter-spacing: 0.06em; margin-top: 2px; }

/* Takeaway cards */
.tcard {
    border-radius: 12px;
    padding: 1.1rem 1.25rem;
    line-height: 1.6;
    font-size: 0.88rem;
    height: 100%;
}
.tcard-blue  { background: linear-gradient(135deg,#eff6ff,#dbeafe); border-left: 4px solid #2563eb; color: #1e3a5f; }
.tcard-green { background: linear-gradient(135deg,#f0fdf4,#dcfce7); border-left: 4px solid #16a34a; color: #14532d; }
.tcard-amber { background: linear-gradient(135deg,#fffbeb,#fef3c7); border-left: 4px solid #d97706; color: #78350f; }
.tcard b { font-weight: 700; }
.tcard-num { font-size: 1.9rem; font-weight: 800; display: block; margin-bottom: 0.1rem; }
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
            timeout=60,  # Render free tier cold start can take up to 60s
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


@st.cache_data(ttl=300)
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
    """
    Load valid categorical values directly from the trained model encoders.

    The API only exposes /health and /predict — valid values come from
    the model package which is installed alongside the dashboard.
    """
    from salary_predictor.model import get_valid_values, load_model, is_model_loaded

    if not is_model_loaded():
        load_model()

    return {
        "job_title":          get_valid_values("job_title"),
        "employee_residence": get_valid_values("employee_residence"),
        "company_location":   get_valid_values("company_location"),
    }


@st.cache_data(ttl=86400)
def load_raw_data() -> pd.DataFrame:
    """Load & label the raw salary CSV for EDA charts. Cached for 24 h."""
    csv = Path(__file__).resolve().parents[3] / "data" / "raw" / "ds_salaries.csv"
    df  = pd.read_csv(csv)
    df["Experience"]       = df["experience_level"].map({"EN":"Entry","MI":"Mid","SE":"Senior","EX":"Executive"})
    df["Work Arrangement"] = df["remote_ratio"].map({0:"On-site",50:"Hybrid",100:"Fully Remote"})
    df["Company Size"]     = df["company_size"].map({"S":"Small","M":"Medium","L":"Large"})
    return df


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

    tab1, tab2, tab3 = st.tabs(["🔮 Prediction", "📊 All Predictions", "📈 Data Insights"])

    # ── Tab 1: Prediction ─────────────────────────────────────
    with tab1:

        if predict_clicked:

            # Step 1 — call the API
            with st.spinner("Predicting salary — first request may take up to 60s to wake the API..."):
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
            load_all_predictions.clear()  # only bust the predictions cache, not the model

        # ── Display results (from session_state) ──────────────
        if "result" in st.session_state:
            res   = st.session_state["result"]
            inp   = res["inputs"]
            sal   = res["salary"]

            # Readable labels for chips
            _EXP    = {"EN":"Entry-level","MI":"Mid-level","SE":"Senior","EX":"Executive"}
            _SIZE   = {"S":"Small co.","M":"Medium co.","L":"Large co."}
            _REMOTE = {0:"On-site",50:"Hybrid",100:"Fully Remote"}
            _EMPL   = {"FT":"Full-time","PT":"Part-time","CT":"Contract","FL":"Freelance"}

            exp_lbl    = _EXP.get(inp["experience_level"], inp["experience_level"])
            size_lbl   = _SIZE.get(inp["company_size"], inp["company_size"])
            remote_lbl = _REMOTE.get(inp["remote_ratio"], str(inp["remote_ratio"]))
            empl_lbl   = _EMPL.get(inp["employment_type"], inp["employment_type"])

            # ── Hero card ──────────────────────────────────────
            st.markdown(f"""
<div class="pred-hero">
  <div class="pred-hero-label">Predicted Annual Salary</div>
  <div class="pred-hero-salary">${sal:,.0f}</div>
  <div class="pred-hero-role">{inp['job_title']} &nbsp;·&nbsp; {exp_lbl} &nbsp;·&nbsp; {size_lbl} &nbsp;·&nbsp; {remote_lbl}</div>
  <div class="chips-row" style="justify-content:center;margin-top:1rem;">
    <span class="chip">📋 {inp['job_title']}</span>
    <span class="chip">🪜 {exp_lbl}</span>
    <span class="chip">🏢 {size_lbl}</span>
    <span class="chip">💻 {remote_lbl}</span>
    <span class="chip">⏰ {empl_lbl}</span>
    <span class="chip">📅 {inp['work_year']}</span>
    <span class="chip">🌍 {inp['employee_residence']}</span>
    <span class="chip">🏳️ {inp['company_location']}</span>
  </div>
</div>""", unsafe_allow_html=True)

            # ── Market position bar ────────────────────────────
            # Reference percentiles from the training dataset
            P10, P25, P50, P75, P90 = 40_000, 62_726, 101_570, 150_000, 215_000
            pct = round((sal - P10) / (P90 - P10) * 100, 1)
            pct = max(3, min(96, pct))
            p25_pos = round((P25 - P10) / (P90 - P10) * 100, 1)
            p50_pos = round((P50 - P10) / (P90 - P10) * 100, 1)
            p75_pos = round((P75 - P10) / (P90 - P10) * 100, 1)

            st.markdown(f"""
<div class="mkt-wrap">
  <div class="mkt-labels">
    <span>Market position (training data percentiles)</span>
    <span style="color:#ef4444;font-weight:700;">● Your estimate</span>
  </div>
  <div class="mkt-track">
    <div class="mkt-fill" style="width:{pct}%"></div>
    <div class="mkt-dot" style="left:{pct}%">
      <div class="mkt-dot-label">${sal/1000:.0f}k</div>
    </div>
    <div style="position:absolute;top:11px;left:{p25_pos}%;transform:translateX(-50%);
         font-size:0.63rem;color:#94a3b8;">P25</div>
    <div style="position:absolute;top:11px;left:{p50_pos}%;transform:translateX(-50%);
         font-size:0.63rem;color:#94a3b8;">P50</div>
    <div style="position:absolute;top:11px;left:{p75_pos}%;transform:translateX(-50%);
         font-size:0.63rem;color:#94a3b8;">P75</div>
  </div>
  <div class="mkt-ticks">
    <span>$40k (P10)</span><span>$63k (P25)</span>
    <span>$102k (P50)</span><span>$150k (P75)</span><span>$215k (P90)</span>
  </div>
</div>""", unsafe_allow_html=True)

            # ── Narrative  |  Charts ───────────────────────────
            col_left, col_right = st.columns([1.1, 1], gap="large")

            with col_left:
                st.markdown("""
<div class="narrative-header">📝 &nbsp;AI Salary Analysis</div>""",
                    unsafe_allow_html=True)
                if res["narrative"]:
                    st.markdown(
                        f"<div class='narrative-box'>{res['narrative']}</div>",
                        unsafe_allow_html=True,
                    )
                else:
                    st.info("Narrative unavailable — make sure Ollama is running.")

            with col_right:
                st.markdown("""<div class="narrative-header">📊 &nbsp;Salary by Experience Level</div>""",
                    unsafe_allow_html=True)
                img1 = decode_chart(res["chart_b64"])
                if img1:
                    st.image(img1, use_container_width=True)

                st.markdown("""<div class="narrative-header" style="margin-top:0.8rem;">🌍 &nbsp;Salary by Work Arrangement</div>""",
                    unsafe_allow_html=True)
                img2 = decode_chart(res["remote_b64"])
                if img2:
                    st.image(img2, use_container_width=True)

        else:
            # ── Placeholder — how it works ─────────────────────
            st.markdown("""
<div class="steps-row">
  <div class="step-card">
    <div class="step-num">1</div>
    <div class="step-icon">👈</div>
    <div class="step-title">Fill in your details</div>
    <div class="step-desc">Choose your job title, experience level,
      company size and work arrangement in the sidebar.</div>
  </div>
  <div class="step-card">
    <div class="step-num">2</div>
    <div class="step-icon">🔮</div>
    <div class="step-title">Get your prediction</div>
    <div class="step-desc">Our Random Forest model — trained on 607 real
      data science salaries — estimates your annual pay instantly.</div>
  </div>
  <div class="step-card">
    <div class="step-num">3</div>
    <div class="step-icon">📊</div>
    <div class="step-title">See the full picture</div>
    <div class="step-desc">An AI narrative + two context charts show
      exactly where you stand in the market.</div>
  </div>
</div>
<div class="powered-strip">
  ⚡ Powered by <strong>607 real salaries</strong> across
  <strong>50 job titles</strong> and <strong>46 countries</strong> &nbsp;·&nbsp;
  Random Forest · Ollama LLM · Supabase
</div>""", unsafe_allow_html=True)

    # ── Tab 2: All Predictions ────────────────────────────────
    with tab2:

        col_head, col_btn = st.columns([5, 1])
        with col_head:
            st.markdown("#### All Predictions")
            st.caption("Every prediction made through the dashboard, stored in Supabase.")
        with col_btn:
            if st.button("🔄 Refresh", use_container_width=True):
                load_all_predictions.clear()

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


    # ── Tab 3 : Data Insights ─────────────────────────────────
    with tab3:
        raw = load_raw_data()

        # ── shared chart theme ─────────────────────────────────
        _T = dict(
            plot_bgcolor="white", paper_bgcolor="white",
            font=dict(family="Inter,sans-serif", size=12, color="#374151"),
            title_font=dict(size=14, color="#111827"),
            hoverlabel=dict(bgcolor="white", font_size=12, bordercolor="#e5e7eb"),
            margin=dict(t=50, b=30, l=10, r=15),
        )
        EXP_ORDER  = ["Entry","Mid","Senior","Executive"]
        SIZE_ORDER = ["Small","Medium","Large"]
        ARR_ORDER  = ["On-site","Hybrid","Fully Remote"]
        BLUES      = ["#bae6fd","#7dd3fc","#38bdf8","#0ea5e9","#0284c7","#0369a1"]

        # ── pre-compute stats used in callouts ─────────────────
        med_all    = int(raw["salary_in_usd"].median())
        p25        = int(raw["salary_in_usd"].quantile(.25))
        p75        = int(raw["salary_in_usd"].quantile(.75))
        entry_med  = int(raw[raw["experience_level"]=="EN"]["salary_in_usd"].median())
        senior_med = int(raw[raw["experience_level"]=="SE"]["salary_in_usd"].median())
        exec_med   = int(raw[raw["experience_level"]=="EX"]["salary_in_usd"].median())
        mult       = round(senior_med / entry_med, 1)
        remote_med = int(raw[raw["remote_ratio"]==100]["salary_in_usd"].median())
        onsite_med = int(raw[raw["remote_ratio"]==0]["salary_in_usd"].median())
        rem_pct    = round((remote_med - onsite_med) / onsite_med * 100, 1)
        yr_min     = int(raw["work_year"].min())
        yr_max     = int(raw["work_year"].max())
        sal_start  = int(raw[raw["work_year"]==yr_min]["salary_in_usd"].median())
        sal_end    = int(raw[raw["work_year"]==yr_max]["salary_in_usd"].median())
        growth     = round((sal_end - sal_start) / sal_start * 100, 1)

        title_stats = (
            raw.groupby("job_title")["salary_in_usd"]
            .agg(median="median", n="count").query("n >= 5")
            .nlargest(15,"median").reset_index()
        )
        top_role   = title_stats.iloc[-1]["job_title"]
        top_median = int(title_stats.iloc[-1]["median"])

        # ── HERO ──────────────────────────────────────────────
        st.markdown(f"""
<div style="background:linear-gradient(135deg,#0f2027 0%,#203a43 55%,#2c5364 100%);
     border-radius:16px;padding:1.7rem 2rem 1.4rem;margin-bottom:0.2rem;">
  <h2 style="margin:0 0 0.3rem;font-size:1.5rem;font-weight:800;color:#fff;">
    📊 Data Science Salary Explorer
  </h2>
  <p style="margin:0 0 1.2rem;color:#94a3b8;font-size:0.9rem;">
    What does the data really tell us about who earns what — and why?
  </p>
  <div style="display:flex;flex-wrap:wrap;gap:0.8rem;">
    <div class="hstat"><div class="hstat-v">{len(raw):,}</div><div class="hstat-l">Records</div></div>
    <div class="hstat"><div class="hstat-v">{raw["job_title"].nunique()}</div><div class="hstat-l">Job titles</div></div>
    <div class="hstat"><div class="hstat-v">{raw["employee_residence"].nunique()}</div><div class="hstat-l">Countries</div></div>
    <div class="hstat"><div class="hstat-v">{yr_min}–{yr_max}</div><div class="hstat-l">Years</div></div>
    <div class="hstat"><div class="hstat-v">${med_all:,}</div><div class="hstat-l">Median salary</div></div>
  </div>
</div>""", unsafe_allow_html=True)

        # ══════════════════════════════════════════════════════
        # 1 · SALARY DISTRIBUTION
        # ══════════════════════════════════════════════════════
        st.markdown("""<div class="eda-banner">
  <div class="eda-icon">📐</div>
  <div><p class="eda-title">The salary landscape</p>
  <p class="eda-sub">How are salaries spread across all data science roles?</p></div>
</div>""", unsafe_allow_html=True)

        st.markdown(f"""<div class="callout">
Most salaries sit between <b>${p25:,}</b> and <b>${p75:,}</b> (the middle 50%).
The median is <b>${med_all:,}</b> — well above the mean because a long right tail
of outliers pulls the average up. A handful of roles hit <b>$600k</b>.
</div>""", unsafe_allow_html=True)

        fig_hist = px.histogram(
            raw, x="salary_in_usd", nbins=65,
            color_discrete_sequence=["#38bdf8"],
            labels={"salary_in_usd":"Annual Salary (USD)"},
            title="Salary Distribution — All Roles",
        )
        fig_hist.update_traces(marker_line_width=0, opacity=0.82)
        fig_hist.add_vline(x=med_all, line_dash="dash", line_color="#ef4444", line_width=2,
            annotation_text=f"Median ${med_all:,}", annotation_font_color="#ef4444",
            annotation_position="top right")
        fig_hist.add_vrect(x0=p25, x1=p75, fillcolor="#dbeafe", opacity=0.22, line_width=0,
            annotation_text="Middle 50%", annotation_position="top left",
            annotation_font_color="#2563eb", annotation_font_size=11)
        fig_hist.update_layout(**_T, bargap=0.04,
            xaxis=dict(tickprefix="$", tickformat=",", title="Annual Salary (USD)"),
            yaxis_title="Number of Roles")
        st.plotly_chart(fig_hist, use_container_width=True)

        # ══════════════════════════════════════════════════════
        # 2 · EXPERIENCE LEVEL — violin
        # ══════════════════════════════════════════════════════
        st.markdown("""<div class="eda-banner">
  <div class="eda-icon">🪜</div>
  <div><p class="eda-title">Experience is the single biggest lever</p>
  <p class="eda-sub">Nothing else in the data moves salaries this far, this consistently.</p></div>
</div>""", unsafe_allow_html=True)

        st.markdown(f"""<div class="callout">
Senior roles earn <b>{mult}× more than Entry-level</b>
(${entry_med:,} → ${senior_med:,}). Executive pay spreads <i>wider</i> than any other
group — leadership comp varies heavily by company, not just seniority.
</div>""", unsafe_allow_html=True)

        fig_vio = px.violin(
            raw[raw["Experience"].notna()],
            x="Experience", y="salary_in_usd",
            color="Experience", box=True, points=False,
            category_orders={"Experience": EXP_ORDER},
            color_discrete_sequence=BLUES[1:],
            labels={"salary_in_usd":"Annual Salary (USD)", "Experience":""},
            title="Salary Distribution by Experience Level",
        )
        fig_vio.update_layout(**_T, showlegend=False,
                              yaxis=dict(tickprefix="$", tickformat=","))
        for lvl in EXP_ORDER:
            m = int(raw[raw["Experience"]==lvl]["salary_in_usd"].median())
            fig_vio.add_annotation(x=lvl, y=m, text=f"${m//1000}k",
                showarrow=False, yshift=16,
                font=dict(size=11, color="#111827", family="Inter,sans-serif"))
        st.plotly_chart(fig_vio, use_container_width=True)

        # ══════════════════════════════════════════════════════
        # 3 · COMPANY SIZE  &  REMOTE WORK — side by side
        # ══════════════════════════════════════════════════════
        st.markdown("""<div class="eda-banner">
  <div class="eda-icon">🏢</div>
  <div><p class="eda-title">Company size & remote work — do they actually matter?</p>
  <p class="eda-sub">Two factors everyone talks about. Here's what the data says.</p></div>
</div>""", unsafe_allow_html=True)

        ca, cb = st.columns(2, gap="large")
        with ca:
            sign = "+" if rem_pct > 0 else ""
            st.markdown(f"""<div class="callout">
Fully remote workers earn <b>{sign}{rem_pct}%</b> more than on-site
(${onsite_med:,} vs ${remote_med:,}).
Smaller than the headlines suggest — <b>role and seniority dominate.</b>
</div>""", unsafe_allow_html=True)
            fig_rem = px.box(
                raw[raw["Work Arrangement"].notna()],
                x="Work Arrangement", y="salary_in_usd",
                color="Work Arrangement",
                category_orders={"Work Arrangement": ARR_ORDER},
                color_discrete_sequence=["#bae6fd","#38bdf8","#0369a1"],
                labels={"salary_in_usd":"Annual Salary (USD)", "Work Arrangement":""},
                title="Salary by Work Arrangement",
            )
            fig_rem.update_layout(**_T, showlegend=False,
                                  yaxis=dict(tickprefix="$", tickformat=","))
            st.plotly_chart(fig_rem, use_container_width=True)

        with cb:
            lg_med = int(raw[raw["company_size"]=="L"]["salary_in_usd"].median())
            sm_med = int(raw[raw["company_size"]=="S"]["salary_in_usd"].median())
            st.markdown(f"""<div class="callout">
<b>Large companies</b> pay a median of <b>${lg_med:,}</b> vs <b>${sm_med:,}</b>
at small companies. But large-company salaries also spread wider —
top performers earn <i>far</i> more, but so does variance.
</div>""", unsafe_allow_html=True)
            fig_sz = px.box(
                raw[raw["Company Size"].notna()],
                x="Company Size", y="salary_in_usd",
                color="Company Size",
                category_orders={"Company Size": SIZE_ORDER},
                color_discrete_sequence=["#bae6fd","#38bdf8","#0369a1"],
                labels={"salary_in_usd":"Annual Salary (USD)", "Company Size":""},
                title="Salary by Company Size",
            )
            fig_sz.update_layout(**_T, showlegend=False,
                                 yaxis=dict(tickprefix="$", tickformat=","))
            st.plotly_chart(fig_sz, use_container_width=True)

        # ══════════════════════════════════════════════════════
        # 4 · TOP-PAYING ROLES
        # ══════════════════════════════════════════════════════
        st.markdown(f"""<div class="eda-banner">
  <div class="eda-icon">🏆</div>
  <div><p class="eda-title">Which titles actually pay the most?</p>
  <p class="eda-sub">Top 15 roles with ≥ 5 data points, ranked by median salary.</p></div>
</div>""", unsafe_allow_html=True)

        st.markdown(f"""<div class="callout">
<b>{top_role}</b> tops the chart at <b>${top_median:,}</b> median.
Titles with "Principal", "Lead", or "Research" cluster at the top.
"Data Analyst" — the most common entry point — sits near the bottom.
</div>""", unsafe_allow_html=True)

        sorted_t = title_stats.sort_values("median")
        bar_colors = ["#0369a1" if i >= len(sorted_t)-3 else "#38bdf8"
                      for i in range(len(sorted_t))]
        fig_roles = go.Figure(go.Bar(
            x=sorted_t["median"], y=sorted_t["job_title"],
            orientation="h",
            marker=dict(color=bar_colors, line_width=0),
            text=[f"${v:,.0f}" for v in sorted_t["median"]],
            textposition="outside",
            hovertemplate="<b>%{y}</b><br>Median: $%{x:,.0f}<extra></extra>",
        ))
        fig_roles.update_layout(**{**_T, "margin": dict(t=50, b=30, l=10, r=85)},
            title="Top 15 Highest-Paying Data Science Roles (Median)",
            xaxis=dict(tickprefix="$", tickformat=",", title="Median Salary (USD)"),
            yaxis_title="", height=490)
        st.plotly_chart(fig_roles, use_container_width=True)

        # ══════════════════════════════════════════════════════
        # 5 · YEAR-OVER-YEAR TREND
        # ══════════════════════════════════════════════════════
        st.markdown(f"""<div class="eda-banner">
  <div class="eda-icon">📈</div>
  <div><p class="eda-title">Salaries grew {growth}% from {yr_min} to {yr_max}</p>
  <p class="eda-sub">The overall tide rose — but not equally for every level.</p></div>
</div>""", unsafe_allow_html=True)

        st.markdown(f"""<div class="callout">
The overall median jumped from <b>${sal_start:,}</b> ({yr_min}) to
<b>${sal_end:,}</b> ({yr_max}) — a <b>{growth}% rise</b> in just
{yr_max - yr_min} years. Senior and Executive levels pulled ahead the most,
widening the income gap with Entry-level over time.
</div>""", unsafe_allow_html=True)

        yoy = (
            raw[raw["Experience"].notna()]
            .groupby(["work_year","Experience"])["salary_in_usd"]
            .median().reset_index()
            .rename(columns={"salary_in_usd":"Median Salary"})
        )
        fig_yoy = px.area(
            yoy, x="work_year", y="Median Salary", color="Experience",
            line_group="Experience", markers=True,
            category_orders={"Experience": EXP_ORDER},
            color_discrete_sequence=["#bae6fd","#7dd3fc","#0ea5e9","#0369a1"],
            labels={"work_year":"Year","Median Salary":"Median Salary (USD)"},
            title=f"Median Salary by Experience Level — {yr_min} to {yr_max}",
        )
        fig_yoy.update_traces(opacity=0.72)
        fig_yoy.update_layout(**_T,
            yaxis=dict(tickprefix="$", tickformat=","),
            xaxis=dict(tickmode="linear", dtick=1, title="Year"))
        st.plotly_chart(fig_yoy, use_container_width=True)

        # ══════════════════════════════════════════════════════
        # 6 · TOP COUNTRIES (median salary, ≥ 10 records)
        # ══════════════════════════════════════════════════════
        st.markdown("""<div class="eda-banner">
  <div class="eda-icon">🌍</div>
  <div><p class="eda-title">Where in the world do data scientists earn the most?</p>
  <p class="eda-sub">Top countries by median salary (≥ 10 employee records).</p></div>
</div>""", unsafe_allow_html=True)

        country_stats = (
            raw.groupby("employee_residence")["salary_in_usd"]
            .agg(median="median", n="count").query("n >= 10")
            .nlargest(12,"median").reset_index()
            .sort_values("median")
        )
        us_med = int(raw[raw["employee_residence"]=="US"]["salary_in_usd"].median())
        st.markdown(f"""<div class="callout">
The US dominates both in <b>volume</b> ({int(raw["employee_residence"].value_counts()["US"])} records)
and in pay — with a median of <b>${us_med:,}</b>.
Some smaller markets pay even higher medians, but with far fewer data points.
</div>""", unsafe_allow_html=True)

        c_colors = ["#0369a1" if row["employee_residence"]=="US" else "#38bdf8"
                    for _, row in country_stats.iterrows()]
        fig_geo = go.Figure(go.Bar(
            x=country_stats["median"], y=country_stats["employee_residence"],
            orientation="h",
            marker=dict(color=c_colors, line_width=0),
            text=[f"${v:,.0f}" for v in country_stats["median"]],
            textposition="outside",
            hovertemplate="<b>%{y}</b><br>Median: $%{x:,.0f}<extra></extra>",
        ))
        fig_geo.update_layout(**{**_T, "margin": dict(t=50, b=30, l=10, r=85)},
            title="Median Salary by Country (≥ 10 records)",
            xaxis=dict(tickprefix="$", tickformat=",", title="Median Salary (USD)"),
            yaxis_title="", height=420)
        st.plotly_chart(fig_geo, use_container_width=True)

        # ══════════════════════════════════════════════════════
        # 7 · CORRELATION HEATMAP
        # ══════════════════════════════════════════════════════
        st.markdown("""<div class="eda-banner">
  <div class="eda-icon">🔬</div>
  <div><p class="eda-title">What actually correlates with salary?</p>
  <p class="eda-sub">Pearson correlation matrix — red = negative, blue = positive.</p></div>
</div>""", unsafe_allow_html=True)

        st.markdown("""<div class="callout">
<b>Experience</b> has the strongest positive correlation with salary — confirming
what the violin plot showed visually. Remote ratio and company size have
surprisingly <b>weak</b> correlations, meaning they add context but aren't strong
standalone predictors on their own.
</div>""", unsafe_allow_html=True)

        corr_df = raw[["work_year","salary_in_usd","remote_ratio"]].copy()
        corr_df["Experience"]   = raw["experience_level"].map({"EN":0,"MI":1,"SE":2,"EX":3})
        corr_df["Company Size"] = raw["company_size"].map({"S":0,"M":1,"L":2})
        corr_df = corr_df.rename(columns={
            "work_year":"Year","salary_in_usd":"Salary","remote_ratio":"Remote %"})
        cm = corr_df.corr()

        fig_heat = go.Figure(go.Heatmap(
            z=cm.values, x=cm.columns.tolist(), y=cm.index.tolist(),
            colorscale="RdBu", reversescale=True, zmin=-1, zmax=1,
            text=[[f"{v:.2f}" for v in row] for row in cm.values],
            texttemplate="%{text}",
            textfont=dict(size=13, family="Inter,sans-serif"),
            showscale=True,
            hovertemplate="%{y} × %{x}<br>r = %{z:.3f}<extra></extra>",
        ))
        fig_heat.update_layout(**{**_T, "margin": dict(t=50, b=25, l=10, r=15)},
            title="Feature Correlation Matrix",
            height=390)
        st.plotly_chart(fig_heat, use_container_width=True)

        # ══════════════════════════════════════════════════════
        # 8 · KEY TAKEAWAYS
        # ══════════════════════════════════════════════════════
        st.markdown("""<div class="eda-banner">
  <div class="eda-icon">💡</div>
  <div><p class="eda-title">3 things the data made crystal clear</p>
  <p class="eda-sub">The surprises, the confirmations, and the question the model answers.</p></div>
</div>""", unsafe_allow_html=True)

        k1, k2, k3 = st.columns(3, gap="medium")
        with k1:
            st.markdown(f"""<div class="tcard tcard-blue">
<span class="tcard-num">{mult}×</span>
<b>Experience is everything.</b><br>
Senior roles pay {mult}× more than Entry-level — more than any other
factor in the dataset. One promotion matters more than changing companies.
</div>""", unsafe_allow_html=True)
        with k2:
            st.markdown(f"""<div class="tcard tcard-green">
<span class="tcard-num">+{growth}%</span>
<b>The market accelerated fast.</b><br>
Salaries rose {growth}% from {yr_min}→{yr_max}.
Demand for ML and AI talent drove the steepest gains at Senior level.
</div>""", unsafe_allow_html=True)
        with k3:
            sign2 = "+" if rem_pct > 0 else ""
            st.markdown(f"""<div class="tcard tcard-amber">
<span class="tcard-num">{sign2}{rem_pct}%</span>
<b>Remote pays more — barely.</b><br>
Fully remote roles earn just {sign2}{rem_pct}% more than on-site.
The "remote premium" is real but tiny. Role and seniority dominate everything.
</div>""", unsafe_allow_html=True)


if __name__ == "__main__":
    main()
