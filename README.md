# Salary Predictor 💼

> **Instant, data-backed salary intelligence for data science roles.**  
> Fill in a profile → get a precise salary figure → understand where it sits in the market.

[![Python](https://img.shields.io/badge/Python-3.13-blue?logo=python&logoColor=white)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.32-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![Supabase](https://img.shields.io/badge/Supabase-Postgres-3ECF8E?logo=supabase&logoColor=white)](https://supabase.com/)
[![Deployed on Render](https://img.shields.io/badge/API-Render-46E3B7?logo=render&logoColor=white)](https://render.com/)

---

## What It Does

Salary Predictor takes a few inputs — job title, experience level, company size, remote setup — and returns a precise USD salary prediction trained on thousands of real data science compensation records.

It doesn't just return a number. It gives you:

- A **market position** — whether the salary is above or below average, and by how much
- A **plain-language narrative** explaining the result in context
- A **prediction history** with summary statistics and charts
- A **data insights tab** showing salary distributions across the full dataset

---

## Live Demo

| Layer | URL |
|---|---|
| Dashboard | [salary-predictor.streamlit.app](https://salary-predictor.streamlit.app) |
| API | Deployed on Render (cold start may take ~30s on first request) |
| API Docs | `{API_URL}/docs` |

---

## System Architecture

```
User fills sidebar form
         │
         ▼
┌─────────────────────┐
│  Streamlit Dashboard │  — Three tabs: Predict / History / Data Insights
└────────┬────────────┘
         │ HTTP request
         ▼
┌─────────────────────┐
│   FastAPI Backend    │  — Deployed on Render
│   /predict endpoint  │
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│   Random Forest      │  — Trained on ds_salaries dataset
│   Model (.pkl)       │  — R² = 0.47 | MAE = $31,633
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│   Supabase (Postgres)│  — Stores every prediction
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│   LLM Narrative      │  — Ollama / Llama 3.1 (local)
│   (market insight)   │  — Compares to real Supabase averages
└─────────────────────┘
```

---

## Project Structure

```
salary-predictor/
│
├── src/salary_predictor/
│   ├── api/                  # FastAPI app
│   │   ├── main.py           # App entry point, model load on startup
│   │   ├── predict.py        # /predict endpoint
│   │   └── schemas.py        # Pydantic request/response models
│   │
│   ├── dashboard/
│   │   └── app.py            # Streamlit dashboard (3 tabs)
│   │
│   ├── llm/
│   │   ├── analyst.py        # LLM narrative generation
│   │   └── charts.py         # Plotly chart builders
│   │
│   ├── db/
│   │   ├── supabase_client.py # Supabase connection
│   │   └── persist.py         # Save & load predictions
│   │
│   ├── pipeline/             # Data processing pipeline
│   └── model.py              # Load model, encode inputs, predict
│
├── models/                   # Trained model artifacts
│   ├── salary_model.pkl
│   ├── encoders.pkl
│   └── model_metadata.pkl
│
├── data/
│   ├── raw/                  # Original ds_salaries.csv
│   └── processed/            # Cleaned & encoded data
│
├── scripts/
│   ├── run_pipeline.py       # Train & save the model
│   └── run_analysis.py       # EDA scripts
│
├── tests/
│   ├── test_api.py
│   └── test_model.py
│
├── streamlit_app.py          # Streamlit Cloud entry point
├── render.yaml               # Render deployment config
└── pyproject.toml            # Project dependencies
```

---

## Model

| Property | Value |
|---|---|
| Algorithm | Random Forest (baseline) |
| Training data | `ds_salaries` — real AI/ML job salaries 2020–2023 |
| R² (test) | 0.47 |
| MAE | $31,633 |
| Cross-val R² | 0.49 |

> **Why Random Forest?** A tuned GridSearchCV model had a higher CV R² (0.53) but worse test R² and higher MAE — signs of overfitting. The baseline was chosen for better real-world generalization.

---

## API Reference

### `GET /predict`

Returns a predicted salary in USD.

**Query Parameters**

| Parameter | Type | Accepted Values |
|---|---|---|
| `experience_level` | string | `EN` `MI` `SE` `EX` |
| `employment_type` | string | `FT` `PT` `CT` `FL` |
| `company_size` | string | `S` `M` `L` |
| `remote_ratio` | int | `0` `50` `100` |
| `work_year` | int | `2020` – `2023` |
| `job_title` | string | See encoder classes |
| `employee_residence` | string | Country code e.g. `US` |
| `company_location` | string | Country code e.g. `US` |

**Example Request**
```bash
curl "https://your-api.onrender.com/predict?\
experience_level=SE&\
employment_type=FT&\
company_size=L&\
remote_ratio=100&\
work_year=2023&\
job_title=Data%20Scientist&\
employee_residence=US&\
company_location=US"
```

**Example Response**
```json
{
  "predicted_salary_usd": 148500.0,
  "inputs": {
    "experience_level": "SE",
    "employment_type": "FT",
    "company_size": "L",
    "remote_ratio": 100,
    "work_year": 2023,
    "job_title": "Data Scientist",
    "employee_residence": "US",
    "company_location": "US"
  }
}
```

### `GET /health`

```json
{ "status": "ok", "model_loaded": true }
```

---

## Local Setup

### 1. Clone the repo

```bash
git clone https://github.com/ali-hamad0/salary-prediction-app.git
cd salary-predictor
```

### 2. Install dependencies

```bash
pip install -e .
```

### 3. Set environment variables

Create a `.env` file at the project root:

```env
API_BASE_URL=http://127.0.0.1:8000
SUPABASE_URL=your_supabase_url
SUPABASE_KEY=your_supabase_anon_key
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=llama3.1:latest
```

### 4. Train the model (first time only)

```bash
python scripts/run_pipeline.py
```

### 5. Start the API

```bash
uvicorn salary_predictor.api.main:app --reload
```

### 6. Start the dashboard

```bash
streamlit run streamlit_app.py
```

---

## Running Tests

```bash
pytest tests/ -v
```

Tests follow the **Arrange → Act → Assert** pattern and cover the prediction logic and API endpoints.

---

## Deployment

| Service | Platform | Config |
|---|---|---|
| API | Render | `render.yaml` |
| Dashboard | Streamlit Cloud | `streamlit_app.py` |
| Database | Supabase | Managed Postgres |
| LLM | Ollama (local) | Llama 3.1 |

---

## Known Limitations

- Dataset covers **2020–2023** — predictions for newer years are extrapolated
- LLM narrative requires a **local Ollama instance** — not available in the hosted version
- Model R² of 0.47 reflects the natural variance in salary data across regions and titles
- No authentication on the API — not production-ready as-is

---

## Roadmap

- [ ] Refresh dataset with 2024–2025 salary data
- [ ] Replace local Ollama with a hosted LLM API
- [ ] Add salary trend charts over time
- [ ] HR platform integration via API
- [ ] Authentication layer for production use

---

## Author

**Ali Hamad**  
Built as part of the AI Engineering course — first end-to-end ML product.
