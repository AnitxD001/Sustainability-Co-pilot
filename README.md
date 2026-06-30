# AI-Powered Supply Chain Sustainability Tracking System

An AI sustainability copilot for a simulated apparel manufacturing supply chain
(50 suppliers → 8 plants → 15 warehouses → 30 distribution centers, 365 days
of synthetic operational data). It predicts emissions, scores facilities,
detects anomalies, ranks carbon hotspots, generates recommendations, and
answers natural-language questions through a Groq-powered assistant — all
backed by a FastAPI service and a Streamlit dashboard.

## What's actually "AI" here, and what isn't

| Module | Approach |
|---|---|
| Carbon emission prediction | XGBoost regression |
| Sustainability scoring | Rule-based weighted scoring (deliberately not ML — needs to be explainable) |
| Anomaly detection | Isolation Forest |
| Carbon hotspot ranking | Plain ranking, no model needed |
| Recommendations | Rule-based expert system |
| Natural-language Q&A / report prose | Groq LLM (`meta-llama/llama-4-scout-17b-16e-instruct`) — explains and summarizes pre-computed numbers, never invents or recomputes them |

## Project layout

```
app.py                      Streamlit entry point (multi-page navigation)
backend/
  api.py                    FastAPI app factory
  routes.py                 All /api endpoints
  llm.py                    Groq integration (RAG-style grounding)
  recommendation.py         Rule-based recommendation engine
ml/
  prediction.py             XGBoost carbon emission forecasting
  anomaly.py                Isolation Forest anomaly detection
  scoring.py                Rule-based sustainability scoring
  hotspot.py                Carbon hotspot ranking
database/
  schema.py                 SQLAlchemy models
  connection.py             Engine/session (Postgres or SQLite)
  seed.py                   Loads synthetic CSVs into the DB
  queries.py                Shared DB -> DataFrame helpers
synthetic_data/
  generator.py               Generates the 365-day synthetic dataset
dashboard/
  api_client.py              Streamlit -> FastAPI HTTP client
  pages/                      Home, Facility View, AI Insights, Chatbot, ESG Report
reports/
  generator.py                Module 7: Markdown + PDF ESG report generator
  output/                     Generated reports land here
models/                       Trained model artifacts (.joblib)
requirements.txt
.env.example
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt

cp .env.example .env               # then edit .env and add your GROQ_API_KEY
```

By default the project uses a local SQLite file (`./data/sustainability.db`) —
no database server required to try it out. To use PostgreSQL instead, set
`DATABASE_URL` in `.env` (the SQLAlchemy models work unchanged against either).

## Running it

**1. Generate the synthetic dataset** (writes `data/sustainability_data.csv` and `data/facility_master.csv`):
```bash
python -m synthetic_data.generator
```

**2. Load it into the database:**
```bash
python -m database.seed
```

**3. Start the backend:**
```bash
uvicorn backend.api:app --reload --port 8000
```
API docs available at `http://localhost:8000/docs`.

**4. Train the prediction model** (one-time, or whenever you want to retrain):
```bash
curl -X POST http://localhost:8000/api/predict/train
```
(You can also do this from the dashboard's Facility View → Predictions tab.)

**5. Start the dashboard** (in a separate terminal, with the backend still running):
```bash
streamlit run app.py
```
Open `http://localhost:8501`.

## Notes on the Groq LLM integration

- The API key is read from the `GROQ_API_KEY` environment variable only —
  it is never hardcoded anywhere in this codebase. If a key is ever exposed
  (committed, pasted in chat, etc.), rotate it immediately in the
  [Groq console](https://console.groq.com/keys).
- The chatbot and ESG report's executive summary both work by pulling
  pre-computed metrics from the database (RAG-style) and handing them to the
  LLM with an explicit instruction not to invent figures outside that
  context. If `GROQ_API_KEY` isn't set, the chatbot page and report generator
  degrade gracefully (clear warning message / computed-metrics-only report)
  rather than failing silently.

## Testing notes

Every module in this repo was run end-to-end against the generated synthetic
dataset before delivery: the data generator, DB seeding, the scoring engine,
anomaly detection (recovers ~2.5% of records against a ~2.57% injection
rate), the XGBoost model (R² ≈ 0.985 on a 30-day holdout), the recommendation
engine, all FastAPI routes, the PDF/Markdown report generator, and all five
Streamlit pages (via Streamlit's `AppTest` headless test runner). The one
piece that could *not* be tested in the build environment is the live Groq
API call itself, since that environment has no outbound network access to
`api.groq.com` — the integration code follows Groq's standard OpenAI-compatible
chat completions format, so it should work as-is once you supply a real key,
but it's worth a quick manual check on your end.

## Future scope (not implemented)

Per the original spec: live IoT sensor integration, ERP integration (SAP/Oracle),
real-time logistics APIs, satellite-based monitoring, computer vision for waste
segregation, carbon credit estimation, multi-company benchmarking, supplier ESG
compliance monitoring, and a multi-agent AI assistant.
