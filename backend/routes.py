"""
API Routes
==========
All HTTP route handlers, grouped under /api.

Design note: each route is a THIN wrapper around a plain `_*_logic` function
with ordinary Python defaults. This matters because backend/llm.py's context
builder (`_build_chat_context`) needs to call this logic directly (not as an
HTTP request) to assemble RAG context for the chatbot — calling the
FastAPI-decorated route functions directly would break, since their default
arguments are `Query(...)` sentinel objects that only resolve inside the
request pipeline.
"""

from datetime import timedelta
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel

from database.queries import get_daily_metrics, get_facilities, compute_kpis
from ml.scoring import compute_scores
from ml.anomaly import detect_anomalies
from ml.hotspot import top_emitters, monthly_hotspot_summary
from ml.prediction import load_model, predict_future, train_model
from backend.recommendation import generate_recommendations
from backend.llm import ask_assistant

router = APIRouter(prefix="/api")


# ---------------------------------------------------------------------------
# Core logic (plain functions, callable from routes OR from the chat context
# builder, with ordinary Python default values)
# ---------------------------------------------------------------------------

def _kpis_logic(window_days: int = 30) -> dict:
    df = get_daily_metrics()
    if df.empty:
        raise HTTPException(404, "No data loaded. Run the seed script first.")
    return compute_kpis(df, window_days=window_days)


def _scores_logic(window_days: int = 30) -> list:
    df = get_daily_metrics()
    if df.empty:
        raise HTTPException(404, "No data loaded.")
    end = df["date"].max()
    start = end - timedelta(days=window_days)
    result = compute_scores(df, period_start=start, period_end=end)
    return result.to_dict(orient="records")


def _anomalies_logic(facility_id: str = None, severity: str = None, window_days: int = 90) -> list:
    df = get_daily_metrics(facility_id=facility_id)
    if df.empty:
        raise HTTPException(404, "No data loaded.")
    if not facility_id:
        cutoff = df["date"].max() - timedelta(days=window_days)
        df = df[df["date"] >= cutoff]
    result = detect_anomalies(df)
    if severity:
        result = result[result["severity"].str.lower() == severity.lower()]
    return result.to_dict(orient="records")


def _hotspots_logic(n: int = 10, facility_type: str = None, window_days: int = 30) -> list:
    df = get_daily_metrics()
    if df.empty:
        raise HTTPException(404, "No data loaded.")
    result = top_emitters(df, n=n, facility_type=facility_type, window_days=window_days)
    return result.to_dict(orient="records")


def _recommendations_logic(facility_id: str, window_days: int = 30) -> list:
    df = get_daily_metrics(facility_id=facility_id)
    if df.empty:
        raise HTTPException(404, f"No data for facility {facility_id}")
    full_df = get_daily_metrics()  # need peer group for percentile comparisons
    return generate_recommendations(full_df, facility_id, window_days=window_days)


# ---------------------------------------------------------------------------
# Facilities & KPIs
# ---------------------------------------------------------------------------

@router.get("/facilities")
def list_facilities():
    df = get_facilities()
    return df.to_dict(orient="records")


@router.get("/facilities/{facility_id}/history")
def facility_history(facility_id: str, days: int = Query(90, ge=1, le=365)):
    df = get_daily_metrics(facility_id=facility_id)
    if df.empty:
        raise HTTPException(404, f"No data for facility {facility_id}")
    df = df.sort_values("date").tail(days)
    return df.to_dict(orient="records")


@router.get("/kpis")
def kpis(window_days: int = Query(30, ge=1, le=365)):
    return _kpis_logic(window_days)


# ---------------------------------------------------------------------------
# Sustainability scoring
# ---------------------------------------------------------------------------

@router.get("/scores")
def scores(window_days: int = Query(30, ge=1, le=365)):
    return _scores_logic(window_days)


@router.get("/scores/{facility_id}")
def facility_score(facility_id: str, window_days: int = Query(30, ge=1, le=365)):
    all_scores = _scores_logic(window_days)
    match = [s for s in all_scores if s["facility_id"] == facility_id]
    if not match:
        raise HTTPException(404, f"No score available for {facility_id}")
    return match[0]


# ---------------------------------------------------------------------------
# Anomaly detection
# ---------------------------------------------------------------------------

@router.get("/anomalies")
def anomalies(facility_id: str = None, severity: str = None, window_days: int = Query(90, ge=1, le=365)):
    return _anomalies_logic(facility_id, severity, window_days)


# ---------------------------------------------------------------------------
# Carbon hotspots
# ---------------------------------------------------------------------------

@router.get("/hotspots")
def hotspots(n: int = Query(10, ge=1, le=50), facility_type: str = None, window_days: int = Query(30, ge=1, le=365)):
    return _hotspots_logic(n, facility_type, window_days)


@router.get("/hotspots/monthly")
def hotspots_monthly():
    df = get_daily_metrics()
    if df.empty:
        raise HTTPException(404, "No data loaded.")
    return monthly_hotspot_summary(df).to_dict(orient="records")


# ---------------------------------------------------------------------------
# AI Insights (auto-generated trend call-outs for the dashboard's Insights page)
# ---------------------------------------------------------------------------

@router.get("/insights")
def insights(window_days: int = Query(30, ge=1, le=180)):
    from reports.generator import _trend_lines  # local import: avoids a circular import at module load time
    df = get_daily_metrics()
    if df.empty:
        raise HTTPException(404, "No data loaded.")
    trends = _trend_lines(df, window_days=window_days)
    anomalies = _anomalies_logic(window_days=window_days)
    high = [a for a in anomalies if a["severity"] == "High"]
    return {
        "positive_trends": trends["positive"],
        "negative_trends": trends["negative"],
        "high_severity_anomalies": high[:10],
        "total_anomalies": len(anomalies),
    }


# ---------------------------------------------------------------------------
# Carbon emission prediction
# ---------------------------------------------------------------------------

@router.post("/predict/train")
def train_prediction_model():
    df = get_daily_metrics()
    if df.empty:
        raise HTTPException(404, "No data loaded.")
    artifact = train_model(df)
    return {"status": "trained", "metrics": artifact["metrics"]}


@router.get("/predict/{facility_id}")
def predict(facility_id: str):
    df = get_daily_metrics(facility_id=facility_id)
    if df.empty:
        raise HTTPException(404, f"No data for facility {facility_id}")
    try:
        artifact = load_model()
    except FileNotFoundError:
        raise HTTPException(409, "Model not trained yet. POST /api/predict/train first.")
    try:
        return predict_future(df, facility_id, artifact)
    except ValueError as e:
        raise HTTPException(404, str(e))


# ---------------------------------------------------------------------------
# Recommendations
# ---------------------------------------------------------------------------

@router.get("/recommendations/{facility_id}")
def recommendations(facility_id: str, window_days: int = Query(30, ge=1, le=365)):
    return _recommendations_logic(facility_id, window_days)


# ---------------------------------------------------------------------------
# AI Executive Assistant (Groq)
# ---------------------------------------------------------------------------

class ChatRequest(BaseModel):
    question: str
    facility_id: str | None = None


def _build_chat_context(req: ChatRequest) -> dict:
    """Lightweight intent routing: pull just the data relevant to the question."""
    context = {}
    q = req.question.lower()

    if req.facility_id:
        context["facility_id"] = req.facility_id
        context["recent_metrics"] = get_daily_metrics(facility_id=req.facility_id).tail(30).to_dict("records")

    if "score" in q or "rated" in q:
        context["scores"] = _scores_logic()[:15]
    if "anomal" in q or "unusual" in q or "spike" in q:
        context["anomalies"] = _anomalies_logic()[:15]
    if "emit" in q or "highest carbon" in q or "hotspot" in q or "worst" in q:
        context["top_emitters"] = _hotspots_logic()
    if "recommend" in q or "improve" in q or "should" in q:
        if req.facility_id:
            context["recommendations"] = _recommendations_logic(req.facility_id)
    if "summary" in q or "report" in q or "performance" in q:
        context["kpis"] = _kpis_logic()
        context["top_emitters"] = _hotspots_logic()
        context["scores"] = _scores_logic()[:10]

    if not context:
        # Fallback: give general KPIs so the model isn't answering with zero grounding
        context["kpis"] = _kpis_logic()

    return context


@router.post("/chat")
def chat(req: ChatRequest):
    context = _build_chat_context(req)
    try:
        answer = ask_assistant(req.question, context)
    except RuntimeError as e:
        raise HTTPException(500, str(e))
    except Exception as e:
        raise HTTPException(502, f"LLM request failed: {e}")
    return {"question": req.question, "answer": answer, "context_used": list(context.keys())}
