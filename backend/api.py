"""
FastAPI Application
====================
Entry point for the backend. Run with:

    uvicorn backend.api:app --reload --port 8000

On startup, ensures the database schema exists (it does NOT auto-seed data —
run `python -m synthetic_data.generator` then `python -m database.seed`
once before starting the server).
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from database.connection import init_db
from backend.routes import router

app = FastAPI(
    title="AI Supply Chain Sustainability Tracking System",
    description="Synthetic-data-driven sustainability copilot: KPIs, predictions, "
                "anomaly detection, scoring, recommendations, and an LLM assistant.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten this in production
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def on_startup():
    init_db()


@app.get("/")
def root():
    return {
        "status": "ok",
        "name": "AI Supply Chain Sustainability Tracking System",
        "docs": "/docs",
    }


@app.get("/health")
def health():
    return {"status": "healthy"}


app.include_router(router)
