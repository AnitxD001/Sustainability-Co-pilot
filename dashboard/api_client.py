"""
API Client
==========
Thin requests wrapper used by every dashboard page to talk to the FastAPI
backend. Centralizes the base URL, timeouts, and error handling so pages
don't each reinvent connection-error messaging.
"""

import os
import requests
import streamlit as st

API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")


class APIError(Exception):
    pass


def _request(method: str, path: str, **kwargs):
    url = f"{API_BASE_URL}{path}"
    try:
        resp = requests.request(method, url, timeout=30, **kwargs)
    except requests.exceptions.ConnectionError:
        raise APIError(
            f"Can't reach the backend at {API_BASE_URL}. "
            "Start it with: uvicorn backend.api:app --reload --port 8000"
        )
    if resp.status_code >= 400:
        try:
            detail = resp.json().get("detail", resp.text)
        except Exception:
            detail = resp.text
        raise APIError(detail)
    return resp.json()


@st.cache_data(ttl=60)
def get_facilities():
    return _request("GET", "/api/facilities")


@st.cache_data(ttl=60)
def get_kpis(window_days: int = 30):
    return _request("GET", "/api/kpis", params={"window_days": window_days})


@st.cache_data(ttl=60)
def get_facility_history(facility_id: str, days: int = 90):
    return _request("GET", f"/api/facilities/{facility_id}/history", params={"days": days})


@st.cache_data(ttl=60)
def get_scores(window_days: int = 30):
    return _request("GET", "/api/scores", params={"window_days": window_days})


def get_facility_score(facility_id: str, window_days: int = 30):
    return _request("GET", f"/api/scores/{facility_id}", params={"window_days": window_days})


@st.cache_data(ttl=60)
def get_anomalies(facility_id: str = None, severity: str = None, window_days: int = 90):
    params = {"window_days": window_days}
    if facility_id:
        params["facility_id"] = facility_id
    if severity:
        params["severity"] = severity
    return _request("GET", "/api/anomalies", params=params)


@st.cache_data(ttl=60)
def get_hotspots(n: int = 10, facility_type: str = None, window_days: int = 30):
    params = {"n": n, "window_days": window_days}
    if facility_type:
        params["facility_type"] = facility_type
    return _request("GET", "/api/hotspots", params=params)


@st.cache_data(ttl=60)
def get_hotspots_monthly():
    return _request("GET", "/api/hotspots/monthly")


@st.cache_data(ttl=60)
def get_insights(window_days: int = 30):
    return _request("GET", "/api/insights", params={"window_days": window_days})


def train_prediction_model():
    return _request("POST", "/api/predict/train")


def get_prediction(facility_id: str):
    return _request("GET", f"/api/predict/{facility_id}")


def get_recommendations(facility_id: str, window_days: int = 30):
    return _request("GET", f"/api/recommendations/{facility_id}", params={"window_days": window_days})


def post_chat(question: str, facility_id: str = None):
    payload = {"question": question}
    if facility_id:
        payload["facility_id"] = facility_id
    return _request("POST", "/api/chat", json=payload)
