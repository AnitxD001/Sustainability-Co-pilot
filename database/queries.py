"""
Data Access Helpers
=====================
Shared helpers for pulling DB data into pandas DataFrames. Used by the
FastAPI routes, the Streamlit dashboard (indirectly, via the API), and the
report generator. Centralizing this avoids every module re-writing the same
SQLAlchemy-to-DataFrame boilerplate.
"""

import pandas as pd
from sqlalchemy import select

from database.connection import get_session
from database.schema import Facility, DailyMetric


def get_daily_metrics(facility_id: str = None, start_date=None, end_date=None) -> pd.DataFrame:
    session = get_session()
    try:
        stmt = select(DailyMetric)
        if facility_id:
            stmt = stmt.where(DailyMetric.facility_id == facility_id)
        if start_date:
            stmt = stmt.where(DailyMetric.date >= start_date)
        if end_date:
            stmt = stmt.where(DailyMetric.date <= end_date)
        rows = session.execute(stmt).scalars().all()

        return pd.DataFrame([{
            "date": r.date, "facility_id": r.facility_id, "facility_type": r.facility_type,
            "supplier_id": r.supplier_id, "electricity_kwh": r.electricity_kwh,
            "diesel_litres": r.diesel_litres, "water_litres": r.water_litres,
            "carbon_emissions_kg": r.carbon_emissions_kg, "waste_generated_kg": r.waste_generated_kg,
            "recycling_pct": r.recycling_pct, "renewable_energy_pct": r.renewable_energy_pct,
            "shipments_processed": r.shipments_processed, "shipment_distance_km": r.shipment_distance_km,
            "transport_mode": r.transport_mode, "production_volume": r.production_volume,
            "sustainability_incidents": r.sustainability_incidents,
        } for r in rows])
    finally:
        session.close()


def get_facilities() -> pd.DataFrame:
    session = get_session()
    try:
        rows = session.execute(select(Facility)).scalars().all()
        return pd.DataFrame([{
            "facility_id": r.facility_id, "facility_type": r.facility_type,
            "facility_name": r.facility_name, "region": r.region,
        } for r in rows])
    finally:
        session.close()


def compute_kpis(daily_metrics: pd.DataFrame, window_days: int = 30) -> dict:
    """Aggregate headline KPIs for the dashboard home page."""
    df = daily_metrics.copy()
    if df.empty:
        return {}
    df["date"] = pd.to_datetime(df["date"])
    cutoff = df["date"].max() - pd.Timedelta(days=window_days)
    recent = df[df["date"] >= cutoff]

    total_shipments = recent["shipments_processed"].sum()
    total_emissions = recent["carbon_emissions_kg"].sum()

    return dict(
        total_carbon_emissions_kg=round(float(total_emissions), 1),
        avg_carbon_per_shipment_kg=round(float(total_emissions / max(1, total_shipments)), 2),
        total_water_litres=round(float(recent["water_litres"].sum()), 1),
        total_energy_kwh=round(float(recent["electricity_kwh"].sum()), 1),
        total_waste_kg=round(float(recent["waste_generated_kg"].sum()), 1),
        recycling_rate_pct=round(float(recent["recycling_pct"].mean()), 1),
        renewable_energy_pct=round(float(recent["renewable_energy_pct"].mean()), 1),
        carbon_intensity_kg_per_unit=round(
            float(total_emissions / max(1, recent["production_volume"].sum())), 3
        ),
        window_days=window_days,
        period_start=recent["date"].min().date().isoformat() if not recent.empty else None,
        period_end=recent["date"].max().date().isoformat() if not recent.empty else None,
    )
