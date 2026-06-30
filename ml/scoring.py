"""
Sustainability Scoring Engine
==============================
Rule-based weighted scoring (deliberately NOT machine learning) so the score
is fully transparent and explainable to executives.

Weights (per spec):
    Carbon Emissions     25%
    Energy Consumption   20%
    Water Usage          15%
    Waste Generation     15%
    Renewable Energy     15%
    Recycling Rate       10%

Each facility is scored against peers of the SAME facility_type (a Plant
should not be penalized for using more water than a Distribution Center —
they aren't comparable). Metrics are min-max normalized within the peer
group over the scoring window, then combined.

Classification:
    90-100  Excellent
    75-89   Good
    60-74   Average
    <60     Needs Attention
"""

import pandas as pd
import numpy as np

WEIGHTS = {
    "carbon": 0.25,
    "energy": 0.20,
    "water": 0.15,
    "waste": 0.15,
    "renewable": 0.15,
    "recycling": 0.10,
}


def classify(score: float) -> str:
    if score >= 90:
        return "Excellent"
    if score >= 75:
        return "Good"
    if score >= 60:
        return "Average"
    return "Needs Attention"


def _normalize_lower_is_better(series: pd.Series) -> pd.Series:
    lo, hi = series.min(), series.max()
    if hi - lo < 1e-9:
        return pd.Series(100.0, index=series.index)
    return 100 * (1 - (series - lo) / (hi - lo))


def _normalize_higher_is_better(series: pd.Series) -> pd.Series:
    lo, hi = series.min(), series.max()
    if hi - lo < 1e-9:
        return pd.Series(100.0, index=series.index)
    return 100 * (series - lo) / (hi - lo)


def compute_scores(daily_metrics: pd.DataFrame, period_start=None, period_end=None) -> pd.DataFrame:
    """
    Compute sustainability scores per facility over the given window.

    Parameters
    ----------
    daily_metrics : DataFrame with columns matching database.schema.DailyMetric
                     (facility_id, facility_type, date, electricity_kwh, diesel_litres,
                      water_litres, carbon_emissions_kg, waste_generated_kg,
                      recycling_pct, renewable_energy_pct, shipments_processed)
    period_start, period_end : optional date filters; defaults to the full range present.

    Returns
    -------
    DataFrame: facility_id, facility_type, period_start, period_end, sub-scores,
               overall_score, classification
    """
    df = daily_metrics.copy()
    df["date"] = pd.to_datetime(df["date"])

    if period_start:
        df = df[df["date"] >= pd.to_datetime(period_start)]
    if period_end:
        df = df[df["date"] <= pd.to_datetime(period_end)]

    if df.empty:
        return pd.DataFrame()

    agg = df.groupby(["facility_id", "facility_type"]).agg(
        avg_carbon_per_shipment=("carbon_emissions_kg", lambda s: s.sum() / max(1, df.loc[s.index, "shipments_processed"].sum())),
        total_carbon=("carbon_emissions_kg", "sum"),
        total_electricity=("electricity_kwh", "sum"),
        total_water=("water_litres", "sum"),
        total_waste=("waste_generated_kg", "sum"),
        avg_recycling=("recycling_pct", "mean"),
        avg_renewable=("renewable_energy_pct", "mean"),
        total_shipments=("shipments_processed", "sum"),
    ).reset_index()

    # Normalize per facility_type peer group
    results = []
    for ftype, group in agg.groupby("facility_type"):
        g = group.copy()
        g["carbon_subscore"] = _normalize_lower_is_better(g["total_carbon"])
        g["energy_subscore"] = _normalize_lower_is_better(g["total_electricity"])
        g["water_subscore"] = _normalize_lower_is_better(g["total_water"])
        g["waste_subscore"] = _normalize_lower_is_better(g["total_waste"])
        g["renewable_subscore"] = _normalize_higher_is_better(g["avg_renewable"])
        g["recycling_subscore"] = _normalize_higher_is_better(g["avg_recycling"])
        results.append(g)

    scored = pd.concat(results, ignore_index=True)

    scored["overall_score"] = (
        scored["carbon_subscore"] * WEIGHTS["carbon"]
        + scored["energy_subscore"] * WEIGHTS["energy"]
        + scored["water_subscore"] * WEIGHTS["water"]
        + scored["waste_subscore"] * WEIGHTS["waste"]
        + scored["renewable_subscore"] * WEIGHTS["renewable"]
        + scored["recycling_subscore"] * WEIGHTS["recycling"]
    ).round(1)

    scored["classification"] = scored["overall_score"].apply(classify)
    scored["period_start"] = df["date"].min().date()
    scored["period_end"] = df["date"].max().date()

    cols = ["facility_id", "facility_type", "period_start", "period_end",
            "carbon_subscore", "energy_subscore", "water_subscore", "waste_subscore",
            "renewable_subscore", "recycling_subscore", "overall_score", "classification"]
    return scored[cols].sort_values("overall_score", ascending=False).reset_index(drop=True)


if __name__ == "__main__":
    # Quick standalone smoke test against the seeded DB
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from database.connection import get_session
    from database.schema import DailyMetric

    session = get_session()
    rows = session.query(DailyMetric).all()
    df = pd.DataFrame([{
        "facility_id": r.facility_id, "facility_type": r.facility_type, "date": r.date,
        "electricity_kwh": r.electricity_kwh, "diesel_litres": r.diesel_litres,
        "water_litres": r.water_litres, "carbon_emissions_kg": r.carbon_emissions_kg,
        "waste_generated_kg": r.waste_generated_kg, "recycling_pct": r.recycling_pct,
        "renewable_energy_pct": r.renewable_energy_pct, "shipments_processed": r.shipments_processed,
    } for r in rows])
    session.close()

    scores = compute_scores(df)
    print(scores.head(10).to_string(index=False))
    print(f"\nTotal facilities scored: {len(scores)}")
    print(scores["classification"].value_counts())
