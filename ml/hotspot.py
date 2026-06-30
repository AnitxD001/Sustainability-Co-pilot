"""
Carbon Hotspot Detection
=========================
No ML required per spec — pure ranking. Identifies the highest-emitting
facilities by total emissions, carbon intensity (per shipment), and carbon
per production unit, for the "Top 10 Emitters" dashboard view.
"""

import pandas as pd


def top_emitters(daily_metrics: pd.DataFrame, n: int = 10, facility_type: str = None,
                  window_days: int = 30) -> pd.DataFrame:
    df = daily_metrics.copy()
    df["date"] = pd.to_datetime(df["date"])
    cutoff = df["date"].max() - pd.Timedelta(days=window_days)
    df = df[df["date"] >= cutoff]

    if facility_type:
        df = df[df["facility_type"] == facility_type]

    agg = df.groupby(["facility_id", "facility_type"]).agg(
        total_emissions_kg=("carbon_emissions_kg", "sum"),
        total_shipments=("shipments_processed", "sum"),
        total_production=("production_volume", "sum"),
    ).reset_index()

    agg["carbon_per_shipment"] = agg["total_emissions_kg"] / agg["total_shipments"].replace(0, pd.NA)
    agg["carbon_per_production_unit"] = agg["total_emissions_kg"] / agg["total_production"].replace(0, pd.NA)

    return agg.sort_values("total_emissions_kg", ascending=False).head(n).reset_index(drop=True)


def monthly_hotspot_summary(daily_metrics: pd.DataFrame) -> pd.DataFrame:
    """Total emissions by facility_type by calendar month — for trend charts."""
    df = daily_metrics.copy()
    df["date"] = pd.to_datetime(df["date"])
    df["month"] = df["date"].dt.to_period("M").astype(str)
    return (
        df.groupby(["month", "facility_type"])["carbon_emissions_kg"]
        .sum()
        .reset_index()
        .rename(columns={"carbon_emissions_kg": "total_emissions_kg"})
    )
