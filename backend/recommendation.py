"""
Recommendation Engine
=======================
Rule-based expert system (explicitly NOT an LLM — the spec keeps the LLM
out of decision-making and reserves it for natural-language explanation
only). Each rule fires off a threshold computed against the facility's own
recent performance and its peer group, and reports an estimated impact.
"""

import pandas as pd


def _peer_percentile(value: float, peer_values: pd.Series) -> float:
    if len(peer_values) <= 1:
        return 50.0
    return float((peer_values < value).mean() * 100)


def generate_recommendations(daily_metrics: pd.DataFrame, facility_id: str, window_days: int = 30) -> list:
    """
    Generate recommendations for a single facility based on its trailing
    `window_days` performance versus its facility_type peer group.

    Returns a list of dicts: category, recommendation_text, estimated_impact, priority.
    """
    df = daily_metrics.copy()
    df["date"] = pd.to_datetime(df["date"])
    cutoff = df["date"].max() - pd.Timedelta(days=window_days)
    recent = df[df["date"] >= cutoff]

    fac_rows = recent[recent["facility_id"] == facility_id]
    if fac_rows.empty:
        return []

    ftype = fac_rows["facility_type"].iloc[0]
    peers = recent[recent["facility_type"] == ftype]

    recs = []

    # --- Diesel / transportation ---
    fac_diesel = fac_rows["diesel_litres"].mean()
    peer_diesel = peers.groupby("facility_id")["diesel_litres"].mean()
    diesel_pctile = _peer_percentile(fac_diesel, peer_diesel)
    if diesel_pctile > 70:
        reduction = min(35, (diesel_pctile - 50) * 0.8)
        recs.append(dict(
            category="Transportation",
            recommendation_text="Reduce diesel-powered transportation by shifting eligible road "
                                  "shipments to rail and consolidating partial loads.",
            estimated_impact=f"~{reduction:.0f}% reduction in diesel-related emissions",
            priority="High" if diesel_pctile > 85 else "Medium",
        ))

    # --- Renewable energy ---
    fac_renewable = fac_rows["renewable_energy_pct"].mean()
    if fac_renewable < 25:
        target = min(50, fac_renewable + 20)
        recs.append(dict(
            category="Energy",
            recommendation_text=f"Increase renewable energy adoption (currently {fac_renewable:.1f}%) "
                                  "through on-site solar or renewable power purchase agreements.",
            estimated_impact=f"Could reduce non-renewable grid emissions by up to "
                              f"{(target - fac_renewable):.0f} percentage points of total load",
            priority="High" if fac_renewable < 15 else "Medium",
        ))

    # --- Recycling / waste ---
    fac_recycling = fac_rows["recycling_pct"].mean()
    if fac_recycling < 40:
        recs.append(dict(
            category="Waste",
            recommendation_text="Improve recycling and waste segregation processes; "
                                  "current recycling rate is below the operational target.",
            estimated_impact=f"Raising recycling rate from {fac_recycling:.0f}% toward 60% "
                              f"could cut landfill-bound waste meaningfully",
            priority="Medium",
        ))

    # --- Carbon intensity per shipment ---
    fac_shipments = fac_rows["shipments_processed"].sum()
    fac_emissions = fac_rows["carbon_emissions_kg"].sum()
    if fac_shipments > 0:
        fac_intensity = fac_emissions / fac_shipments
        peer_intensity = (
            peers.groupby("facility_id")
            .apply(lambda g: g["carbon_emissions_kg"].sum() / max(1, g["shipments_processed"].sum()))
        )
        intensity_pctile = _peer_percentile(fac_intensity, peer_intensity)
        if intensity_pctile > 70:
            recs.append(dict(
                category="Logistics",
                recommendation_text="Optimize shipment routes and consolidate smaller shipments to "
                                      "lower carbon emissions per shipment.",
                estimated_impact=f"Carbon-per-shipment is in the top {100 - intensity_pctile:.0f}% "
                                  f"of {ftype.lower()}s — route optimization could close most of that gap",
                priority="Medium",
            ))

    # --- Production shift (only meaningful for Plants) ---
    if ftype == "Plant":
        fac_carbon_per_unit = fac_rows["carbon_emissions_kg"].sum() / max(1, fac_rows["production_volume"].sum())
        peer_cpu = peers.groupby("facility_id").apply(
            lambda g: g["carbon_emissions_kg"].sum() / max(1, g["production_volume"].sum())
        )
        cpu_pctile = _peer_percentile(fac_carbon_per_unit, peer_cpu)
        if cpu_pctile > 75:
            best_peer = peer_cpu.idxmin()
            recs.append(dict(
                category="Production",
                recommendation_text=f"Consider shifting a portion of production volume toward "
                                      f"lower-emission plants such as {best_peer}.",
                estimated_impact=f"This facility's emissions per production unit rank in the "
                                  f"bottom {100 - cpu_pctile:.0f}% of plants",
                priority="High" if cpu_pctile > 90 else "Medium",
            ))

    if not recs:
        recs.append(dict(
            category="General",
            recommendation_text="No urgent issues detected. Maintain current sustainability practices "
                                  "and continue monitoring monthly trends.",
            estimated_impact="N/A",
            priority="Low",
        ))

    priority_order = {"High": 0, "Medium": 1, "Low": 2}
    recs.sort(key=lambda r: priority_order[r["priority"]])
    return recs


if __name__ == "__main__":
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from database.connection import get_session
    from database.schema import DailyMetric

    session = get_session()
    rows = session.query(DailyMetric).all()
    df = pd.DataFrame([{
        "facility_id": r.facility_id, "facility_type": r.facility_type, "date": r.date,
        "diesel_litres": r.diesel_litres, "renewable_energy_pct": r.renewable_energy_pct,
        "recycling_pct": r.recycling_pct, "carbon_emissions_kg": r.carbon_emissions_kg,
        "shipments_processed": r.shipments_processed, "production_volume": r.production_volume,
    } for r in rows])
    session.close()

    for fid in ["PLANT-005", "WH-011", "SUP-001"]:
        print(f"\n=== Recommendations for {fid} ===")
        for r in generate_recommendations(df, fid):
            print(f"[{r['priority']}] {r['category']}: {r['recommendation_text']}")
            print(f"    Impact: {r['estimated_impact']}")
