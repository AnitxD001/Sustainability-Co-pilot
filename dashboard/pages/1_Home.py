import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import streamlit as st
import pandas as pd
import plotly.express as px

from dashboard.api_client import (
    get_kpis, get_hotspots, get_hotspots_monthly, get_scores, get_facilities, APIError
)

st.title("🌱 Sustainability Dashboard")
st.caption("AI Sustainability Copilot — Overview")

window = st.sidebar.slider("KPI window (days)", min_value=7, max_value=180, value=30, step=1)

try:
    kpis = get_kpis(window_days=window)
except APIError as e:
    st.error(str(e))
    st.stop()

st.subheader(f"Key Performance Indicators — last {window} days "
             f"({kpis.get('period_start')} to {kpis.get('period_end')})")

c1, c2, c3, c4 = st.columns(4)
c1.metric("Total Carbon Emissions", f"{kpis['total_carbon_emissions_kg']:,.0f} kg")
c2.metric("Carbon / Shipment", f"{kpis['avg_carbon_per_shipment_kg']:.2f} kg")
c3.metric("Total Energy", f"{kpis['total_energy_kwh']:,.0f} kWh")
c4.metric("Total Water", f"{kpis['total_water_litres']:,.0f} L")

c5, c6, c7, c8 = st.columns(4)
c5.metric("Total Waste", f"{kpis['total_waste_kg']:,.0f} kg")
c6.metric("Recycling Rate", f"{kpis['recycling_rate_pct']:.1f}%")
c7.metric("Renewable Energy", f"{kpis['renewable_energy_pct']:.1f}%")
c8.metric("Carbon Intensity", f"{kpis['carbon_intensity_kg_per_unit']:.3f} kg/unit")

st.divider()

col_left, col_right = st.columns([3, 2])

with col_left:
    st.subheader("Emission Trends by Facility Type")
    try:
        monthly = pd.DataFrame(get_hotspots_monthly())
        if not monthly.empty:
            fig = px.line(
                monthly, x="month", y="total_emissions_kg", color="facility_type",
                markers=True, labels={"total_emissions_kg": "Emissions (kg CO2)", "month": "Month"},
            )
            fig.update_layout(height=380, legend_title="Facility Type")
            st.plotly_chart(fig, width='stretch')
        else:
            st.info("No historical data available yet.")
    except APIError as e:
        st.error(str(e))

with col_right:
    st.subheader("Top 10 Carbon Emitters")
    try:
        hotspots = pd.DataFrame(get_hotspots(n=10, window_days=window))
        if not hotspots.empty:
            fig2 = px.bar(
                hotspots.sort_values("total_emissions_kg"),
                x="total_emissions_kg", y="facility_id", color="facility_type",
                orientation="h", labels={"total_emissions_kg": "kg CO2", "facility_id": ""},
            )
            fig2.update_layout(height=380, showlegend=False)
            st.plotly_chart(fig2, width='stretch')
        else:
            st.info("No emissions data available.")
    except APIError as e:
        st.error(str(e))

st.divider()
st.subheader("Facility Rankings")

try:
    scores = pd.DataFrame(get_scores(window_days=window))
    facilities = pd.DataFrame(get_facilities())
    if not scores.empty:
        merged = scores.merge(facilities[["facility_id", "facility_name", "region"]], on="facility_id", how="left")
        merged = merged[["facility_id", "facility_name", "facility_type", "region",
                          "overall_score", "classification"]].sort_values("overall_score", ascending=False)

        def _badge(c):
            colors = {"Excellent": "🟢", "Good": "🔵", "Average": "🟡", "Needs Attention": "🔴"}
            return f"{colors.get(c, '')} {c}"

        merged["classification"] = merged["classification"].apply(_badge)
        merged = merged.rename(columns={
            "facility_id": "Facility ID", "facility_name": "Name", "facility_type": "Type",
            "region": "Region", "overall_score": "Score", "classification": "Status",
        })
        st.dataframe(merged, width='stretch', hide_index=True)
    else:
        st.info("No scores available yet.")
except APIError as e:
    st.error(str(e))
