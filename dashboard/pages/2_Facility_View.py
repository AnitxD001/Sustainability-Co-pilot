import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import streamlit as st
import pandas as pd
import plotly.express as px

from dashboard.api_client import (
    get_facilities, get_facility_history, get_facility_score,
    get_prediction, get_recommendations, train_prediction_model, APIError
)

st.title("🏭 Facility View")

try:
    facilities = pd.DataFrame(get_facilities())
except APIError as e:
    st.error(str(e))
    st.stop()

if facilities.empty:
    st.info("No facilities found. Run the data generator and seed script first.")
    st.stop()

facilities["label"] = facilities["facility_id"] + " — " + facilities["facility_name"]
choice = st.selectbox("Select a facility", facilities["label"])
facility_id = facilities.loc[facilities["label"] == choice, "facility_id"].iloc[0]

days = st.slider("History window (days)", 14, 365, 90)

# ---------------------------------------------------------------------------
# 1. Historical Data — every metric, charted one after another
# ---------------------------------------------------------------------------
st.header("📈 Historical Data")

try:
    history = pd.DataFrame(get_facility_history(facility_id, days=days))
except APIError as e:
    st.error(str(e))
    history = pd.DataFrame()

if not history.empty:
    history["date"] = pd.to_datetime(history["date"])

    metric_specs = [
        ("carbon_emissions_kg", "Carbon Emissions (kg CO2)"),
        ("electricity_kwh", "Electricity Consumption (kWh)"),
        ("diesel_litres", "Diesel Consumption (L)"),
        ("water_litres", "Water Usage (L)"),
        ("waste_generated_kg", "Waste Generated (kg)"),
        ("renewable_energy_pct", "Renewable Energy (%)"),
        ("recycling_pct", "Recycling Rate (%)"),
        ("shipments_processed", "Shipments Processed"),
    ]

    for col, title in metric_specs:
        st.subheader(title)
        fig = px.line(history, x="date", y=col, markers=False)
        fig.update_layout(height=280, margin=dict(t=10, b=10))
        st.plotly_chart(fig, width="stretch", key=f"hist_{col}")

    with st.expander("Raw data"):
        st.dataframe(history, width="stretch", hide_index=True)
else:
    st.info("No history available for this facility.")

st.divider()

# ---------------------------------------------------------------------------
# 2. Predictions
# ---------------------------------------------------------------------------
st.header("🔮 Predictions")
st.caption("Forecasts use the XGBoost carbon emission model (see `ml/prediction.py`), "
           "trained on operational metrics across all facilities.")

if st.button("Train / Retrain Model"):
    with st.spinner("Training XGBoost model..."):
        try:
            result = train_prediction_model()
            st.success(f"Trained. R² = {result['metrics']['r2']:.3f}, "
                       f"MAE = {result['metrics']['mae']:.1f} kg")
        except APIError as e:
            st.error(str(e))

try:
    pred = get_prediction(facility_id)
    m1, m2, m3 = st.columns(3)
    m1.metric("Tomorrow", f"{pred['tomorrow']['predicted_emissions_kg']:,.1f} kg",
               help=pred['tomorrow']['predicted_date'])
    m2.metric("Next Week", f"{pred['next_week']['predicted_emissions_kg']:,.1f} kg",
               help=pred['next_week']['predicted_date'])
    m3.metric("Next Month", f"{pred['next_month']['predicted_emissions_kg']:,.1f} kg",
               help=pred['next_month']['predicted_date'])
    st.caption(f"Model performance — MAE: {pred['model_metrics']['mae']:.1f} kg, "
               f"RMSE: {pred['model_metrics']['rmse']:.1f} kg, "
               f"R²: {pred['model_metrics']['r2']:.3f}")
except APIError as e:
    if "not trained" in str(e).lower():
        st.warning("Model not trained yet. Click 'Train / Retrain Model' above.")
    else:
        st.error(str(e))

st.divider()

# ---------------------------------------------------------------------------
# 3. Sustainability Score
# ---------------------------------------------------------------------------
st.header("🏆 Sustainability Score")

try:
    score = get_facility_score(facility_id)
    st.metric("Overall Sustainability Score", f"{score['overall_score']:.1f} / 100", score["classification"])

    sub_df = pd.DataFrame({
        "Component": ["Carbon", "Energy", "Water", "Waste", "Renewable", "Recycling"],
        "Score": [score["carbon_subscore"], score["energy_subscore"], score["water_subscore"],
                  score["waste_subscore"], score["renewable_subscore"], score["recycling_subscore"]],
    })
    fig = px.bar(sub_df, x="Component", y="Score", range_y=[0, 100],
                 color="Score", color_continuous_scale="RdYlGn")
    fig.update_layout(height=350, coloraxis_showscale=False)
    st.plotly_chart(fig, width="stretch", key="score_breakdown")
    st.caption(f"Period: {score['period_start']} to {score['period_end']} — "
               "scored relative to peer facilities of the same type.")
except APIError as e:
    st.error(str(e))

st.divider()

# ---------------------------------------------------------------------------
# 4. Recommendations
# ---------------------------------------------------------------------------
st.header("💡 Recommendations")

try:
    recs = get_recommendations(facility_id)
    if not recs:
        st.info("No recommendations available.")
    for r in recs:
        priority_icon = {"High": "🔴", "Medium": "🟡", "Low": "🟢"}.get(r["priority"], "")
        with st.container(border=True):
            st.markdown(f"**{priority_icon} {r['category']}** — *{r['priority']} priority*")
            st.write(r["recommendation_text"])
            st.caption(f"Estimated impact: {r['estimated_impact']}")
except APIError as e:
    st.error(str(e))
