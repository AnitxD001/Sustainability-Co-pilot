import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import streamlit as st
import pandas as pd

from dashboard.api_client import get_insights, get_anomalies, APIError

st.title("💡 AI Insights")
st.caption("Auto-generated observations from the scoring, anomaly, and trend-analysis modules.")

window = st.sidebar.slider("Insight window (days)", min_value=14, max_value=120, value=30, step=1)

try:
    insights = get_insights(window_days=window)
except APIError as e:
    st.error(str(e))
    st.stop()

col1, col2 = st.columns(2)

with col1:
    st.subheader("📈 Improving Facilities")
    if insights["positive_trends"]:
        for fid, pct in insights["positive_trends"]:
            st.success(f"**{fid}** reduced emissions by **{abs(pct):.1f}%** vs. the prior period.")
    else:
        st.info("No significant improving trends this period.")

with col2:
    st.subheader("📉 Worsening Facilities")
    if insights["negative_trends"]:
        for fid, pct in insights["negative_trends"]:
            st.warning(f"**{fid}** increased emissions by **{pct:.1f}%** vs. the prior period.")
    else:
        st.info("No significant worsening trends this period.")

st.divider()
st.subheader("⚠️ High-Severity Anomalies")
st.caption(f"{insights['total_anomalies']} total anomalies detected in the last {window} days.")

if insights["high_severity_anomalies"]:
    for a in insights["high_severity_anomalies"]:
        with st.container(border=True):
            c1, c2 = st.columns([3, 1])
            c1.markdown(f"**{a['facility_id']}** ({a['facility_type']}) — {a['date']}")
            c1.write(a["likely_cause"])
            c2.markdown("🔴 **High**")
else:
    st.info("No high-severity anomalies detected in this window.")

st.divider()
st.subheader("All Anomalies (this window)")
try:
    all_anom = pd.DataFrame(get_anomalies(window_days=window))
    if not all_anom.empty:
        display_df = all_anom[["facility_id", "facility_type", "date", "severity", "likely_cause"]]
        st.dataframe(display_df, width='stretch', hide_index=True)
    else:
        st.info("No anomalies in this window.")
except APIError as e:
    st.error(str(e))
