"""
Streamlit Dashboard — Entry Point
====================================
Run with:

    streamlit run app.py

Requires the FastAPI backend running separately for most pages:

    uvicorn backend.api:app --reload --port 8000

(The ESG Report page is the one exception — it calls the report generator
directly rather than over HTTP, since it writes files to disk.)
"""

import streamlit as st
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PAGES_DIR = ROOT / "dashboard" / "pages"

st.set_page_config(
    page_title="AI Sustainability Copilot",
    page_icon="🌱",
    layout="wide",
)

pages = [
    st.Page(str(PAGES_DIR / "1_Home.py"), title="Home", icon="🏠", default=True),
    st.Page(str(PAGES_DIR / "2_Facility_View.py"), title="Facility View", icon="🏭"),
    st.Page(str(PAGES_DIR / "3_AI_Insights.py"), title="AI Insights", icon="💡"),
    st.Page(str(PAGES_DIR / "4_Chatbot.py"), title="Chatbot", icon="💬"),
    st.Page(str(PAGES_DIR / "5_ESG_Report.py"), title="ESG Report", icon="📄"),
]

pg = st.navigation(pages)
pg.run()
