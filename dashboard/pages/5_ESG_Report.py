import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import streamlit as st

st.title("📄 Monthly ESG Report")
st.caption("Generates a markdown + PDF ESG report from the computed KPIs, scores, anomalies, "
           "hotspots, and trends. The executive summary paragraph is written by the Groq LLM, "
           "grounded strictly in those computed numbers.")

window = st.slider("Reporting window (days)", 7, 90, 30)
use_llm = st.checkbox(
    "Generate LLM executive summary (requires GROQ_API_KEY)", value=True,
    help="If unchecked, or if no API key is configured, the report is generated with computed "
         "metrics only and a placeholder summary."
)

if st.button("Generate Report", type="primary"):
    with st.spinner("Computing metrics and generating report..."):
        try:
            # Generated directly (not via HTTP) since this writes files to disk and the
            # dashboard runs on the same host as the rest of the platform.
            from reports.generator import generate_report
            result = generate_report(window_days=window, use_llm=use_llm)
            st.session_state["last_report"] = result
            st.success("Report generated.")
        except Exception as e:
            st.error(f"Report generation failed: {e}")

if "last_report" in st.session_state:
    result = st.session_state["last_report"]
    data = result["data"]

    st.divider()
    st.subheader("Executive Summary")
    st.write(data["executive_summary"] or "_Not generated._")

    st.subheader("Preview")
    md_text = Path(result["markdown_path"]).read_text()
    st.markdown(md_text)

    st.divider()
    col1, col2 = st.columns(2)
    with col1:
        st.download_button(
            "⬇️ Download Markdown",
            data=md_text,
            file_name=Path(result["markdown_path"]).name,
            mime="text/markdown",
        )
    with col2:
        pdf_bytes = Path(result["pdf_path"]).read_bytes()
        st.download_button(
            "⬇️ Download PDF",
            data=pdf_bytes,
            file_name=Path(result["pdf_path"]).name,
            mime="application/pdf",
        )
