import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import streamlit as st

from dashboard.api_client import get_facilities, post_chat, APIError

st.title("💬 AI Sustainability Assistant")
st.caption("Ask questions in plain English. Answers are grounded in the platform's computed metrics — "
           "the LLM explains and summarizes, it does not invent numbers.")

try:
    facilities = [f["facility_id"] for f in get_facilities()]
except APIError as e:
    st.error(str(e))
    facilities = []

with st.sidebar:
    scope = st.selectbox("Scope question to a facility (optional)", ["(none)"] + facilities)
    facility_id = None if scope == "(none)" else scope
    st.markdown("**Example questions**")
    st.caption("Which supplier emitted the highest carbon this month?")
    st.caption("Show the least sustainable warehouse.")
    st.caption("What anomalies were detected this week?")
    st.caption("Summarize this month's sustainability performance.")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

question = st.chat_input("Ask the sustainability assistant...")

if question:
    st.session_state.chat_history.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.write(question)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                result = post_chat(question, facility_id=facility_id)
                answer = result["answer"]
                st.write(answer)
                st.caption(f"Grounded in: {', '.join(result['context_used'])}")
                st.session_state.chat_history.append({"role": "assistant", "content": answer})
            except APIError as e:
                msg = str(e)
                if "GROQ_API_KEY" in msg:
                    st.warning(
                        "The Groq API key isn't configured yet. Add `GROQ_API_KEY` to your `.env` "
                        "file (see `.env.example`) to enable the assistant."
                    )
                else:
                    st.error(msg)
