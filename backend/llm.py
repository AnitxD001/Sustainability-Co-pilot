"""
AI Sustainability Assistant (LLM layer)
=========================================
The LLM (Groq, meta-llama/llama-4-scout-17b-16e-instruct) is deliberately NOT
responsible for prediction, scoring, anomaly detection, or recommendations —
those come from the deterministic/ML modules. The LLM's job is limited to:

    - answering natural-language questions
    - explaining model outputs
    - summarizing performance
    - generating executive-friendly report prose

To keep answers grounded, the backend pulls relevant computed metrics from
the database (RAG-style) and injects them into the prompt as context; the
system prompt explicitly forbids the model from inventing numbers not
present in that context.

The API key is read from the GROQ_API_KEY environment variable. It is never
hardcoded. If the key is missing, functions raise a clear RuntimeError
rather than failing silently or fabricating a response.
"""

import os
import json
import requests

GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
GROQ_MODEL = "meta-llama/llama-4-scout-17b-16e-instruct"

SYSTEM_PROMPT = """You are an AI Sustainability Assistant for a supply chain ESG platform.

Rules:
- Only use the facts provided in the CONTEXT block below. Do not invent numbers, facility names, or trends that are not present in the context.
- If the context doesn't contain enough information to answer, say so plainly and suggest what data would be needed.
- You do not run predictions, scoring, or anomaly detection yourself — those are pre-computed and given to you in the context. Explain and summarize them; do not recompute or contradict them.
- Be concise and executive-friendly. Use plain language, not jargon.
- When asked for a report or summary, structure it with short headers (Executive Summary, Trends, Recommendations) but keep it readable in chat.
"""


def _get_api_key() -> str:
    key = os.getenv("GROQ_API_KEY")
    if not key:
        raise RuntimeError(
            "GROQ_API_KEY is not set. Add it to your .env file or environment — never hardcode it."
        )
    return key


def build_context(question: str, db_data: dict) -> str:
    """
    Assemble a RAG-style context block from pre-computed query results.

    db_data is expected to already contain only the data relevant to the
    question (the caller — backend/routes.py — decides what to fetch based
    on simple intent detection, e.g. "highest emitter" -> hotspot query,
    "anomalies" -> anomaly table, "score" -> scoring table, etc).
    """
    return json.dumps(db_data, indent=2, default=str)


def ask_assistant(question: str, db_data: dict, temperature: float = 0.2, max_tokens: int = 700) -> str:
    """
    Send a grounded question to the Groq LLM and return the text response.

    Parameters
    ----------
    question : the user's natural-language question
    db_data  : dict of pre-fetched, relevant metrics/scores/anomalies/recommendations
               to ground the answer in (RAG-style context, NOT raw row dumps)
    """
    api_key = _get_api_key()
    context_block = build_context(question, db_data)

    payload = {
        "model": GROQ_MODEL,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"CONTEXT:\n{context_block}\n\nQUESTION:\n{question}"},
        ],
    }
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    response = requests.post(GROQ_API_URL, headers=headers, json=payload, timeout=30)
    response.raise_for_status()
    data = response.json()
    return data["choices"][0]["message"]["content"]


def generate_executive_summary(metrics_snapshot: dict) -> str:
    """Convenience wrapper for Module 7 (ESG report) — same grounding rules apply."""
    prompt = (
        "Generate a concise executive summary (3-5 sentences) of this month's "
        "sustainability performance based strictly on the data provided."
    )
    return ask_assistant(prompt, metrics_snapshot)
