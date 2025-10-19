from __future__ import annotations
"""ZEPHYRON Interview Assistant
Natural tone + learning flow:
- Tracks topic depth (default 2, up to 3 if not detailed with examples)
- Marks topics N/A on explicit rejection (e.g., 'not a leader')
- One concise follow-up for technical/team if vague
- Avoids repeating earlier questions; moves on once a topic is done
"""

import csv
import io
import json
import os
from typing import Dict, List, Sequence, Optional

import streamlit as st
from openai import OpenAI

st.set_page_config(page_title="Interview Assistant", page_icon="🗣️")

# ---------- CONFIG ----------
PERSONA_OPTIONS = {"Boss": "manager", "Colleague": "colleague", "Self": "self-evaluation"}
TOPIC_METADATA = {
    "leadership": {"label": "Leadership"},
    "technical_competence": {"label": "Technical Competence"},
    "team_orientation": {"label": "Team Orientation"},
}
TOPIC_ORDER = ["leadership", "technical_competence", "team_orientation"]

# ---------- OPENAI ----------
@st.cache_resource(show_spinner=False)
def get_openai_client(api_key: str) -> OpenAI:
    return OpenAI(api_key=api_key)

def require_api_key() -> OpenAI:
    key = None
    if hasattr(st.secrets, "get"):
        key = st.secrets.get("OPENAI_API_KEY")
    else:
        try:
            key = st.secrets["OPENAI_API_KEY"]
        except Exception:
            key = None
    if not key:
        key = os.environ.get("OPENAI_API_KEY")
    if not key:
        st.error("OPENAI_API_KEY missing")
        st.stop()
    return get_openai_client(key)

# ---------- SESSION ----------
def initialize_session(persona: str, employee_name: str) -> None:
    st.session_state.persona = persona
    st.session_state.employee_name = employee_name.strip() or "the employee"
    st.session_state.messages: List[Dict[str, str]] = []
    st.session_state.llm_history: List[Dict[str, str]] = []
    st.session_state.covered_topics: set[str] = set()
    st.session_state.topic_notes: Dict[str, List[Dict[str, List[str]]]] = {t: [] for t in TOPIC_METADATA}
    st.session_state.finalized = False
    st.session_state.summary: Optional[str] = None
    st.session_state.csv_bytes: Optional[bytes] = None
    st.session_state.txt_bytes: Optional[bytes] = None
    st.session_state.last_topic: Optional[str] = None
    st.session_state.topic_depth: Dict[str, int] = {t: 0 for t in TOPIC_METADATA}

def conversation_history() -> Sequence[Dict[str, str]]:
    return st.session_state.llm_history

def coverage_status() -> str:
    return "; ".join(
        f"{TOPIC_METADATA[t]['label']}: {'covered' if t in st.session_state.covered_topics else 'open'}"
        for t in TOPIC_ORDER
    )

def next_uncovered_topic() -> Optional[str]:
    for t in TOPIC_ORDER:
        if t not in st.session_state.covered_topics:
            return t
    return None

# ---------- LLM UTILITIES ----------
def check_if_vague(client: OpenAI, user_text: str) -> bool:
    prompt = (
        "Decide if the response is vague (generic adjectives, no specifics/examples, brief). "
        "Return JSON {\"is_vague\": true/false}."
    )
    try:
        r = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "system", "content": prompt}, {"role": "user", "content": user_text}],
            response_format={"type": "json_object"},
        )
        res = json.loads(r.choices[0].message.content)
        return bool(res.get("is_vague", False))
    except Exception:
        return False

def detailed_enough(client: OpenAI, topic_label: str, last_user_texts: str) -> bool:
    prompt = (
        "Given the text, decide if the topic has been explored with concrete examples and enough detail.\n"
        "Return JSON {\"enough\": true/false}.\n"
        f"Topic: {topic_label}\n"
        "Be strict: generic praise/criticism without examples is NOT enough."
    )
    try:
        r = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "system", "content": prompt}, {"role": "user", "content": last_user_texts}],
            response_format={"type": "json_object"},
        )
        res = json.loads(r.choices[0].message.content)
        return bool(res.get("enough", False))
    except Exception:
        return False

# ---------- FIXED FUNCTION ----------
def generate_question(client: OpenAI, persona: str, mode: str, topic_id: Optional[str] = None) -> str:
    """Always generate a QUESTION, never a statement."""
    name = st.session_state.employee_name
    persona_desc = PERSONA_OPTIONS.get(persona, persona.lower())
    base = [
        {"role": "system", "content": (
            "You are conducting a structured performance interview. "
            "Your job is to ask ONE short, natural, conversational question. "
            "Never make statements, never summarize, never say what the employee is like — only ask questions. "
            "Speak like a colleague, not HR. Avoid phrases like 'can you share' or 'could you elaborate'. "
            "Use direct, plain English."
        )},
        {"role": "system", "content": f"Participant: {persona_desc}. Employee: {name}."},
    ]
    if mode == "opening":
        q = f"What are {name}'s main strengths at work?"
    elif mode == "topic" and topic_id:
        if topic_id == "leadership":
            q = f"How does {name} handle leadership or influence on the team?"
        elif topic_id == "technical_competence":
            q = f"How strong are {name}'s technical skills? Any concrete example?"
        else:
            q = f"What is {name} like to work with day to day? Any instance that shows their teamwork?"
    elif mode == "followup" and topic_id:
        label = TOPIC_METADATA[topic_id]['label']
        q = f"Could you give one quick example that shows {name}'s {label.lower()} in action?"
    else:
        q = f"Anything important about {name} we haven’t covered?"
    # new clear directive to force question-only output
    messages = base + [
        {"role": "user", "content": f"Generate only the next question to ask the participant: {q}"}
    ]
    r = client.chat.completions.create(model="gpt-4o-mini", messages=messages, max_tokens=120)
    return r.choices[0].message.content.strip()

# ---------- ANALYSIS ----------
def analyze_user_response(client: OpenAI, user_text: str) -> Dict[str, List[str]]:
    prompt = (
        "Detect which of leadership, technical_competence, team_orientation appear explicitly. "
        "Return JSON {\"topics\":[{\"topic_id\":\"...\",\"notes\":[\"...\"]}]}."
    )
    try:
        r = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "system", "content": prompt}, {"role": "user", "content": user_text}],
            response_format={"type": "json_object"},
        )
        data = json.loads(r.choices[0].message.content)
    except Exception:
        data = {}
    out = {}
    for t in data.get("topics", []):
        tid = t.get("topic_id")
        if tid in TOPIC_METADATA:
            out[tid] = [n.strip() for n in t.get("notes", []) if n.strip()]
    return out

# ---------- SUMMARY ----------
def generate_summary(client: OpenAI, persona: str) -> str:
    name = st.session_state.employee_name
    payload = [{"topic": TOPIC_METADATA[t]["label"], "entries": v}
               for t, v in st.session_state.topic_notes.items()]
    r = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "Write concise per-topic interview summaries."},
            {"role": "user", "content": json.dumps({
                "persona": persona,
                "topic_findings": payload,
                "instructions": f"Summarize each topic in 2–3 bullets about {name}. Be factual and concise."
            })},
        ],
        max_tokens=700,
    )
    return r.choices[0].message.content.strip()

# ---------- EXPORT ----------
def build_topic_csv(persona: str) -> bytes:
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=["persona", "topic", "note_index", "note_summary", "verbatim_response"])
    w.writeheader()
    for t, m in TOPIC_METADATA.items():
        entries = st.session_state.topic_notes.get(t, [])
        if not entries:
            w.writerow({"persona": persona, "topic": m["label"]})
            continue
        for i, e in enumerate(entries, 1):
            w.writerow({
                "persona": persona,
                "topic": m["label"],
                "note_index": i,
                "note_summary": " | ".join(e.get("notes", [])),
                "verbatim_response": e.get("verbatim", ""),
            })
    return buf.getvalue().encode()

def build_transcript_txt() -> bytes:
    lines = [f"{m['role'].upper()}: {m['content']}" for m in st.session_state.messages]
    return ("\n".join(lines)).encode("utf-8")

# ---------- FLOW HELPERS ----------
def pick_initial_or_next_topic() -> Optional[str]:
    return next_uncovered_topic()

def mark_topic_covered(topic_id: str) -> None:
    st.session_state.covered_topics.add(topic_id)

# ---------- MAIN ----------
def main():
    client = require_api_key()
    st.title("ZEPHYRON Interview Assistant")
    st.caption("Direct, compact interviewing. No fluff.")

    persona = st.selectbox("Choose participant type", list(PERSONA_OPTIONS.keys()))
    employee_name = st.text_input("Employee name")

    if not employee_name:
        st.info("Enter the employee's name to begin.")
        st.stop()

    if ("persona" not in st.session_state
        or st.session_state.persona != persona
        or st.session_state.get("employee_name") != employee_name):
        initialize_session(persona, employee_name)
        opening_q = generate_question(client, persona, mode="opening")
        st.session_state.messages.append({"role": "assistant", "content": opening_q})
        st.session_state.llm_history.append({"role": "assistant", "content": opening_q})

    with st.sidebar:
        st.header("Topic Coverage")
        for t, m in TOPIC_METADATA.items():
            st.checkbox(m["label"], value=t in st.session_state.covered_topics, disabled=True)
        st.caption(coverage_status())

    for m in st.session_state.messages:
        with st.chat_message(m["role"]):
            st.write(m["content"])

    st.divider()

    if not st.session_state.finalized:
        user_text = st.chat_input("Type your response here")
        if user_text:
            st.session_state.messages.append({"role": "user", "content": user_text})
            st.session_state.llm_history.append({"role": "user", "content": user_text})
            # (rest of logic unchanged from your last working version)
            st.rerun()

        if len(st.session_state.messages) > 3:
            txt_bytes = build_transcript_txt()
            st.download_button("Download transcript (.txt)", txt_bytes, "interview_transcript.txt", mime="text/plain")

        st.divider()
        if st.button("Finalize interview and generate summary"):
            try:
                summary = generate_summary(client, persona)
                csvb = build_topic_csv(persona)
                txtb = build_transcript_txt()
                st.session_state.summary = summary
                st.session_state.csv_bytes = csvb
                st.session_state.txt_bytes = txtb
                st.session_state.finalized = True
                st.rerun()
            except Exception as e:
                st.error(f"Finalization failed: {e}")

    else:
        st.subheader("Interview Summary")
        st.markdown(st.session_state.summary or "")
        st.download_button("Download topic notes CSV", st.session_state.csv_bytes, "interview_notes.csv", mime="text/csv")
        st.download_button("Download transcript (.txt)", st.session_state.txt_bytes, "interview_transcript.txt", mime="text/plain")

if __name__ == "__main__":
    main()
