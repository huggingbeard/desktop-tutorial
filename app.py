"""Interactive Streamlit app for an LLM-guided interview assistant.
Revised 2025-10-19 for smoother UX and cleaner logic.
"""
from __future__ import annotations

import csv
import io
import json
import os
from typing import Dict, List, Sequence

import streamlit as st
from openai import OpenAI

st.set_page_config(page_title="Interview Assistant", page_icon="🗣️")

PERSONA_OPTIONS = {
    "Boss": "manager",
    "Colleague": "colleague",
    "Self": "self-evaluation",
}

TOPIC_METADATA: Dict[str, Dict[str, str]] = {
    "leadership": {"label": "Leadership"},
    "technical_competence": {"label": "Technical Competence"},
    "team_orientation": {"label": "Team Orientation"},
}


# ----------  API client ----------

@st.cache_resource(show_spinner=False)
def get_openai_client(api_key: str) -> OpenAI:
    return OpenAI(api_key=api_key)


def require_api_key() -> OpenAI:
    """Keep existing behavior exactly."""
    api_key: str | None = None
    if hasattr(st.secrets, "get"):
        api_key = st.secrets.get("OPENAI_API_KEY")
    else:
        try:
            api_key = st.secrets["OPENAI_API_KEY"]
        except Exception:
            api_key = None
    if not api_key:
        api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        st.error(
            "OPENAI_API_KEY is not set. Provide it via Streamlit secrets or environment."
        )
        st.stop()
    return get_openai_client(api_key)


# ----------  Session init ----------

def initialize_session(persona: str) -> None:
    st.session_state.persona = persona
    st.session_state.messages = []
    st.session_state.llm_history = []
    st.session_state.covered_topics = []
    st.session_state.topic_notes = {t: [] for t in TOPIC_METADATA}
    st.session_state.finalized = False
    st.session_state.summary = None
    st.session_state.csv_bytes = None
    st.session_state.just_asked_followup = False
    st.session_state.followup_count = 0
    st.session_state.topic_depth = {t: 0 for t in TOPIC_METADATA}
    st.session_state.history_summary = ""


def conversation_history() -> Sequence[Dict[str, str]]:
    return st.session_state.llm_history[-12:]  # trim to last 12 turns


def persona_description(persona: str) -> str:
    descriptor = PERSONA_OPTIONS.get(persona, persona.lower())
    return f"You are speaking with a {descriptor} of the employee."


# ----------  Utilities ----------

def coverage_status() -> str:
    seg = []
    for t, m in TOPIC_METADATA.items():
        status = "covered" if t in st.session_state.covered_topics else "open"
        seg.append(f"{m['label']}: {status}")
    return "; ".join(seg)


def next_uncovered_topic() -> str | None:
    for t in TOPIC_METADATA:
        if t not in st.session_state.covered_topics:
            return t
    return None


# ----------  Core LLM calls ----------

def single_llm_decision(client: OpenAI, persona: str, user_text: str | None) -> Dict:
    """
    Consolidated call:
    - decide next question
    - detect topics and vagueness
    Returns dict with keys: next_msg, topics, notes, is_vague
    """
    system = (
        "You are an interviewing assistant conducting a structured qualitative interview.\n"
        "You must always return valid JSON of this exact shape:\n"
        '{\n'
        '  "next_msg": "string",\n'
        '  "topics": [{"topic_id": "...", "notes": ["..."]}],\n'
        '  "is_vague": true/false\n'
        '}\n\n'
        "Interview goals: cover leadership, technical_competence, team_orientation. "
        "Ask one warm, professional question at a time. "
        "If all are covered, wrap up politely. "
        "If the last reply was vague, ask for a specific example."
    )
    status = (
        f"Participant type: {persona}. {persona_description(persona)} "
        f"Coverage: {coverage_status()}. "
    )
    if not conversation_history():
        status += (
            "This is the beginning of the interview; start with an open, reflective question "
            "on the employee’s overall strengths."
        )

    user_block = user_text or "No prior response yet."

    resp = client.chat.completions.create(
        model="gpt-5-nano",
        messages=[
            {"role": "system", "content": system},
            *conversation_history(),
            {"role": "system", "content": status},
            {"role": "user", "content": user_block},
        ],
        response_format={"type": "json_object"},
        max_tokens=500,
        temperature=0,
    )

    try:
        payload = json.loads(resp.choices[0].message.content)
    except Exception:
        payload = {"next_msg": "Could you elaborate a bit more?", "topics": [], "is_vague": False}
    return payload


def generate_summary(client: OpenAI, persona: str) -> str:
    topic_payload = []
    for t, entries in st.session_state.topic_notes.items():
        topic_payload.append(
            {"topic": TOPIC_METADATA[t]["label"], "entries": entries}
        )

    resp = client.chat.completions.create(
        model="gpt-5-nano",
        messages=[
            {"role": "system", "content": "Create concise, professional summaries by topic."},
            {
                "role": "user",
                "content": json.dumps(
                    {
                        "persona": persona,
                        "topic_findings": topic_payload,
                        "instructions": (
                            "Write 2-3 bullet points per topic that synthesize notes; "
                            "mention if limited detail."
                        ),
                    }
                ),
            },
        ],
        max_tokens=600,
        temperature=0,
    )
    return resp.choices[0].message.content.strip()


# ----------  Notes + CSV ----------

def record_topic_notes(user_text: str, topics: List[Dict[str, List[str]]]) -> None:
    for t in topics:
        tid = t.get("topic_id")
        notes = [n.strip() for n in t.get("notes", []) if n.strip()]
        if tid in TOPIC_METADATA and notes:
            st.session_state.topic_notes[tid].append(
                {"verbatim": user_text.strip(), "notes": notes}
            )
            st.session_state.topic_depth[tid] += 1
            if st.session_state.topic_depth[tid] >= 2 and tid not in st.session_state.covered_topics:
                st.session_state.covered_topics.append(tid)


def build_topic_csv(persona: str) -> bytes:
    buf = io.StringIO()
    w = csv.DictWriter(
        buf,
        fieldnames=["persona", "topic", "note_index", "note_summary", "verbatim_response"],
    )
    w.writeheader()
    for t, meta in TOPIC_METADATA.items():
        entries = st.session_state.topic_notes.get(t, [])
        if not entries:
            w.writerow({"persona": persona, "topic": meta["label"]})
            continue
        for i, e in enumerate(entries, 1):
            w.writerow(
                {
                    "persona": persona,
                    "topic": meta["label"],
                    "note_index": i,
                    "note_summary": " | ".join(e["notes"]),
                    "verbatim_response": e["verbatim"],
                }
            )
    return buf.getvalue().encode("utf-8")


# ----------  Streamlit main ----------

def main() -> None:
    client = require_api_key()

    st.title("ZEPHYRON Interview Assistant")
    st.caption("LLM-guided structured interviews for qualitative evaluations.")

    persona = st.selectbox("Choose participant type", list(PERSONA_OPTIONS.keys()))

    if "persona" not in st.session_state or st.session_state.persona != persona:
        initialize_session(persona)
        # opening message
        opening = single_llm_decision(client, persona, None)["next_msg"]
        st.session_state.messages.append({"role": "assistant", "content": opening})
        st.session_state.llm_history.append({"role": "assistant", "content": opening})

    with st.sidebar:
        st.header("Topic Coverage")
        prog = len(st.session_state.covered_topics) / len(TOPIC_METADATA)
        st.progress(prog)
        for t, m in TOPIC_METADATA.items():
            st.checkbox(m["label"], value=t in st.session_state.covered_topics, disabled=True)
        st.caption(f"Follow-ups used: {st.session_state.followup_count}/5")
        st.markdown("Finalize to get summary and CSV export.")

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

    if st.session_state.finalized:
        st.info("Interview finalized. Download or copy summary below.")
    else:
        user_text = st.chat_input("Type your response here")
        if user_text:
            st.session_state.messages.append({"role": "user", "content": user_text})
            st.session_state.llm_history.append({"role": "user", "content": user_text})

            payload = single_llm_decision(client, persona, user_text)
            record_topic_notes(user_text, payload.get("topics", []))
            is_vague = payload.get("is_vague", False)

            # follow-up control
            if is_vague and st.session_state.followup_count < 5:
                st.session_state.followup_count += 1
                st.session_state.just_asked_followup = True
            else:
                st.session_state.just_asked_followup = False

            next_msg = payload.get("next_msg", "Could you elaborate?")
            st.session_state.messages.append({"role": "assistant", "content": next_msg})
            st.session_state.llm_history.append({"role": "assistant", "content": next_msg})
            st.rerun()

    st.divider()
    if st.button("Finalize interview and generate summary", disabled=st.session_state.finalized):
        summary = generate_summary(client, persona)
        csv_bytes = build_topic_csv(persona)
        st.session_state.summary = summary
        st.session_state.csv_bytes = csv_bytes
        st.session_state.finalized = True
        st.rerun()

    if st.session_state.finalized and st.session_state.summary:
        st.subheader("Interview Summary")
        st.markdown(st.session_state.summary)
        st.download_button(
            "Download topic notes CSV",
            data=st.session_state.csv_bytes,
            file_name="interview_notes.csv",
            mime="text/csv",
        )


if __name__ == "__main__":
    main()
"""Interactive Streamlit app for an LLM-guided interview assistant.
Revised 2025-10-19 for smoother UX and cleaner logic.
"""
from __future__ import annotations

import csv
import io
import json
import os
from typing import Dict, List, Sequence

import streamlit as st
from openai import OpenAI

st.set_page_config(page_title="Interview Assistant", page_icon="🗣️")

PERSONA_OPTIONS = {
    "Boss": "manager",
    "Colleague": "colleague",
    "Self": "self-evaluation",
}

TOPIC_METADATA: Dict[str, Dict[str, str]] = {
    "leadership": {"label": "Leadership"},
    "technical_competence": {"label": "Technical Competence"},
    "team_orientation": {"label": "Team Orientation"},
}


# ----------  API client ----------

@st.cache_resource(show_spinner=False)
def get_openai_client(api_key: str) -> OpenAI:
    return OpenAI(api_key=api_key)


def require_api_key() -> OpenAI:
    """Keep existing behavior exactly."""
    api_key: str | None = None
    if hasattr(st.secrets, "get"):
        api_key = st.secrets.get("OPENAI_API_KEY")
    else:
        try:
            api_key = st.secrets["OPENAI_API_KEY"]
        except Exception:
            api_key = None
    if not api_key:
        api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        st.error(
            "OPENAI_API_KEY is not set. Provide it via Streamlit secrets or environment."
        )
        st.stop()
    return get_openai_client(api_key)


# ----------  Session init ----------

def initialize_session(persona: str) -> None:
    st.session_state.persona = persona
    st.session_state.messages = []
    st.session_state.llm_history = []
    st.session_state.covered_topics = []
    st.session_state.topic_notes = {t: [] for t in TOPIC_METADATA}
    st.session_state.finalized = False
    st.session_state.summary = None
    st.session_state.csv_bytes = None
    st.session_state.just_asked_followup = False
    st.session_state.followup_count = 0
    st.session_state.topic_depth = {t: 0 for t in TOPIC_METADATA}
    st.session_state.history_summary = ""


def conversation_history() -> Sequence[Dict[str, str]]:
    return st.session_state.llm_history[-12:]  # trim to last 12 turns


def persona_description(persona: str) -> str:
    descriptor = PERSONA_OPTIONS.get(persona, persona.lower())
    return f"You are speaking with a {descriptor} of the employee."


# ----------  Utilities ----------

def coverage_status() -> str:
    seg = []
    for t, m in TOPIC_METADATA.items():
        status = "covered" if t in st.session_state.covered_topics else "open"
        seg.append(f"{m['label']}: {status}")
    return "; ".join(seg)


def next_uncovered_topic() -> str | None:
    for t in TOPIC_METADATA:
        if t not in st.session_state.covered_topics:
            return t
    return None


# ----------  Core LLM calls ----------

def single_llm_decision(client: OpenAI, persona: str, user_text: str | None) -> Dict:
    """
    Consolidated call:
    - decide next question
    - detect topics and vagueness
    Returns dict with keys: next_msg, topics, notes, is_vague
    """
    system = (
        "You are an interviewing assistant conducting a structured qualitative interview.\n"
        "You must always return valid JSON of this exact shape:\n"
        '{\n'
        '  "next_msg": "string",\n'
        '  "topics": [{"topic_id": "...", "notes": ["..."]}],\n'
        '  "is_vague": true/false\n'
        '}\n\n'
        "Interview goals: cover leadership, technical_competence, team_orientation. "
        "Ask one warm, professional question at a time. "
        "If all are covered, wrap up politely. "
        "If the last reply was vague, ask for a specific example."
    )
    status = (
        f"Participant type: {persona}. {persona_description(persona)} "
        f"Coverage: {coverage_status()}. "
    )
    if not conversation_history():
        status += (
            "This is the beginning of the interview; start with an open, reflective question "
            "on the employee’s overall strengths."
        )

    user_block = user_text or "No prior response yet."

    resp = client.chat.completions.create(
        model="gpt-5-nano",
        messages=[
            {"role": "system", "content": system},
            *conversation_history(),
            {"role": "system", "content": status},
            {"role": "user", "content": user_block},
        ],
        response_format={"type": "json_object"},
        max_tokens=500,
        temperature=0,
    )

    try:
        payload = json.loads(resp.choices[0].message.content)
    except Exception:
        payload = {"next_msg": "Could you elaborate a bit more?", "topics": [], "is_vague": False}
    return payload


def generate_summary(client: OpenAI, persona: str) -> str:
    topic_payload = []
    for t, entries in st.session_state.topic_notes.items():
        topic_payload.append(
            {"topic": TOPIC_METADATA[t]["label"], "entries": entries}
        )

    resp = client.chat.completions.create(
        model="gpt-5-nano",
        messages=[
            {"role": "system", "content": "Create concise, professional summaries by topic."},
            {
                "role": "user",
                "content": json.dumps(
                    {
                        "persona": persona,
                        "topic_findings": topic_payload,
                        "instructions": (
                            "Write 2-3 bullet points per topic that synthesize notes; "
                            "mention if limited detail."
                        ),
                    }
                ),
            },
        ],
        max_tokens=600,
        temperature=0,
    )
    return resp.choices[0].message.content.strip()


# ----------  Notes + CSV ----------

def record_topic_notes(user_text: str, topics: List[Dict[str, List[str]]]) -> None:
    for t in topics:
        tid = t.get("topic_id")
        notes = [n.strip() for n in t.get("notes", []) if n.strip()]
        if tid in TOPIC_METADATA and notes:
            st.session_state.topic_notes[tid].append(
                {"verbatim": user_text.strip(), "notes": notes}
            )
            st.session_state.topic_depth[tid] += 1
            if st.session_state.topic_depth[tid] >= 2 and tid not in st.session_state.covered_topics:
                st.session_state.covered_topics.append(tid)


def build_topic_csv(persona: str) -> bytes:
    buf = io.StringIO()
    w = csv.DictWriter(
        buf,
        fieldnames=["persona", "topic", "note_index", "note_summary", "verbatim_response"],
    )
    w.writeheader()
    for t, meta in TOPIC_METADATA.items():
        entries = st.session_state.topic_notes.get(t, [])
        if not entries:
            w.writerow({"persona": persona, "topic": meta["label"]})
            continue
        for i, e in enumerate(entries, 1):
            w.writerow(
                {
                    "persona": persona,
                    "topic": meta["label"],
                    "note_index": i,
                    "note_summary": " | ".join(e["notes"]),
                    "verbatim_response": e["verbatim"],
                }
            )
    return buf.getvalue().encode("utf-8")


# ----------  Streamlit main ----------

def main() -> None:
    client = require_api_key()

    st.title("ZEPHYRON Interview Assistant")
    st.caption("LLM-guided structured interviews for qualitative evaluations.")

    persona = st.selectbox("Choose participant type", list(PERSONA_OPTIONS.keys()))

    if "persona" not in st.session_state or st.session_state.persona != persona:
        initialize_session(persona)
        # opening message
        opening = single_llm_decision(client, persona, None)["next_msg"]
        st.session_state.messages.append({"role": "assistant", "content": opening})
        st.session_state.llm_history.append({"role": "assistant", "content": opening})

    with st.sidebar:
        st.header("Topic Coverage")
        prog = len(st.session_state.covered_topics) / len(TOPIC_METADATA)
        st.progress(prog)
        for t, m in TOPIC_METADATA.items():
            st.checkbox(m["label"], value=t in st.session_state.covered_topics, disabled=True)
        st.caption(f"Follow-ups used: {st.session_state.followup_count}/5")
        st.markdown("Finalize to get summary and CSV export.")

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

    if st.session_state.finalized:
        st.info("Interview finalized. Download or copy summary below.")
    else:
        user_text = st.chat_input("Type your response here")
        if user_text:
            st.session_state.messages.append({"role": "user", "content": user_text})
            st.session_state.llm_history.append({"role": "user", "content": user_text})

            payload = single_llm_decision(client, persona, user_text)
            record_topic_notes(user_text, payload.get("topics", []))
            is_vague = payload.get("is_vague", False)

            # follow-up control
            if is_vague and st.session_state.followup_count < 5:
                st.session_state.followup_count += 1
                st.session_state.just_asked_followup = True
            else:
                st.session_state.just_asked_followup = False

            next_msg = payload.get("next_msg", "Could you elaborate?")
            st.session_state.messages.append({"role": "assistant", "content": next_msg})
            st.session_state.llm_history.append({"role": "assistant", "content": next_msg})
            st.rerun()

    st.divider()
    if st.button("Finalize interview and generate summary", disabled=st.session_state.finalized):
        summary = generate_summary(client, persona)
        csv_bytes = build_topic_csv(persona)
        st.session_state.summary = summary
        st.session_state.csv_bytes = csv_bytes
        st.session_state.finalized = True
        st.rerun()

    if st.session_state.finalized and st.session_state.summary:
        st.subheader("Interview Summary")
        st.markdown(st.session_state.summary)
        st.download_button(
            "Download topic notes CSV",
            data=st.session_state.csv_bytes,
            file_name="interview_notes.csv",
            mime="text/csv",
        )


if __name__ == "__main__":
    main()
