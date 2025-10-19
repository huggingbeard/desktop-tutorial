from __future__ import annotations

import asyncio
import csv
import io
import json
import os
import random
import re
from typing import Dict, List, Optional, Sequence

import streamlit as st
from openai import OpenAI

st.set_page_config(page_title="Interview Assistant", page_icon="🗣️")

# ---------- CONFIG ----------
PERSONA_OPTIONS = {
    "Boss": "manager",
    "Colleague": "colleague",
    "Self": "self-evaluation",
}

TOPIC_METADATA = {
    "strengths": {"label": "Strengths"},
    "collaboration": {"label": "Collaboration"},
    "growth": {"label": "Growth Areas"},
}

TOPIC_SEQUENCE = ["strengths", "collaboration", "growth"]

OPENING_QUESTION_BANK = [
    "When you look at {name}'s recent work, what strengths jump out first?",
    "What's the most impressive thing {name} has brought to the table lately?",
    "Where have you seen {name} really shine in the last stretch?",
]

TOPIC_QUESTION_BANK = {
    "strengths": [
        "Thinking about the last few weeks, where has {name} made the biggest positive impact?",
        "What do teammates rely on {name} for when things get busy?",
        "How would you describe {name}'s signature strength right now?",
    ],
    "collaboration": [
        "How does {name} shape the tone or flow of the team when you're working together?",
        "What have you noticed about the way {name} supports everyone else?",
        "Can you recall a moment where {name} helped the group stay aligned?",
    ],
    "growth": [
        "Where do you think {name} still has room to grow or get sharper?",
        "If {name} had a bit more time or coaching, what would you focus it on?",
        "What habits could {name} adjust to have even more impact?",
    ],
}

FOLLOWUP_QUESTION_BANK = {
    "strengths": [
        "What's a snapshot or story that captures that strength in action?",
        "When did you last see {name} lean on that strength in a meaningful way?",
    ],
    "collaboration": [
        "Who benefited most from {name}'s approach in that situation?",
        "What did {name} actually do to keep the team moving?",
    ],
    "growth": [
        "What makes that area important for {name} right now?",
        "Have you seen a moment that shows why this is still a stretch for {name}?",
    ],
}

CLOSING_PROMPTS = [
    "Before we wrap, is there anything else about working with {name} that should be on the radar?",
    "Anything we haven't touched on that the review team should keep in mind about {name}?",
]

STOP_REQUESTS = {
    "no we are done",
    "no we're done",
    "we are done",
    "we're done",
    "finish",
    "all done",
    "done",
    "that's all",
    "that is all",
    "stop",
}

SKIP_PATTERNS = {
    "next",
    "move on",
    "skip",
    "nothing else",
    "already told you",
    "already said",
    "as i said",
    "same as before",
    "just said",
}

GENERIC_PHRASES = {
    "not sure",
    "hard to say",
    "same",
    "nothing really",
    "no idea",
    "can't think",
    "i guess",
    "okay",
    "fine",
}

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
    st.session_state.followups_sent: Dict[str, bool] = {t: False for t in TOPIC_METADATA}
    st.session_state.topic_attempts: Dict[str, int] = {t: 0 for t in TOPIC_METADATA}
    st.session_state.last_topic: Optional[str] = None
    st.session_state.awaiting_closing_reply = False
    st.session_state.finalized = False
    st.session_state.summary: Optional[str] = None
    st.session_state.csv_bytes: Optional[bytes] = None
    st.session_state.txt_bytes: Optional[bytes] = None
    st.session_state.pending_inputs: List[str] = []
    st.session_state.processing_stage: str = "idle"
    st.session_state.topic_last_response: Dict[str, str] = {}
    st.session_state.no_more_questions = False


def conversation_history() -> Sequence[Dict[str, str]]:
    return st.session_state.llm_history


# ---------- HELPERS ----------
def append_assistant_message(content: str) -> None:
    message = {"role": "assistant", "content": content}
    st.session_state.messages.append(message)
    st.session_state.llm_history.append(message)


def user_wants_to_stop(user_text: str) -> bool:
    lowered = user_text.strip().lower()
    return lowered in STOP_REQUESTS


def user_requests_next(user_text: str) -> bool:
    lowered = re.sub(r"[.!?]+$", "", user_text).strip().lower()
    return lowered in SKIP_PATTERNS


def response_is_repeat(topic_id: Optional[str], user_text: str) -> bool:
    if not topic_id:
        return False
    previous = st.session_state.topic_last_response.get(topic_id)
    if not previous:
        return False
    normalized = re.sub(r"\s+", " ", user_text.strip().lower())
    prev_normalized = re.sub(r"\s+", " ", previous.strip().lower())
    if normalized == prev_normalized:
        return True
    if normalized in {"same", "same thing"}:
        return True
    return False


def needs_followup(topic_id: Optional[str], user_text: str) -> bool:
    if not topic_id:
        return False
    words = user_text.strip().split()
    if len(words) <= 8:
        return True
    lowered = user_text.strip().lower()
    if any(phrase in lowered for phrase in GENERIC_PHRASES):
        return True
    return False


def pick_topic_question(topic_id: str) -> str:
    templates = TOPIC_QUESTION_BANK.get(topic_id, [])
    name = st.session_state.employee_name
    if not templates:
        return f"What else has stood out about working with {name}?"
    return random.choice(templates).format(name=name)


def pick_followup_question(topic_id: str) -> str:
    templates = FOLLOWUP_QUESTION_BANK.get(topic_id, [])
    name = st.session_state.employee_name
    if not templates:
        return f"What's a moment that brings {name} to mind there?"
    return random.choice(templates).format(name=name)


def pick_opening_question() -> str:
    name = st.session_state.employee_name
    return random.choice(OPENING_QUESTION_BANK).format(name=name)


def pick_closing_question() -> str:
    name = st.session_state.employee_name
    return random.choice(CLOSING_PROMPTS).format(name=name)


# ---------- ANALYSIS ----------
def analyze_user_response(client: OpenAI, user_text: str) -> Dict[str, List[str]]:
    prompt = (
        "Identify whether the response discusses strengths, collaboration, or growth areas for the employee. "
        "Return JSON {\"topics\":[{\"topic_id\":\"strengths|collaboration|growth\",\"notes\":[\"...\"]}]}."
    )
    try:
        r = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": user_text},
            ],
            response_format={"type": "json_object"},
        )
        data = json.loads(r.choices[0].message.content)
    except Exception:
        data = {}
    out: Dict[str, List[str]] = {}
    for topic in data.get("topics", []):
        tid = topic.get("topic_id")
        if tid in TOPIC_METADATA:
            out[tid] = [note.strip() for note in topic.get("notes", []) if note.strip()]
    return out


async def analyze_user_response_async(client: OpenAI, user_text: str) -> Dict[str, List[str]]:
    return await asyncio.to_thread(analyze_user_response, client, user_text)


def record_topic_notes(user_text: str, notes_by_topic: Dict[str, List[str]]) -> None:
    for topic_id, notes in notes_by_topic.items():
        st.session_state.topic_notes.setdefault(topic_id, []).append(
            {"notes": notes, "verbatim": user_text}
        )


# ---------- SUMMARY ----------
def generate_summary(client: OpenAI, persona: str) -> str:
    name = st.session_state.employee_name
    payload = [
        {"topic": TOPIC_METADATA[topic]["label"], "entries": entries}
        for topic, entries in st.session_state.topic_notes.items()
    ]
    r = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "Write concise per-topic interview summaries."},
            {
                "role": "user",
                "content": json.dumps(
                    {
                        "persona": persona,
                        "topic_findings": payload,
                        "instructions": (
                            f"Summarize each topic in 2–3 bullets about {name}. Be factual and concise."
                        ),
                    }
                ),
            },
        ],
        max_tokens=700,
    )
    return r.choices[0].message.content.strip()


# ---------- EXPORT ----------
def build_topic_csv(persona: str) -> bytes:
    buf = io.StringIO()
    writer = csv.DictWriter(
        buf,
        fieldnames=["persona", "topic", "note_index", "note_summary", "verbatim_response"],
    )
    writer.writeheader()
    for topic_id, meta in TOPIC_METADATA.items():
        entries = st.session_state.topic_notes.get(topic_id, [])
        if not entries:
            writer.writerow({"persona": persona, "topic": meta["label"]})
            continue
        for index, entry in enumerate(entries, 1):
            writer.writerow(
                {
                    "persona": persona,
                    "topic": meta["label"],
                    "note_index": index,
                    "note_summary": " | ".join(entry.get("notes", [])),
                    "verbatim_response": entry.get("verbatim", ""),
                }
            )
    return buf.getvalue().encode()


def build_transcript_txt() -> bytes:
    lines = [f"{message['role'].upper()}: {message['content']}" for message in st.session_state.messages]
    return ("\n".join(lines)).encode("utf-8")


# ---------- FLOW ----------
def mark_topic_covered(topic_id: str) -> None:
    st.session_state.covered_topics.add(topic_id)


def next_uncovered_topic() -> Optional[str]:
    for topic in TOPIC_SEQUENCE:
        if topic not in st.session_state.covered_topics:
            return topic
    return None


def handle_user_response(client: OpenAI, persona: str, user_text: str) -> None:
    if st.session_state.finalized or st.session_state.no_more_questions:
        return

    current_topic = st.session_state.last_topic
    if current_topic:
        st.session_state.topic_last_response[current_topic] = user_text

    if user_wants_to_stop(user_text):
        st.session_state.no_more_questions = True
        st.session_state.awaiting_closing_reply = False
        append_assistant_message("Understood. We'll wrap here. Thanks for the insights.")
        return

    notes = asyncio.run(analyze_user_response_async(client, user_text))
    if notes:
        record_topic_notes(user_text, notes)
        for topic_id in notes:
            mark_topic_covered(topic_id)

    if current_topic and current_topic not in notes:
        st.session_state.topic_notes.setdefault(current_topic, []).append(
            {"notes": [user_text.strip()], "verbatim": user_text}
        )

    if current_topic and user_requests_next(user_text):
        mark_topic_covered(current_topic)
        append_assistant_message("No worries, let's shift to something else.")
        st.session_state.last_topic = None
    elif current_topic and response_is_repeat(current_topic, user_text):
        mark_topic_covered(current_topic)
        append_assistant_message("Got it, we'll keep moving.")
        st.session_state.last_topic = None
    elif current_topic and not st.session_state.followups_sent[current_topic] and needs_followup(
        current_topic, user_text
    ):
        followup = pick_followup_question(current_topic)
        append_assistant_message(followup)
        st.session_state.followups_sent[current_topic] = True
        st.session_state.topic_attempts[current_topic] += 1
        return
    else:
        if current_topic:
            mark_topic_covered(current_topic)
            st.session_state.last_topic = None

    next_topic = next_uncovered_topic()
    if next_topic:
        question = pick_topic_question(next_topic)
        append_assistant_message(question)
        st.session_state.last_topic = next_topic
        st.session_state.topic_attempts[next_topic] += 1
        return

    if not st.session_state.awaiting_closing_reply:
        closing = pick_closing_question()
        append_assistant_message(closing)
        st.session_state.awaiting_closing_reply = True
        return

    append_assistant_message("Appreciate you sharing all of that. Thanks again.")
    st.session_state.awaiting_closing_reply = False
    st.session_state.no_more_questions = True


def process_pending_inputs_if_ready(client: OpenAI, persona: str) -> tuple[bool, str]:
    stage = st.session_state.get("processing_stage", "idle")
    if stage == "display":
        st.session_state.processing_stage = "process"
        return False, stage
    if stage == "process" and st.session_state.pending_inputs:
        pending = list(st.session_state.pending_inputs)
        st.session_state.pending_inputs.clear()
        for entry in pending:
            handle_user_response(client, persona, entry)
        st.session_state.processing_stage = "idle"
        return True, stage
    return False, stage


# ---------- MAIN ----------
def main():
    client = require_api_key()
    st.title("ZEPHYRON Interview Assistant")
    st.caption("High-trust conversations about strengths and growth.")

    persona = st.selectbox("Choose participant type", list(PERSONA_OPTIONS.keys()))
    employee_name = st.text_input("Employee name")

    if not employee_name:
        st.info("Enter the employee's name to begin.")
        st.stop()

    if (
        "persona" not in st.session_state
        or st.session_state.persona != persona
        or st.session_state.get("employee_name") != employee_name
    ):
        initialize_session(persona, employee_name)
        opening = pick_opening_question()
        append_assistant_message(opening)
        st.session_state.last_topic = "strengths"
        st.session_state.topic_attempts["strengths"] += 1

    with st.sidebar:
        st.header("Topic Coverage")
        for topic_id, meta in TOPIC_METADATA.items():
            st.checkbox(meta["label"], value=topic_id in st.session_state.covered_topics, disabled=True)

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    st.divider()

    if not st.session_state.finalized:
        user_text = st.chat_input("Type your response here")
        if user_text:
            st.session_state.messages.append({"role": "user", "content": user_text})
            st.session_state.llm_history.append({"role": "user", "content": user_text})
            st.session_state.pending_inputs.append(user_text)
            st.session_state.processing_stage = "display"
            st.rerun()

        if len(st.session_state.messages) > 3:
            txt_bytes = build_transcript_txt()
            st.download_button(
                "Download transcript (.txt)",
                txt_bytes,
                "interview_transcript.txt",
                mime="text/plain",
            )

        st.divider()
        if st.button("Finalize interview and generate summary"):
            try:
                summary = generate_summary(client, persona)
                csv_bytes = build_topic_csv(persona)
                txt_bytes = build_transcript_txt()
                st.session_state.summary = summary
                st.session_state.csv_bytes = csv_bytes
                st.session_state.txt_bytes = txt_bytes
                st.session_state.finalized = True
                st.rerun()
            except Exception as exc:
                st.error(f"Finalization failed: {exc}")
    else:
        st.subheader("Interview Summary")
        st.markdown(st.session_state.summary or "")
        st.download_button(
            "Download topic notes CSV",
            st.session_state.csv_bytes,
            "interview_notes.csv",
            mime="text/csv",
        )
        st.download_button(
            "Download transcript (.txt)",
            st.session_state.txt_bytes,
            "interview_transcript.txt",
            mime="text/plain",
        )

    processed, previous_stage = process_pending_inputs_if_ready(client, persona)
    if processed:
        st.rerun()
    elif previous_stage == "display" and st.session_state.pending_inputs:
        st.rerun()


if __name__ == "__main__":
    main()
