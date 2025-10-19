from __future__ import annotations

# ZEPHYRON Interview Assistant
# Natural tone + learning flow:
# - Tracks topic depth (default 2, up to 3 if not detailed with examples)
# - Marks topics N/A on explicit rejection (e.g., "not a leader")
# - One concise follow-up for technical/team if vague
# - Avoids repeating earlier questions; moves on once a topic is done

import asyncio
import csv
import io
import json
import os
import random
import re
from typing import Dict, List, Sequence, Optional, Set

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

OPENING_QUESTION_BANK = [
    "When you think about working with {name} lately, what's been standing out?",
    "What's been the most memorable part of working with {name} recently?",
    "What have you noticed most about teaming up with {name} these days?",
]

TOPIC_QUESTION_BANK = {
    "leadership": [
        "{mirror}Where have you seen {name} take the lead or steady the group lately?",
        "{mirror}How does {name} rally people when the team needs direction?",
        "{mirror}What's a moment when {name} stepped in to steer the team?",
    ],
    "technical_competence": [
        "{mirror}Walk me through a recent moment where {name}'s technical judgment made the difference.",
        "{mirror}Tell me about a project that shows {name}'s technical chops in action.",
        "{mirror}When did {name}'s build-or-fix skills really save the day lately?",
    ],
    "team_orientation": [
        "{mirror}What does working with {name} feel like when the team is under pressure?",
        "{mirror}Who benefits most from {name}'s way of showing up for the team?",
        "{mirror}What's a recent moment that captures how {name} supports everyone else?",
    ],
}

FOLLOWUP_TOPICS = {"technical_competence", "team_orientation"}

STOPWORDS = {
    "the",
    "and",
    "for",
    "that",
    "with",
    "have",
    "this",
    "from",
    "they",
    "their",
    "about",
    "been",
    "your",
    "when",
    "what",
    "where",
    "which",
    "because",
    "while",
    "there",
    "really",
    "just",
    "like",
    "maybe",
    "very",
    "were",
    "also",
    "into",
    "those",
    "these",
    "through",
    "still",
    "even",
    "could",
    "should",
    "would",
    "does",
    "don't",
    "didn't",
    "can't",
    "isn't",
    "wasn't",
    "aren't",
    "won't",
    "hasn't",
}


SKIP_PATTERNS = {
    "next",
    "move on",
    "skip",
    "nothing else",
    "that's it",
    "already told you",
    "as i said",
    "same as before",
    "just said",
    "already said",
    "i already told you",
    "i already said",
    "told you already",
}

MAX_TOPIC_ATTEMPTS = 3


def extract_mirror_fragment() -> Optional[str]:
    if "llm_history" not in st.session_state:
        return None
    for entry in reversed(st.session_state.llm_history):
        if entry.get("role") != "user":
            continue
        text = entry.get("content", "").strip()
        if not text:
            continue
        sentences = [s.strip() for s in re.split(r"[.!?]+", text) if s.strip()]
        for sentence in reversed(sentences):
            cleaned = sentence.strip()
            if cleaned:
                snippet = cleaned[:80]
                return snippet
        words = re.findall(r"[A-Za-z0-9'-]+", text)
        keywords = [w for w in words if len(w) > 3 and w.lower() not in STOPWORDS]
        if keywords:
            return " ".join(keywords[:3])[:80]
    return None


def build_mirror_prefix() -> str:
    fragment = extract_mirror_fragment()
    if not fragment:
        return ""
    return "You touched on that earlier. "


def record_topic_notes(user_text: str, notes_by_topic: Dict[str, List[str]]) -> None:
    for topic_id, notes in notes_by_topic.items():
        st.session_state.topic_notes.setdefault(topic_id, []).append(
            {"notes": notes, "verbatim": user_text}
        )
        update_topic_brief(topic_id)


TOPIC_REJECTION_PATTERNS = {
    "leadership": [r"\bnot a leader\b", r"\bno leadership\b"],
    "technical_competence": [r"\bnot technical\b", r"\bno technical skills\b"],
    "team_orientation": [r"\bnot (?:a )?team player\b", r"\bno teamwork\b"],
}


def detect_topic_rejection(user_text: str) -> Optional[str]:
    lowered = user_text.lower()
    for topic_id, patterns in TOPIC_REJECTION_PATTERNS.items():
        for pattern in patterns:
            if re.search(pattern, lowered):
                return topic_id
    return None


def user_wants_to_stop(user_text: str) -> bool:
    lowered = user_text.strip().lower()
    return lowered in {
        "no we are done",
        "no we're done",
        "we're done",
        "we are done",
        "finish",
        "all done",
        "done",
        "that's all",
        "that is all",
    }


def user_requests_next(user_text: str) -> bool:
    lowered = re.sub(r"[.!?]+$", "", user_text).strip().lower()
    return lowered in SKIP_PATTERNS


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


def is_repeat_response(topic_id: Optional[str], user_text: str) -> bool:
    if not topic_id:
        return False
    previous = st.session_state.topic_last_response.get(topic_id)
    if not previous:
        return False
    current = normalize_text(user_text)
    prev_norm = normalize_text(previous)
    if current == prev_norm:
        return True
    if current in {"same", "same thing", "already told you", "as i said"}:
        return True
    if len(current) <= 40 and current in prev_norm:
        return True
    return False


def update_topic_brief(topic_id: str) -> None:
    entries = st.session_state.topic_notes.get(topic_id, [])
    snapshots: Set[str] = st.session_state.topic_briefs.setdefault(topic_id, set())
    if not entries:
        return
    latest = entries[-1]
    for note in latest.get("notes", []):
        cleaned = note.strip()
        if cleaned:
            snapshots.add(cleaned)


def build_progress_brief() -> str:
    segments: List[str] = []
    for topic_id in TOPIC_ORDER:
        label = TOPIC_METADATA[topic_id]["label"]
        if topic_id in st.session_state.topic_briefs and st.session_state.topic_briefs[topic_id]:
            snippets = list(st.session_state.topic_briefs[topic_id])
            snippets.sort()
            summary = "; ".join(snippets[:3])
            segments.append(f"{label}: {summary}")
        elif topic_id in st.session_state.covered_topics:
            segments.append(f"{label}: covered")
        else:
            segments.append(f"{label}: pending")
    return " | ".join(segments)


def append_assistant_message(content: str) -> None:
    message = {"role": "assistant", "content": content}
    st.session_state.messages.append(message)
    st.session_state.llm_history.append(message)

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
    st.session_state.topic_briefs: Dict[str, Set[str]] = {t: set() for t in TOPIC_METADATA}
    st.session_state.finalized = False
    st.session_state.summary: Optional[str] = None
    st.session_state.csv_bytes: Optional[bytes] = None
    st.session_state.txt_bytes: Optional[bytes] = None
    st.session_state.last_topic: Optional[str] = None
    st.session_state.last_mode: Optional[str] = None
    st.session_state.topic_depth: Dict[str, int] = {t: 0 for t in TOPIC_METADATA}
    st.session_state.rejected_topics: set[str] = set()
    st.session_state.pending_ack_topic: Optional[str] = None
    st.session_state.followups_sent: set[str] = set()
    st.session_state.no_more_questions = False
    st.session_state.closing_declined = False
    st.session_state.topic_last_response: Dict[str, str] = {}
    st.session_state.pending_inputs: List[str] = []
    st.session_state.processing_stage: str = "idle"

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


async def check_if_vague_async(client: OpenAI, user_text: str) -> bool:
    return await asyncio.to_thread(check_if_vague, client, user_text)


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


async def detailed_enough_async(client: OpenAI, topic_label: str, last_user_texts: str) -> bool:
    return await asyncio.to_thread(detailed_enough, client, topic_label, last_user_texts)

# ---------- FIXED FUNCTION ----------
def generate_question(client: OpenAI, persona: str, mode: str, topic_id: Optional[str] = None) -> str:
    """Always generate a QUESTION, never a statement."""
    name = st.session_state.employee_name
    mirror_prefix = build_mirror_prefix()
    attempts = 0
    if topic_id:
        attempts = st.session_state.topic_depth.get(topic_id, 0)
    if mode == "opening":
        template = random.choice(OPENING_QUESTION_BANK)
        return template.format(name=name)
    if mode == "topic" and topic_id:
        templates = TOPIC_QUESTION_BANK.get(topic_id, [])
        if templates:
            template = random.choice(templates)
            q = template.format(name=name, mirror=mirror_prefix)
        else:
            q = f"{mirror_prefix}What stands out about working with {name}?"
    elif mode == "followup" and topic_id:
        label = TOPIC_METADATA[topic_id]['label']
        q = f"{mirror_prefix}What's one moment that really shows {name}'s {label.lower()} in action?"
    else:
        q = f"{mirror_prefix}Before we wrap, is there anything about working with {name} that you'd want the review team to know?"
    if attempts >= MAX_TOPIC_ATTEMPTS - 1 and mode == "topic" and topic_id:
        q = f"{mirror_prefix}Is there anything else worth noting about {name}'s {TOPIC_METADATA[topic_id]['label'].lower()}, or should we move on?"
    return q.strip()


def gather_topic_judgments(
    client: OpenAI,
    topic_label: str,
    user_text: str,
    need_vague_check: bool,
    need_detail_check: bool,
) -> Dict[str, bool]:
    if not (need_vague_check or need_detail_check):
        return {}

    async def _runner() -> Dict[str, bool]:
        tasks = {}
        if need_vague_check:
            tasks["vague"] = asyncio.create_task(check_if_vague_async(client, user_text))
        if need_detail_check:
            tasks["detail"] = asyncio.create_task(detailed_enough_async(client, topic_label, user_text))
        results: Dict[str, bool] = {}
        for key, task in tasks.items():
            results[key] = await task
        return results

    return asyncio.run(_runner())


def handle_user_response(client: OpenAI, persona: str, user_text: str) -> None:
    if st.session_state.finalized:
        return

    rejection_topic = detect_topic_rejection(user_text)
    if rejection_topic and rejection_topic not in st.session_state.covered_topics:
        st.session_state.rejected_topics.add(rejection_topic)
        st.session_state.covered_topics.add(rejection_topic)
        st.session_state.topic_notes[rejection_topic].append({
            "notes": ["Participant declined to discuss this topic."],
            "verbatim": user_text,
        })
        st.session_state.pending_ack_topic = rejection_topic
        update_topic_brief(rejection_topic)

    notes = asyncio.run(analyze_user_response_async(client, user_text))
    if notes:
        record_topic_notes(user_text, notes)

    last_topic = st.session_state.last_topic

    for topic_id in notes:
        if topic_id != last_topic:
            mark_topic_covered(topic_id)
    send_followup = False
    move_on = False
    requested_next = user_requests_next(user_text)
    repeated = is_repeat_response(last_topic, user_text)

    if last_topic:
        st.session_state.topic_last_response[last_topic] = user_text
        attempts = st.session_state.topic_depth.get(last_topic, 0)
        if attempts >= MAX_TOPIC_ATTEMPTS:
            move_on = True
        if requested_next:
            move_on = True
        if repeated:
            move_on = True
        topic_label = TOPIC_METADATA[last_topic]['label']
        if not move_on:
            need_vague = last_topic in FOLLOWUP_TOPICS and last_topic not in st.session_state.followups_sent
            judgments = gather_topic_judgments(
                client,
                topic_label,
                user_text,
                need_vague_check=need_vague,
                need_detail_check=True,
            )
            if need_vague and judgments.get("vague"):
                send_followup = True
                st.session_state.followups_sent.add(last_topic)
            elif judgments.get("detail"):
                move_on = True
        if move_on:
            mark_topic_covered(last_topic)

    if user_wants_to_stop(user_text):
        st.session_state.no_more_questions = True
        st.session_state.closing_declined = True
        append_assistant_message("Understood. We'll stop here.")
        return

    if st.session_state.last_mode == "closing":
        st.session_state.no_more_questions = True
        append_assistant_message("Appreciate it — that's everything I needed. Thanks for walking me through this.")
        st.session_state.last_mode = None
        
    if st.session_state.no_more_questions:
        return

    if st.session_state.pending_ack_topic:
        topic_id = st.session_state.pending_ack_topic
        label = TOPIC_METADATA[topic_id]["label"]
        append_assistant_message(f"Got it, we'll skip {label.lower()}.")
        st.session_state.pending_ack_topic = None

    if requested_next and last_topic:
        append_assistant_message("Sure, let's jump to something else.")

    if repeated and last_topic and not requested_next:
        append_assistant_message("Right, we've already logged that. I'll switch topics.")

    if send_followup and last_topic:
        question = generate_question(client, persona, mode="followup", topic_id=last_topic)
        append_assistant_message(question)
        st.session_state.last_mode = "followup"
        return

    if move_on:
        st.session_state.last_topic = None
        st.session_state.last_mode = None

    next_topic = pick_initial_or_next_topic()
    while next_topic and st.session_state.topic_depth.get(next_topic, 0) >= MAX_TOPIC_ATTEMPTS:
        mark_topic_covered(next_topic)
        next_topic = pick_initial_or_next_topic()

    if next_topic:
        question = generate_question(client, persona, mode="topic", topic_id=next_topic)
        append_assistant_message(question)
        st.session_state.last_topic = next_topic
        st.session_state.last_mode = "topic"
        st.session_state.topic_depth[next_topic] += 1
        return

    if st.session_state.closing_declined:
        st.session_state.no_more_questions = True
        return

    question = generate_question(client, persona, mode="closing")
    append_assistant_message(question)
    st.session_state.last_mode = "closing"
    st.session_state.last_topic = None

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


async def analyze_user_response_async(client: OpenAI, user_text: str) -> Dict[str, List[str]]:
    return await asyncio.to_thread(analyze_user_response, client, user_text)

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


def process_pending_inputs_if_ready(client: OpenAI, persona: str) -> tuple[bool, str]:
    stage = st.session_state.get("processing_stage", "idle")
    if stage == "display":
        st.session_state.processing_stage = "process"
        return False, stage
    if stage == "process" and st.session_state.pending_inputs:
        queue = list(st.session_state.pending_inputs)
        st.session_state.pending_inputs.clear()
        for pending_text in queue:
            handle_user_response(client, persona, pending_text)
        st.session_state.processing_stage = "idle"
        return True, stage
    return False, stage

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
        append_assistant_message(opening_q)
        st.session_state.last_mode = "opening"
        st.session_state.last_topic = None

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
            st.session_state.pending_inputs.append(user_text)
            st.session_state.processing_stage = "display"
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

    processed, previous_stage = process_pending_inputs_if_ready(client, persona)
    if processed:
        st.rerun()
    elif previous_stage == "display" and st.session_state.pending_inputs:
        st.rerun()

if __name__ == "__main__":
    main()
