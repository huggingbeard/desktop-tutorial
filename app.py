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
    # Learning state
    st.session_state.last_topic: Optional[str] = None
    st.session_state.topic_depth: Dict[str, int] = {t: 0 for t in TOPIC_METADATA}  # how many Q/answers on that topic

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
    """Fast check if a response lacks concrete examples or detail."""
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
    """True if the topic seems sufficiently explored with specific examples/nuance."""
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

def generate_question(client: OpenAI, persona: str, mode: str, topic_id: Optional[str] = None) -> str:
    """Create one short, natural question. mode ∈ {'opening','topic','closing','followup'}."""
    name = st.session_state.employee_name
    persona_desc = PERSONA_OPTIONS.get(persona, persona.lower())
    tone = (
        "You are running an internal evaluation. Speak briefly, plainly, like a colleague. "
        "Ask one direct question. Avoid corporate phrasing. No filler like 'can you share'."
    )
    base = [
        {"role": "system", "content": tone},
        {"role": "system", "content": f"Participant: {persona_desc}. Person evaluated: {name}."},
    ]
    if mode == "opening":
        user = f"What are {name}'s main strengths at work?"
    elif mode == "topic" and topic_id:
        label = TOPIC_METADATA[topic_id]["label"]
        # Ask a fresh question for this topic
        if topic_id == "leadership":
            user = f"How does {name} handle leadership or influence on the team? Any specific instance?"
        elif topic_id == "technical_competence":
            user = f"How strong are {name}'s technical skills? A concrete example would help."
        else:  # team_orientation
            user = f"What is {name} like to work with day to day? Any example of teamwork or support?"
    elif mode == "followup" and topic_id:
        label = TOPIC_METADATA[topic_id]["label"]
        user = f"Could you give one quick example that shows {name}'s {label.lower()} in action?"
    else:  # closing
        user = f"Anything important about {name} we haven’t covered?"
    r = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=base + [{"role": "user", "content": user}],
        max_tokens=120,
    )
    return r.choices[0].message.content.strip()

# ---------- ANALYSIS ----------
def analyze_user_response(client: OpenAI, user_text: str) -> Dict[str, List[str]]:
    """Detect which topics were explicitly discussed (strict)."""
    prompt = (
        "Detect which of leadership, technical_competence, team_orientation appear explicitly. "
        "Rules (strict):\n"
        "- leadership: managing, leading, inspiring, delegating, supervising, decision-making.\n"
        "- technical_competence: skills, expertise, problem solving, quality of work.\n"
        "- team_orientation: teamwork, helping others, collaboration, communication, morale.\n"
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
    out: Dict[str, List[str]] = {}
    text = user_text.lower()
    leadership_terms = ("lead", "manage", "supervis", "delegat", "inspir", "motiv", "decision")
    for t in data.get("topics", []):
        tid = t.get("topic_id")
        if tid == "leadership" and not any(k in text for k in leadership_terms):
            continue  # gate leadership
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
    """Choose next topic to ask about."""
    # Prefer the first still-open topic in TOPIC_ORDER
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
        # Opening
        opening_q = generate_question(client, persona, mode="opening")
        st.session_state.messages.append({"role": "assistant", "content": opening_q})
        st.session_state.llm_history.append({"role": "assistant", "content": opening_q})

    # Sidebar coverage
    with st.sidebar:
        st.header("Topic Coverage")
        for t, m in TOPIC_METADATA.items():
            st.checkbox(m["label"], value=t in st.session_state.covered_topics, disabled=True)
        st.caption(coverage_status())

    # Show chat
    for m in st.session_state.messages:
        with st.chat_message(m["role"]):
            st.write(m["content"])

    st.divider()

    if not st.session_state.finalized:
        user_text = st.chat_input("Type your response here")
        if user_text:
            st.session_state.messages.append({"role": "user", "content": user_text})
            st.session_state.llm_history.append({"role": "user", "content": user_text})

            try:
                text_lower = user_text.lower()

                # --- N/A detection: mark topic as covered when explicitly rejected ---
                na_signals = {
                    "leadership": (
                        "not in a leadership", "not a leader", "doesn't lead", "doesnt lead",
                        "no leadership", "not his job", "not her job", "not their job"
                    ),
                    "technical_competence": ("no technical", "not technical", "doesn't code", "doesnt code"),
                    "team_orientation": ("not a team player", "doesn't work with others", "doesnt work with others"),
                }
                for topic_id, signals in na_signals.items():
                    if topic_id not in st.session_state.covered_topics and any(sig in text_lower for sig in signals):
                        mark_topic_covered(topic_id)

                # --- Analyze user response and record notes (conservative leadership gate) ---
                notes = analyze_user_response(client, user_text)
                for t, vals in notes.items():
                    st.session_state.topic_notes[t].append({"verbatim": user_text, "notes": vals})

                # --- Decide active topic ---
                active = st.session_state.last_topic
                if (not active) or (active in st.session_state.covered_topics):
                    # Prefer detected topic in notes; otherwise next open by order
                    for candidate in TOPIC_ORDER:
                        if candidate in notes and candidate not in st.session_state.covered_topics:
                            active = candidate
                            break
                    if not active:
                        active = pick_initial_or_next_topic()

                reply = None

                # If we have an active topic and it's not covered, apply depth logic
                if active and active not in st.session_state.covered_topics:
                    # If user just declared N/A for this topic, close it
                    if active in st.session_state.covered_topics:
                        st.session_state.last_topic = None
                    else:
                        # Update depth counter (we count only when user text addressed/was prompted on this topic)
                        st.session_state.topic_depth[active] = st.session_state.topic_depth.get(active, 0) + 1
                        depth = st.session_state.topic_depth[active]

                        # Check detail sufficiency
                        label = TOPIC_METADATA[active]["label"]
                        # Consider last 2 user turns (if available)
                        last_user_texts = []
                        for item in reversed(st.session_state.llm_history):
                            if item["role"] == "user":
                                last_user_texts.append(item["content"])
                                if len(last_user_texts) >= 2:
                                    break
                        combined = "\n".join(reversed(last_user_texts)) or user_text

                        enough = detailed_enough(client, label, combined)
                        vague = check_if_vague(client, user_text)

                        # Policy:
                        # - Default done at depth >= 2 if enough==True
                        # - If not enough, allow up to depth == 3 with one concise follow-up
                        # - For technical/team, prefer one example follow-up when vague
                        if depth >= 2 and enough:
                            mark_topic_covered(active)
                        elif depth < 3 and (not enough):
                            # Ask one follow-up tailored to topic
                            if active in ("technical_competence", "team_orientation") and vague:
                                reply = generate_question(client, persona, mode="followup", topic_id=active)
                            else:
                                # Generic short follow-up if still thin
                                reply = generate_question(client, persona, mode="followup", topic_id=active)
                        else:
                            # Depth limit hit → move on anyway
                            mark_topic_covered(active)

                        st.session_state.last_topic = None if active in st.session_state.covered_topics else active

                # If we didn't produce a follow-up, move to next topic or close
                if reply is None:
                    nxt = next_uncovered_topic()
                    if nxt:
                        # Avoid repeating the same topic question; ask a fresh one
                        st.session_state.last_topic = nxt
                        reply = generate_question(client, persona, mode="topic", topic_id=nxt)
                    else:
                        reply = generate_question(client, persona, mode="closing")
                        st.session_state.last_topic = None

                # Emit assistant message
                st.session_state.messages.append({"role": "assistant", "content": reply})
                st.session_state.llm_history.append({"role": "assistant", "content": reply})

            except Exception as e:
                st.error(str(e))

            st.rerun()

        # transcript download during interview
        if len(st.session_state.messages) > 3:
            txt_bytes = build_transcript_txt()
            st.download_button(
                "Download transcript (.txt)",
                data=txt_bytes,
                file_name="interview_transcript.txt",
                mime="text/plain",
                help="Download the conversation so far.",
            )

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
        st.download_button("Download topic notes CSV",
                           st.session_state.csv_bytes,
                           file_name="interview_notes.csv",
                           mime="text/csv")
        st.download_button("Download transcript (.txt)",
                           st.session_state.txt_bytes,
                           file_name="interview_transcript.txt",
                           mime="text/plain")

if __name__ == "__main__":
    main()
