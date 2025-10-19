from __future__ import annotations
"""ZEPHYRON Interview Assistant
LLM-guided structured interviews with smarter topic handling:
- Marks topics N/A when respondent rejects relevance
- Adds one intelligent follow-up for technical competence and teamwork if vague
"""

import csv
import io
import json
import os
from typing import Dict, List, Sequence
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
    st.session_state.messages = []
    st.session_state.llm_history = []
    st.session_state.covered_topics = set()
    st.session_state.topic_notes = {t: [] for t in TOPIC_METADATA}
    st.session_state.finalized = False
    st.session_state.summary = None
    st.session_state.csv_bytes = None
    st.session_state.txt_bytes = None
    st.session_state.pending_followup = None  # track if a follow-up is queued

def conversation_history() -> Sequence[Dict[str, str]]:
    return st.session_state.llm_history

def coverage_status() -> str:
    seg = []
    for t, m in TOPIC_METADATA.items():
        seg.append(f"{m['label']}: {'covered' if t in st.session_state.covered_topics else 'not yet'}")
    return "; ".join(seg)

def next_uncovered_topic() -> str | None:
    for t in TOPIC_METADATA:
        if t not in st.session_state.covered_topics:
            return t
    return None

# ---------- LLM UTILITIES ----------
def check_if_vague(client: OpenAI, user_text: str) -> bool:
    """Ask model if response is vague."""
    prompt = (
        "Determine if the response is vague (generic adjectives, no examples, short, or abstract). "
        "Return JSON {\"is_vague\": true/false}."
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
        res = json.loads(r.choices[0].message.content)
        return res.get("is_vague", False)
    except Exception:
        return False

def generate_followup_for_example(client: OpenAI, user_text: str, topic_label: str, persona: str) -> str:
    """Generate a one-line follow-up asking for an example."""
    name = st.session_state.employee_name
    prompt = (
        f"You are interviewing about {name}. The topic is {topic_label}. "
        "The previous answer was vague. Ask ONE brief, natural follow-up question "
        "inviting a specific example or detail. Keep it polite and conversational."
    )
    r = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": prompt},
            {"role": "user", "content": user_text},
        ],
    )
    return r.choices[0].message.content.strip()

# ---------- CORE LLM ----------
def generate_assistant_message(client: OpenAI, persona: str) -> str:
    name = st.session_state.employee_name
    persona_desc = PERSONA_OPTIONS.get(persona, persona.lower())
    if persona == "Self":
        pronoun_instruction = f"You are interviewing {name} about their own performance. Use 'you/your'."
    else:
        pronoun_instruction = (
            f"You are interviewing the {persona_desc} about {name}. "
            f"Always use {name} or 'they/their' when natural, never 'you'."
        )

    system_prompt = (
        pronoun_instruction
        + "\n\n"
        + "Guide the discussion professionally, one question at a time. "
        "Only cover topics that are relevant and not already marked as covered. "
        "If the participant says a topic does not apply (e.g. not a leader), mark it covered as N/A and skip it. "
        "If all topics are covered, wrap up. Avoid repetition or paraphrasing of old questions."
    )

    status = f"Participant type: {persona}. Topic status: {coverage_status()}."
    if not conversation_history():
        status += f" Begin with an open question about {name}'s overall strengths."
    elif next_uncovered_topic():
        lbl = TOPIC_METADATA[next_uncovered_topic()]['label']
        status += f" The next area to discuss is {lbl}."
    else:
        status += f" All topics are covered. Wrap up politely referring to {name}."

    msgs = [{"role": "system", "content": system_prompt}] + list(conversation_history()) + [
        {"role": "system", "content": status}
    ]
    r = client.chat.completions.create(model="gpt-4o-mini", messages=msgs)
    return r.choices[0].message.content.strip()

def analyze_user_response(client: OpenAI, user_text: str) -> Dict[str, List[str]]:
    """Detect which topics are covered, conservative matching."""
    prompt = (
        "Detect which of leadership, technical_competence, team_orientation appear explicitly. "
        "Rules (strict):\n"
        "- leadership: mentions managing, leading, inspiring, delegating, decision-making.\n"
        "- technical_competence: skills, expertise, problem solving, quality of work.\n"
        "- team_orientation: teamwork, helping others, communication, morale.\n"
        "Return JSON {\"topics\":[{\"topic_id\":\"...\",\"notes\":[\"...\"]}]}."
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
    out = {}
    text = user_text.lower()
    leadership_terms = ("lead", "manage", "supervis", "delegat", "inspir", "motiv", "decision")
    for t in data.get("topics", []):
        tid = t.get("topic_id")
        if tid == "leadership" and not any(k in text for k in leadership_terms):
            continue
        if tid in TOPIC_METADATA:
            out[tid] = [n.strip() for n in t.get("notes", []) if n.strip()]
    return out

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

# ---------- EXPORTS ----------
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
                "note_summary": " | ".join(e["notes"]),
                "verbatim_response": e["verbatim"],
            })
    return buf.getvalue().encode()

def build_transcript_txt() -> bytes:
    lines = [f"{m['role'].upper()}: {m['content']}" for m in st.session_state.messages]
    return ("\n".join(lines)).encode("utf-8")

# ---------- MAIN ----------
def main():
    client = require_api_key()
    st.title("ZEPHYRON Interview Assistant")
    st.caption("LLM-guided structured interviews for qualitative evaluations.")

    persona = st.selectbox("Choose participant type", list(PERSONA_OPTIONS.keys()))
    employee_name = st.text_input("Employee name")

    if not employee_name:
        st.info("Please enter the employee's name to begin.")
        st.stop()

    if ("persona" not in st.session_state
        or st.session_state.persona != persona
        or st.session_state.get("employee_name") != employee_name):
        initialize_session(persona, employee_name)
        opening = generate_assistant_message(client, persona)
        st.session_state.messages.append({"role": "assistant", "content": opening})
        st.session_state.llm_history.append({"role": "assistant", "content": opening})

    with st.sidebar:
        st.header("Topic Coverage")
        for t, m in TOPIC_METADATA.items():
            st.checkbox(m["label"], value=t in st.session_state.covered_topics, disabled=True)

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

                # 1) N/A detection (mark topic as covered when explicitly rejected)
                na_signals = {
                    "leadership": (
                        "not in a leadership", "not a leader", "doesn't lead",
                        "doesnt lead", "no leadership", "not his job", "not her job"
                    ),
                    "technical_competence": ("no technical", "not technical", "doesn't code", "doesnt code"),
                    "team_orientation": ("not a team player", "doesn't work with others", "doesnt work with others"),
                }
                for topic_id, signals in na_signals.items():
                    if any(sig in text_lower for sig in signals):
                        st.session_state.covered_topics.add(topic_id)

                # 2) Analyze topics mentioned
                notes = analyze_user_response(client, user_text)

                # 3) Decide on follow-up BEFORE marking covered
                reply = None
                # Prioritize follow-up for technical competence and team orientation
                for topic_id in ["technical_competence", "team_orientation"]:
                    if topic_id in notes and topic_id not in st.session_state.covered_topics:
                        if check_if_vague(client, user_text):
                            # Ask for one concrete example; do NOT mark covered yet
                            reply = generate_followup_for_example(
                                client, user_text, TOPIC_METADATA[topic_id]["label"], persona
                            )
                            # Still record the note, but leave topic open until the follow-up answer arrives
                            st.session_state.topic_notes[topic_id].append(
                                {"verbatim": user_text, "notes": notes[topic_id]}
                            )
                            break  # only one follow-up per turn

                # 4) If no follow-up needed, record notes and mark covered now (conservative)
                if reply is None:
                    for t, vals in notes.items():
                        st.session_state.topic_notes[t].append({"verbatim": user_text, "notes": vals})
                        # Only mark covered if not previously marked N/A and response wasn't vague for depth-required topics
                        if t in ["technical_competence", "team_orientation"]:
                            if not check_if_vague(client, user_text):
                                st.session_state.covered_topics.add(t)
                        else:
                            # leadership gets marked covered only if actually evidenced (or via N/A above)
                            st.session_state.covered_topics.add(t)

                    # 5) Generate normal next question
                    reply = generate_assistant_message(client, persona)

                # 6) Emit assistant message
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
                help="Download plain-text conversation.",
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
        st.markdown(st.session_state.summary)
        st.download_button(
            "Download topic notes CSV",
            st.session_state.csv_bytes,
            file_name="interview_notes.csv",
            mime="text/csv")
        st.download_button(
            "Download transcript (.txt)",
            st.session_state.txt_bytes,
            file_name="interview_transcript.txt",
            mime="text/plain")

if __name__ == "__main__":
    main()
