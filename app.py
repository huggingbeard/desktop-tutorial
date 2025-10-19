from __future__ import annotations
"""Interactive Streamlit app for an LLM-guided interview assistant.
fixes: stops repeating topics, respects 'we covered that', adds transcript & name"""

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
TOPIC_METADATA = {
    "leadership": {"label": "Leadership"},
    "technical_competence": {"label": "Technical Competence"},
    "team_orientation": {"label": "Team Orientation"},
}

# ---------- API KEY ----------
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

# ---------- CORE LLM ----------
def generate_assistant_message(client: OpenAI, persona: str) -> str:
    name = st.session_state.employee_name
    persona_desc = PERSONA_OPTIONS.get(persona, persona.lower())

    if persona == "Self":
        pronoun_instruction = (
            f"You are interviewing {name} about their own performance. "
            "Use second-person pronouns (you/your)."
        )
    else:
        pronoun_instruction = (
            f"You are interviewing the {persona_desc} about {name}. "
            f"{name} is the employee being discussed. "
            "Always use their name or 'they/their' when natural, never 'you'."
        )

    system_prompt = (
        pronoun_instruction
        + "\n\n"
        + "Guide the discussion with warm, professional language, asking one question at a time. "
        "Cover leadership, technical competence, and team orientation, but never obsess over completeness. "
        "If a topic was clearly discussed or the participant says it's already covered, move on. "
        "No GPT speak - no delve etc. Avoid repeating or rephrasing previous questions. "
        "Once all topics are discussed, wrap up politely."
    )

    status = f"Participant type: {persona}. Topic status: {coverage_status()}."
    if not conversation_history():
        status += f" Begin with an open question about {name}'s overall strengths."
    elif next_uncovered_topic():
        lbl = TOPIC_METADATA[next_uncovered_topic()]['label']
        status += f" The next area to discuss is {lbl}."
    else:
        status += f" All topics are covered. Wrap up politely, referring to {name} by name."

    msgs = [{"role": "system", "content": system_prompt}] + list(conversation_history()) + [
        {"role": "system", "content": status}
    ]
    r = client.chat.completions.create(model="gpt-4o-mini", messages=msgs)
    return r.choices[0].message.content.strip()

def analyze_user_response(client: OpenAI, user_text: str) -> Dict[str, List[str]]:
    prompt = (
        "Detect which topics are explicitly discussed (leadership, technical_competence, team_orientation).\n"
        "Rules (strict):\n"
        "- leadership: only if managing, leading, inspiring, motivating, delegating, supervising, decision-making are mentioned.\n"
        "- technical_competence: skills, expertise, problem solving, analysis, output quality, domain knowledge.\n"
        "- team_orientation: teamwork, helping others, collaboration, communication, morale.\n"
        "Do NOT infer leadership from being lazy, late, or similar. Be conservative.\n"
        "Return JSON: {\"topics\":[{\"topic_id\":\"...\",\"notes\":[\"...\"]}]}."
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

def build_topic_csv(persona: str) -> bytes:
    buf = io.StringIO()
    w = csv.DictWriter(buf,
        fieldnames=["persona", "topic", "note_index", "note_summary", "verbatim_response"])
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
                # detect closure signals
                closure_signals = ("we covered", "already said", "nothing more", "no more", "done")
                if any(sig in user_text.lower() for sig in closure_signals):
                    if next_uncovered_topic():
                        st.session_state.covered_topics.add(next_uncovered_topic())

                notes = analyze_user_response(client, user_text)
                # mark topics covered BEFORE generating next message
                for t, vals in notes.items():
                    st.session_state.covered_topics.add(t)
                    st.session_state.topic_notes[t].append({"verbatim": user_text, "notes": vals})

                reply = generate_assistant_message(client, persona)
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
                help="Download a plain-text copy of the full conversation so far.",
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
