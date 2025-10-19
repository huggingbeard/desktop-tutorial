from __future__ import annotations

import asyncio
import csv
import io
import json
import os
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
    "strengths": {"label": "Strengths", "intro": "First, I'd like to discuss {name}'s strengths."},
    "collaboration": {"label": "Collaboration", "intro": "Now let's talk about how {name} collaborates with the team."},
    "growth": {"label": "Growth Areas", "intro": "Finally, let's discuss areas where {name} could grow or develop further."},
}

TOPIC_SEQUENCE = ["strengths", "collaboration", "growth"]

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
    st.session_state.current_topic_index = 0
    st.session_state.topic_notes: Dict[str, List[Dict[str, str]]] = {t: [] for t in TOPIC_METADATA}
    st.session_state.responses_this_topic = 0
    st.session_state.finalized = False
    st.session_state.summary: Optional[str] = None
    st.session_state.csv_bytes: Optional[bytes] = None
    st.session_state.txt_bytes: Optional[bytes] = None
    st.session_state.pending_inputs: List[str] = []
    st.session_state.processing_stage: str = "idle"
    st.session_state.interview_complete = False


# ---------- HELPERS ----------
def append_assistant_message(content: str) -> None:
    message = {"role": "assistant", "content": content}
    st.session_state.messages.append(message)
    st.session_state.llm_history.append(message)


def current_topic_id() -> Optional[str]:
    idx = st.session_state.current_topic_index
    if idx < len(TOPIC_SEQUENCE):
        return TOPIC_SEQUENCE[idx]
    return None


def get_interview_prompt() -> str:
    name = st.session_state.employee_name
    persona = st.session_state.persona
    current_topic = current_topic_id()
    
    if not current_topic:
        return f"""You are conducting a performance review interview. The participant is a {persona} giving feedback about {name}.

You've covered all main topics. Ask if there's anything else important to share about {name}, then wrap up warmly."""

    topic_label = TOPIC_METADATA[current_topic]["label"]
    responses_so_far = st.session_state.responses_this_topic
    
    base_prompt = f"""You are conducting a performance review interview. The participant is a {persona} giving feedback about {name}.

Current topic: {topic_label}
Responses received on this topic: {responses_so_far}

INSTRUCTIONS:
- Be conversational, warm, and professional
- If the user's response is vague or very brief (under 10 words), ask ONE follow-up question to get a concrete example
- If the user's response is detailed and specific, acknowledge it warmly and decide:
  * If you have good insights on this topic (after 1-2 exchanges), move to the next topic
  * If their response warrants one more clarifying question, ask it
- When moving to a new topic, clearly signal the transition
- Reference specific things they said to show you're listening
- Never ask more than 1 follow-up question per user response

If the user says "next", "move on", "skip", or similar - respect that and transition immediately."""

    return base_prompt


def generate_llm_response(client: OpenAI, user_text: str) -> tuple[str, bool, str]:
    """
    Returns: (assistant_response, should_move_to_next_topic, topic_notes)
    """
    system_prompt = get_interview_prompt()
    
    # Build conversation history
    history = list(st.session_state.llm_history)
    history.append({"role": "user", "content": user_text})
    
    # Add decision-making instruction
    current_topic = current_topic_id()
    if current_topic:
        decision_prompt = f"""After responding, decide if you have enough information about {TOPIC_METADATA[current_topic]['label']} to move on.

Respond with JSON:
{{
    "response": "your conversational response here",
    "move_to_next_topic": true/false,
    "key_insights": "brief summary of what you learned from this response"
}}

Set move_to_next_topic to true if:
- You have 1-2 good specific examples or insights on this topic
- The user explicitly asks to move on
- You've had 3+ exchanges on this topic

Set it to false if:
- This is the first response and it's vague/brief
- You need one clarifying example"""
    else:
        decision_prompt = """The interview is wrapping up. Respond warmly and thank them.

Respond with JSON:
{
    "response": "your conversational response",
    "move_to_next_topic": false,
    "key_insights": "any final thoughts"
}"""
    
    try:
        r = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                *history[:-1],  # All history except last user message
                {"role": "user", "content": f"{user_text}\n\n{decision_prompt}"}
            ],
            response_format={"type": "json_object"},
            temperature=0.7,
        )
        
        result = json.loads(r.choices[0].message.content)
        response = result.get("response", "Thanks for sharing that.")
        should_move = result.get("move_to_next_topic", False)
        insights = result.get("key_insights", user_text)
        
        return response, should_move, insights
        
    except Exception as e:
        # Fallback
        return "Thanks for sharing that. What else can you tell me?", False, user_text


def start_new_topic(client: OpenAI) -> Optional[str]:
    """Start the next topic with a clear introduction"""
    idx = st.session_state.current_topic_index
    if idx >= len(TOPIC_SEQUENCE):
        return None
    
    topic_id = TOPIC_SEQUENCE[idx]
    name = st.session_state.employee_name
    intro_template = TOPIC_METADATA[topic_id]["intro"]
    intro = intro_template.format(name=name)
    
    # Ask LLM to generate a natural opening question for this topic
    prompt = f"""{intro} Generate a natural, specific opening question about {name}'s {TOPIC_METADATA[topic_id]['label'].lower()}.

Make it conversational and specific - ask about recent examples or observable behaviors."""
    
    try:
        r = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=100,
            temperature=0.7,
        )
        question = r.choices[0].message.content.strip()
        return f"{intro} {question}"
    except Exception:
        return f"{intro} What have you observed about {name} in this area?"


# ---------- EXPORT ----------
def build_topic_csv(persona: str) -> bytes:
    buf = io.StringIO()
    writer = csv.DictWriter(
        buf,
        fieldnames=["persona", "topic", "note_index", "key_insights", "verbatim_response"],
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
                    "key_insights": entry.get("insights", ""),
                    "verbatim_response": entry.get("verbatim", ""),
                }
            )
    return buf.getvalue().encode()


def build_transcript_txt() -> bytes:
    lines = [f"{message['role'].upper()}: {message['content']}" for message in st.session_state.messages]
    return ("\n".join(lines)).encode("utf-8")


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


# ---------- FLOW ----------
def handle_user_response(client: OpenAI, persona: str, user_text: str) -> None:
    if st.session_state.finalized or st.session_state.interview_complete:
        return
    
    current_topic = current_topic_id()
    
    # Generate LLM response and get decision
    assistant_response, should_move, insights = generate_llm_response(client, user_text)
    
    # Record notes
    if current_topic:
        st.session_state.topic_notes[current_topic].append({
            "insights": insights,
            "verbatim": user_text
        })
        st.session_state.responses_this_topic += 1
    
    # Send assistant response
    append_assistant_message(assistant_response)
    
    # Handle topic transition
    if should_move:
        st.session_state.current_topic_index += 1
        st.session_state.responses_this_topic = 0
        
        # Start next topic or wrap up
        new_topic_intro = start_new_topic(client)
        if new_topic_intro:
            append_assistant_message(new_topic_intro)
        else:
            # Interview complete
            st.session_state.interview_complete = True


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
        # Start first topic
        opening = start_new_topic(client)
        if opening:
            append_assistant_message(opening)

    with st.sidebar:
        st.header("Topic Coverage")
        for idx, topic_id in enumerate(TOPIC_SEQUENCE):
            meta = TOPIC_METADATA[topic_id]
            is_covered = idx < st.session_state.current_topic_index
            is_current = idx == st.session_state.current_topic_index
            label = f"{'✓ ' if is_covered else '▶ ' if is_current else ''}{meta['label']}"
            st.checkbox(label, value=is_covered, disabled=True)

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
