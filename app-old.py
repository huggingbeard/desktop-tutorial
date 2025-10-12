"""Interactive Streamlit app for an LLM-guided interview assistant."""
## this is after going back to the mid-day code. feels ok, but not very specific with little follow-up
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


@st.cache_resource(show_spinner=False)
def get_openai_client(api_key: str) -> OpenAI:
    """Create an OpenAI client for reuse across reruns."""
    return OpenAI(api_key=api_key)


def require_api_key() -> OpenAI:
    """Return an OpenAI client or stop the app if the API key is missing."""
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
            "OPENAI_API_KEY is not set. Provide it via Streamlit secrets or the environment "
            "before running the interview assistant."
        )
        st.stop()
    return get_openai_client(api_key)


def initialize_session(persona: str) -> None:
    """Prepare session state when the persona changes."""
    st.session_state.persona = persona
    st.session_state.messages = []
    st.session_state.llm_history = []
    st.session_state.covered_topics = set()
    st.session_state.topic_notes = {topic: [] for topic in TOPIC_METADATA}
    st.session_state.finalized = False
    st.session_state.summary = None
    st.session_state.csv_bytes = None


def conversation_history() -> Sequence[Dict[str, str]]:
    """Return the stored chat history for the LLM."""
    return st.session_state.llm_history


def persona_description(persona: str) -> str:
    """Describe the participant persona for prompting."""
    descriptor = PERSONA_OPTIONS.get(persona, persona.lower())
    return f"You are speaking with a {descriptor} of the employee."


def coverage_status() -> str:
    """Provide a human-readable summary of topic coverage."""
    segments = []
    for topic_id, metadata in TOPIC_METADATA.items():
        status = "covered" if topic_id in st.session_state.covered_topics else "not yet covered"
        segments.append(f"{metadata['label']}: {status}")
    return "; ".join(segments)


def next_uncovered_topic() -> str | None:
    for topic_id in TOPIC_METADATA:
        if topic_id not in st.session_state.covered_topics:
            return topic_id
    return None


def generate_assistant_message(client: OpenAI, persona: str) -> str:
    """Use the OpenAI model to craft the next interviewer message."""
    system_prompt = (
        "You are an interviewing assistant facilitating a structured qualitative conversation. "
        "Guide the discussion with warm, professional language, asking one question at a time. "
        "Encourage rich reflections while ensuring the interview ultimately covers leadership, "
        "technical competence, and team orientation. "
        "If a topic is marked as covered, do NOT ask about it again. "
        "When asking for examples, reference what they already said. "
        "If every topic is already covered, invite final reflections and let the participant know "
        "you will summarize what you've heard."
    )

    status_message = (
        f"Participant type: {persona}. {persona_description(persona)} "
        f"Topic status: {coverage_status()}."
    )
    if not conversation_history():
        status_message += (
            " This is the opening of the interview. Begin with an open-ended question that invites "
            "the participant to reflect on the employee's overall strengths."
        )
    elif next_uncovered_topic() is not None:
        topic_label = TOPIC_METADATA[next_uncovered_topic()]["label"]
        status_message += (
            f" {topic_label} has not been thoroughly discussed yet. "
            "If the last response didn't cover it with examples, ask naturally about it. "
            "If the response was vague, ask for a specific example referencing what they said. "
            "If they already gave details, move to the next uncovered topic."
        )
    else:
        status_message += (
            " All topics have been discussed. Offer a closing prompt for any additional thoughts or "
            "wrap up warmly."
        )

    messages = (
        [{"role": "system", "content": system_prompt}]
        + list(conversation_history())
        + [{"role": "system", "content": status_message}]
    )

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages
    )
    return response.choices[0].message.content.strip()


def analyze_user_response(client: OpenAI, user_text: str) -> Dict[str, List[str]]:
    """Classify which topics were covered in a participant response."""
    
    prompt = (
        "You extract which of the following topics are supported by the provided "
        "interview response: leadership, technical_competence, team_orientation. "
        "Be generous in detection - include topics even if mentioned negatively or briefly. "
        "Return a JSON object with this structure:\n"
        '{\n'
        '  "topics": [\n'
        '    {\n'
        '      "topic_id": "leadership",\n'
        '      "notes": ["short bullet point about what was said"]\n'
        '    }\n'
        '  ]\n'
        '}\n\n'
        "Topic definitions:\n"
        "- leadership: managing, directing, influencing, decision-making, inspiring, motivating, "
        "leading teams, delegation. Include negative mentions like 'rude to superiors'\n"
        "- technical_competence: technical skills, knowledge, expertise, problem-solving, "
        "analytical abilities. Include phrases like 'knows his shit', 'reads manuals'\n"
        "- team_orientation: collaboration, interpersonal skills, communication, helping others, "
        "teamwork. Include 'inspires the troops', 'gets team going'\n"
    )

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": prompt},
            {"role": "user", "content": user_text},
        ],
        response_format={"type": "json_object"}
    )

    try:
        payload = json.loads(response.choices[0].message.content)
    except json.JSONDecodeError:
        return {}
    
    result: Dict[str, List[str]] = {}
    for topic in payload.get("topics", []):
        topic_id = topic.get("topic_id")
        if topic_id in TOPIC_METADATA:
            result.setdefault(topic_id, [])
            for note in topic.get("notes", []):
                cleaned_note = note.strip()
                if cleaned_note:
                    result[topic_id].append(cleaned_note)
    return result


def record_topic_notes(user_text: str, notes_by_topic: Dict[str, List[str]]) -> None:
    """Persist evidence for each topic from the latest response."""
    for topic_id, notes in notes_by_topic.items():
        st.session_state.covered_topics.add(topic_id)
        st.session_state.topic_notes[topic_id].append(
            {
                "verbatim": user_text.strip(),
                "notes": notes,
            }
        )


def generate_summary(client: OpenAI, persona: str) -> str:
    """Produce a structured summary of the interview."""
    topic_payload = []
    for topic_id, entries in st.session_state.topic_notes.items():
        topic_payload.append(
            {
                "topic": TOPIC_METADATA[topic_id]["label"],
                "entries": entries,
            }
        )

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": (
                    "You create concise, professional interview summaries organized by topic."
                ),
            },
            {
                "role": "user",
                "content": json.dumps(
                    {
                        "persona": persona,
                        "topic_findings": topic_payload,
                        "instructions": (
                            "Write 2-3 bullet points per topic that synthesize the collected notes. "
                            "Mention when limited information was provided."
                        ),
                    }
                ),
            },
        ],
    )
    return response.choices[0].message.content.strip()


def build_topic_csv(persona: str) -> bytes:
    """Generate a CSV file of the captured notes grouped by topic."""
    buffer = io.StringIO()
    writer = csv.DictWriter(
        buffer,
        fieldnames=["persona", "topic", "note_index", "note_summary", "verbatim_response"],
    )
    writer.writeheader()

    for topic_id, metadata in TOPIC_METADATA.items():
        entries = st.session_state.topic_notes.get(topic_id, [])
        if not entries:
            writer.writerow(
                {
                    "persona": persona,
                    "topic": metadata["label"],
                    "note_index": "",
                    "note_summary": "",
                    "verbatim_response": "",
                }
            )
            continue

        for idx, entry in enumerate(entries, start=1):
            note_summary = " | ".join(entry.get("notes", []))
            writer.writerow(
                {
                    "persona": persona,
                    "topic": metadata["label"],
                    "note_index": idx,
                    "note_summary": note_summary,
                    "verbatim_response": entry.get("verbatim", ""),
                }
            )

    return buffer.getvalue().encode("utf-8")


def main() -> None:
    client = require_api_key()

    st.title("ZEPHYRON Interview Assistant")
    st.caption(
        "Guide structured interviews with managers and colleagues to capture insights about an employee."
    )

    persona = st.selectbox("Choose participant type", list(PERSONA_OPTIONS.keys()))

    if "persona" not in st.session_state or st.session_state.persona != persona:
        initialize_session(persona)
        try:
            opening = generate_assistant_message(client, persona)
            st.session_state.messages.append({"role": "assistant", "content": opening})
            st.session_state.llm_history.append({"role": "assistant", "content": opening})
        except Exception as exc:
            st.error(f"Failed to generate the opening prompt: {exc}")

    with st.sidebar:
        st.header("Topic Coverage")
        for topic_id, metadata in TOPIC_METADATA.items():
            st.checkbox(
                metadata["label"],
                value=topic_id in st.session_state.covered_topics,
                disabled=True,
            )
        st.markdown(
            "Use the finalize button below once you've wrapped up to save notes and get a summary."
        )

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    if st.session_state.finalized:
        st.info("Interview finalized. Download the CSV or copy the summary below.")
    else:
        user_text = st.chat_input("Type your response here")
        if user_text:
            st.session_state.messages.append({"role": "user", "content": user_text})
            st.session_state.llm_history.append({"role": "user", "content": user_text})

            try:
                notes_by_topic = analyze_user_response(client, user_text)
                record_topic_notes(user_text, notes_by_topic)
            except Exception as exc:
                st.warning(f"Could not analyze topics for the response: {exc}")

            try:
                reply = generate_assistant_message(client, persona)
                st.session_state.messages.append({"role": "assistant", "content": reply})
                st.session_state.llm_history.append({"role": "assistant", "content": reply})
            except Exception as exc:
                st.error(f"Failed to generate the next prompt: {exc}")

            st.rerun()

    st.divider()
    if st.button("Finalize interview and generate summary", disabled=st.session_state.finalized):
        try:
            summary_text = generate_summary(client, persona)
            csv_bytes = build_topic_csv(persona)
        except Exception as exc:
            st.error(f"Finalization failed: {exc}")
        else:
            st.session_state.summary = summary_text
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