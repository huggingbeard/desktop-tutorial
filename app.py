from __future__ import annotations
"""Interactive Streamlit app for an LLM-guided interview assistant. quicker print of user input+better self-eval"""

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
    st.session_state.txt_bytes = None  # NEW: transcript bytes
    st.session_state.just_asked_followup = False
    st.session_state.followup_count = 0  # Track total follow-ups to avoid endless loops
    st.session_state.awaiting_assistant_response = False  # Flag to show user message immediately


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
    persona_desc = PERSONA_OPTIONS.get(persona, persona.lower())
    
    # Different pronouns for self-evaluation vs others (UNCHANGED)
    if persona == "Self":
        pronoun_instruction = (
            "You are conducting a self-evaluation interview with the employee themselves. "
            "Use second-person pronouns: 'you', 'your', 'yours'. "
            "You are speaking directly TO the employee about their own performance."
        )
    else:
        pronoun_instruction = (
            f"You are interviewing the {persona_desc} ABOUT an employee. "
            f"CRITICAL: The {persona_desc} is NOT the employee - they are talking about someone else. "
            "Always use third-person pronouns: 'they', 'their', 'them', 'he', 'she', 'his', 'her'. "
            f"NEVER use 'you' or 'your' when referring to the employee."
        )
    
    system_prompt = (
        pronoun_instruction + "\n\n"
        "Guide the discussion with warm, professional language, asking one question at a time. "
        "Encourage rich reflections while ensuring the interview ultimately covers leadership, "
        "technical competence, and team orientation. "
        "If a topic is marked as covered, do NOT ask about it again. "
        "When asking for examples, reference what they already said. "
        "If every topic is already covered, invite final reflections and let the participant know "
        "you will summarize what you've heard."
    )

    # Pronoun reminder based on persona (UNCHANGED)
    if persona == "Self":
        pronoun_reminder = "Remember: You are speaking TO the employee (use you/your)."
    else:
        pronoun_reminder = f"Remember: You are asking the {persona_desc} about THE EMPLOYEE (use they/their/them)."
    
    status_message = (
        f"Participant type: {persona}. {persona_description(persona)} "
        f"{pronoun_reminder} "
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
    """Classify which topics were covered in a participant response.

    CHANGE: stricter rules so 'leadership' is NOT inferred from generic negativity
    like 'lazy', 'late', 'misses deadlines'. Require explicit leadership language.
    """
    prompt = (
        "Decide which of these topics are explicitly discussed in the response: "
        "leadership, technical_competence, team_orientation.\n"
        "Rules (STRICT):\n"
        "- leadership: ONLY if the response explicitly mentions leading/managing/supervising/"
        "delegating/decision-making/inspiring/motivating (words like 'lead', 'manager', "
        "'supervise', 'delegate', 'inspire', 'motivate', 'decision'). Do NOT infer from "
        "generic behavior problems (e.g., lazy, late, misses deadlines) unless the response "
        "connects them to leadership/management acts.\n"
        "- technical_competence: ONLY if it mentions skills, knowledge, expertise, problem-solving, "
        "analytical ability, code quality, output quality, correctness, or domain knowledge.\n"
        "- team_orientation: ONLY if it mentions teamwork, helping others, collaboration, "
        "communication, interpersonal factors, morale, sharing knowledge.\n"
        "Be conservative. If unsure, leave the topic out.\n"
        "Return ONLY JSON:\n"
        "{\n"
        '  "topics": [\n'
        '    {"topic_id": "leadership", "notes": ["sho]()_
