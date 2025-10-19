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
        '    {"topic_id": "leadership", "notes": ["short concise note"]}\n'
        "  ]\n"
        "}"
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
    
    # EXTRA GUARD: leadership only if explicit leadership keywords present in text
    leadership_keywords = (
        "lead", "leads", "leading", "leader", "leadership",
        "manage", "manages", "managing", "manager", "management",
        "supervise", "supervises", "supervision", "supervisor",
        "delegate", "delegates", "delegation",
        "inspire", "inspires", "inspiring", "motivate", "motivates", "motivating",
        "decision", "decisions", "decision-making"
    )
    text_lc = user_text.lower()

    result: Dict[str, List[str]] = {}
    for topic in payload.get("topics", []):
        topic_id = topic.get("topic_id")
        if topic_id in TOPIC_METADATA:
            # apply leadership gate
            if topic_id == "leadership":
                if not any(k in text_lc for k in leadership_keywords):
                    continue  # skip leadership unless explicitly mentioned
            result.setdefault(topic_id, [])
            for note in topic.get("notes", []):
                cleaned_note = note.strip()
                if cleaned_note:
                    result[topic_id].append(cleaned_note)
    return result


def check_if_vague(client: OpenAI, user_text: str) -> bool:
    """Check if the response is too vague and needs a concrete example."""
    
    prompt = (
        "Analyze this interview response. Is it too vague and would benefit from a concrete example? "
        "Return ONLY a JSON object: {\"is_vague\": true/false, \"reason\": \"brief explanation\"}\n\n"
        "Consider vague if:\n"
        "- Uses only generic adjectives without specifics (e.g. 'kind', 'good', 'nice')\n"
        "- Makes claims without examples (e.g. 'team player' without describing behavior)\n"
        "- Mentions problems vaguely (e.g. 'sometimes people are lost' without details)\n"
        "- Brief responses under 10 words\n\n"
        "NOT vague if:\n"
        "- Includes specific examples, behaviors, or situations\n"
        "- Describes concrete actions (e.g. 'buys cakes for birthdays')\n"
        "- User explicitly says they don't know or can't provide examples\n"
        "- Response already has sufficient detail\n"
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
        result = json.loads(response.choices[0].message.content)
        return result.get("is_vague", False)
    except json.JSONDecodeError:
        return False


def generate_followup_for_example(client: OpenAI, user_text: str, persona: str) -> str:
    """Generate a follow-up question asking for a specific example."""
    
    if persona == "Self":
        pronoun_instruction = (
            "You are interviewing the employee themselves in a self-evaluation. "
            "Use second-person pronouns (you/your) when asking for examples."
        )
    else:
        pronoun_instruction = (
            "The person you're interviewing is talking ABOUT an employee (someone else). "
            "Use third-person pronouns (they/their/them/he/she/his/her) when referring to the employee."
        )
    
    prompt = (
        f"{pronoun_instruction}\n\n"
        "The user gave a vague response in an interview. "
        "Generate ONE brief, natural follow-up question (1-2 sentences max) asking for a specific "
        "example or situation that demonstrates what they mentioned. "
        "Reference what they said. Keep it conversational."
    )
    
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": prompt},
            {"role": "user", "content": f"They said: {user_text}"},
        ],
    )
    
    return response.choices[0].message.content.strip()


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
                            "Provide a critical summary. If people are described as unpleasant or headstrong, for example, say so."
                            "Be particularly wary of self-evaluations, which can be self-lauding. Look for concrete examples and evidence"
                            "of particular behavior, not just their assertion. Then summarize."
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

    # Write rows
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


def build_transcript_txt() -> bytes:
    """NEW: Build a plain-text transcript (USER/ASSISTANT lines)."""
    lines = []
    for message in st.session_state.messages:
        role = "USER" if message["role"] == "user" else "ASSISTANT"
        lines.append(f"{role}: {message['content']}")
    return ("\n".join(lines)).encode("utf-8")


def main() -> None:
    client = require_api_key()

    st.title("ZEPHYRON Interview Assistant")
    st.caption(
        "LLM-guided structured interviews for qualitative evaluations."
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
    elif "just_asked_followup" not in st.session_state or not isinstance(st.session_state.get("followup_count", 0), int):
        # Handle case where session exists but doesn't have new attributes or has wrong type
        st.session_state.just_asked_followup = False
        st.session_state.followup_count = 0

    with st.sidebar:
        st.header("Topic Coverage")
        for topic_id, metadata in TOPIC_METADATA.items():
            st.checkbox(
                metadata["label"],
                value=topic_id in st.session_state.covered_topics,
                disabled=True,
            )
        # Safety check for followup_count type
        followup_count = st.session_state.get("followup_count", 0)
        if isinstance(followup_count, int) and followup_count > 0:
            st.caption(f"Follow-ups asked: {followup_count}/5")
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
            # Add user message and rerun immediately to show it
            st.session_state.messages.append({"role": "user", "content": user_text})
            st.session_state.llm_history.append({"role": "user", "content": user_text})
            st.session_state.awaiting_assistant_response = True
            st.rerun()
        
        # Process the last user message if we're awaiting a response
        if st.session_state.awaiting_assistant_response:
            st.session_state.awaiting_assistant_response = False
            
            # Get the last user message
            last_user_msg = st.session_state.llm_history[-1]["content"]

            try:
                notes_by_topic = analyze_user_response(client, last_user_msg)
                record_topic_notes(last_user_msg, notes_by_topic)
            except Exception as exc:
                st.warning(f"Could not analyze topics for the response: {exc}")

            try:
                # Ensure followup_count is an int
                if not isinstance(st.session_state.followup_count, int):
                    st.session_state.followup_count = 0
                
                # Check if we just asked a follow-up and got a response
                if st.session_state.just_asked_followup:
                    # We got the follow-up response, move on regardless of quality
                    st.session_state.just_asked_followup = False
                    st.session_state.followup_count = 0
                    reply = generate_assistant_message(client, persona)
                    st.session_state.messages.append({"role": "assistant", "content": reply})
                    st.session_state.llm_history.append({"role": "assistant", "content": reply})
                
                # Check if response is vague and needs a follow-up (limit to 5 total follow-ups per interview)
                elif st.session_state.followup_count < 5 and check_if_vague(client, last_user_msg):
                    # Ask for a concrete example
                    st.session_state.followup_count += 1
                    st.session_state.just_asked_followup = True
                    followup = generate_followup_for_example(client, last_user_msg, persona)
                    st.session_state.messages.append({"role": "assistant", "content": followup})
                    st.session_state.llm_history.append({"role": "assistant", "content": followup})
                
                else:
                    # Response has enough detail, continue normally
                    st.session_state.just_asked_followup = False
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
            txt_bytes = build_transcript_txt()  # NEW
        except Exception as exc:
            st.error(f"Finalization failed: {exc}")
        else:
            st.session_state.summary = summary_text
            st.session_state.csv_bytes = csv_bytes
            st.session_state.txt_bytes = txt_bytes  # NEW
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

        # NEW: plain-text transcript download
        st.download_button(
            "Download full transcript (.txt)",
            data=st.session_state.txt_bytes,
            file_name="interview_transcript.txt",
            mime="text/plain",
        )


if __name__ == "__main__":
    main()
