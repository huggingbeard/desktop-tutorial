from __future__ import annotations
"""Interactive Streamlit app for an LLM-guided interview assistant.
Revised 2025-10-19 for smoother UX and cleaner logic.
"""

import csv, io, json, os
from typing import Dict, List, Sequence
import streamlit as st
from openai import OpenAI

st.set_page_config(page_title="Interview Assistant", page_icon="🗣️")

PERSONA_OPTIONS = {"Boss": "manager", "Colleague": "colleague", "Self": "self-evaluation"}
TOPIC_METADATA: Dict[str, Dict[str, str]] = {
    "leadership": {"label": "Leadership"},
    "technical_competence": {"label": "Technical Competence"},
    "team_orientation": {"label": "Team Orientation"},
}

# ----------  API key ----------
@st.cache_resource(show_spinner=False)
def get_openai_client(api_key: str) -> OpenAI:
    return OpenAI(api_key=api_key)

def require_api_key() -> OpenAI:
    key=None
    if hasattr(st.secrets,"get"): key=st.secrets.get("OPENAI_API_KEY")
    else:
        try: key=st.secrets["OPENAI_API_KEY"]
        except: key=None
    if not key: key=os.environ.get("OPENAI_API_KEY")
    if not key: st.error("OPENAI_API_KEY missing"); st.stop()
    return get_openai_client(key)

# ----------  Session ----------
def initialize_session(persona:str)->None:
    st.session_state.persona=persona
    st.session_state.messages=[]
    st.session_state.llm_history=[]
    st.session_state.covered_topics=set()
    st.session_state.topic_notes={t:[] for t in TOPIC_METADATA}
    st.session_state.finalized=False
    st.session_state.summary=None
    st.session_state.csv_bytes=None
    st.session_state.txt_bytes=None

def conversation_history()->Sequence[Dict[str,str]]: return st.session_state.llm_history
def persona_description(p:str)->str: return f"You are speaking with a {PERSONA_OPTIONS.get(p,p.lower())}."

def coverage_status()->str:
    return "; ".join(f"{m['label']}: {'covered' if t in st.session_state.covered_topics else 'open'}"
                     for t,m in TOPIC_METADATA.items())
def next_uncovered_topic()->str|None:
    for t in TOPIC_METADATA:
        if t not in st.session_state.covered_topics: return t

# ----------  Core LLM ----------
def generate_assistant_message(c:OpenAI,persona:str)->str:
    desc=PERSONA_OPTIONS.get(persona,persona.lower())
    pronouns="you/your" if persona=="Self" else "they/their/them"
    system=(
        f"You are interviewing a {desc}. Use pronouns {pronouns}. "
        "Ask one warm, professional question at a time. "
        "Ensure all three topics—leadership, technical_competence, team_orientation—are covered."
    )
    status=(f"Participant type: {persona}. {persona_description(persona)} "
            f"Topic status: {coverage_status()}.")
    if not conversation_history():
        status+=" Begin with an open question about the employee’s overall strengths."
    elif next_uncovered_topic():
        lbl=TOPIC_METADATA[next_uncovered_topic()]['label']
        status+=f" Next, naturally explore {lbl}."
    else:
        status+=" All topics covered; wrap up politely."

    r=c.chat.completions.create(model="gpt-4o-mini",
        messages=[{"role":"system","content":system}]+list(conversation_history())+
                 [{"role":"system","content":status}],
        max_tokens=350)
    return r.choices[0].message.content.strip()

def analyze_user_response(c:OpenAI,text:str)->Dict[str,List[str]]:
    """Be strict—only mark topic if explicit clues."""
    rules=(
        "Determine which topics appear explicitly:\n"
        "- leadership: words like lead, manage, supervise, inspire, motivate, delegate.\n"
        "- technical_competence: words like skill, knowledge, expertise, solve, output quality, analysis.\n"
        "- team_orientation: words like team, cooperate, help, collaborate, communicate, morale.\n"
        "Be strict: do NOT infer from generic negativity or laziness.\n"
        "Return JSON {\"topics\":[{\"topic_id\":\"...\",\"notes\":[\"...\"]}]}."
    )
    try:
        r=c.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role":"system","content":rules},
                      {"role":"user","content":text}],
            response_format={"type":"json_object"},
            max_tokens=300)
        data=json.loads(r.choices[0].message.content)
    except Exception: data={}
    out={}
    for t in data.get("topics",[]):
        tid=t.get("topic_id")
        if tid in TOPIC_METADATA:
            out[tid]=[n.strip() for n in t.get("notes",[]) if n.strip()]
    return out

def record_topic_notes(user_text:str,notes:Dict[str,List[str]])->None:
    for tid,vals in notes.items():
        st.session_state.topic_notes[tid].append({"verbatim":user_text,"notes":vals})
        if vals: st.session_state.covered_topics.add(tid)

def generate_summary(c:OpenAI,persona:str)->str:
    payload=[{"topic":TOPIC_METADATA[t]["label"],"entries":v}
             for t,v in st.session_state.topic_notes.items()]
    r=c.chat.completions.create(model="gpt-4o-mini",max_tokens=700,
        messages=[{"role":"system","content":"Write concise interview summaries by topic."},
                  {"role":"user","content":json.dumps(
                      {"persona":persona,"topic_findings":payload,
                       "instructions":"2–3 bullets per topic. Be neutral, concise."})}])
    return r.choices[0].message.content.strip()

def build_topic_csv(persona:str)->bytes:
    buf=io.StringIO(); w=csv.DictWriter(buf,
      fieldnames=["persona","topic","note_index","note_summary","verbatim_response"])
    w.writeheader()
    for tid,meta in TOPIC_METADATA.items():
        entries=st.session_state.topic_notes.get(tid,[])
        if not entries: w.writerow({"persona":persona,"topic":meta["label"]}); continue
        for i,e in enumerate(entries,1):
            w.writerow({"persona":persona,"topic":meta["label"],"note_index":i,
                        "note_summary":" | ".join(e["notes"]),"verbatim_response":e["verbatim"]})
    return buf.getvalue().encode()

def build_transcript_txt()->bytes:
    lines=[]
    for m in st.session_state.messages:
        who="USER:" if m["role"]=="user" else "ASSISTANT:"
        lines.append(f"{who} {m['content']}")
    return "\n".join(lines).encode("utf-8")

# ----------  Main ----------
def main()->None:
    c=require_api_key()
    st.title("ZEPHYRON Interview Assistant")
    st.caption("LLM-guided structured interviews for qualitative evaluations.")

    persona=st.selectbox("Choose participant type", list(PERSONA_OPTIONS.keys()))
    if "persona" not in st.session_state or st.session_state.persona!=persona:
        initialize_session(persona)
        opening=generate_assistant_message(c,persona)
        st.session_state.messages.append({"role":"assistant","content":opening})
        st.session_state.llm_history.append({"role":"assistant","content":opening})

    with st.sidebar:
        st.header("Topic Coverage")
        for t,m in TOPIC_METADATA.items():
            st.checkbox(m["label"], value=t in st.session_state.covered_topics, disabled=True)
        st.caption("Finalize below to get summary and downloads.")

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]): st.write(msg["content"])

    if not st.session_state.finalized:
        user_text=st.chat_input("Type your response here")
        if user_text:
            st.session_state.messages.append({"role":"user","content":user_text})
            st.session_state.llm_history.append({"role":"user","content":user_text})
            try:
                notes=analyze_user_response(c,user_text)
                record_topic_notes(user_text,notes)
                reply=generate_assistant_message(c,persona)
                st.session_state.messages.append({"role":"assistant","content":reply})
                st.session_state.llm_history.append({"role":"assistant","content":reply})
            except Exception as e: st.error(f"Model error: {e}")
            st.rerun()

    st.divider()
    if st.button("Finalize interview and generate summary", disabled=st.session_state.finalized):
        try:
            summary=generate_summary(c,persona)
            csv_bytes=build_topic_csv(persona)
            txt_bytes=build_transcript_txt()
        except Exception as e:
            st.error(f"Finalization failed: {e}")
        else:
            st.session_state.summary=summary
            st.session_state.csv_bytes=csv_bytes
            st.session_state.txt_bytes=txt_bytes
            st.session_state.finalized=True
            st.rerun()

    if st.session_state.finalized and st.session_state.summary:
        st.subheader("Interview Summary")
        st.markdown(st.session_state.summary)
        st.download_button("Download topic notes CSV",
            data=st.session_state.csv_bytes,file_name="interview_notes.csv",mime="text/csv")
        st.download_button("Download full transcript (.txt)",
            data=st.session_state.txt_bytes,file_name="interview_transcript.txt",mime="text/plain")

if __name__=="__main__": main()
    main()
