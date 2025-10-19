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
        next_topic_idx = st.session_state.current_topic_index + 1
        next_topic = TOPIC_SEQUENCE[next_topic_idx] if next_topic_idx < len(TOPIC_SEQUENCE) else None
        
        if next_topic:
            next_topic_info = TOPIC_METADATA[next_topic]
            decision_prompt = f"""After responding, decide if you have enough information about {TOPIC_METADATA[current_topic]['label']} to move on.

Respond with JSON:
{{
    "response": "your conversational response here",
    "move_to_next_topic": true/false,
    "key_insights": "brief summary of what you learned from this response"
}}

If move_to_next_topic is TRUE, your response should:
1. Briefly acknowledge what they just said (1 sentence)
2. Transition to the next topic: {next_topic_info['label']}
3. Ask your opening question about {next_topic_info['label']}

ALL IN ONE MESSAGE. Example: "That leadership during the launch really stands out. Now let's talk about collaboration. How does {st.session_state.employee_name} work with the team day-to-day?"

Set move_to_next_topic to true if:
- You have 1-2 good specific examples on this topic
- The user explicitly asks to move on
- You've had 3+ exchanges on this topic

If FALSE, just ask one follow-up question about the current topic: {TOPIC_METADATA[current_topic]['label']}."""
        else:
            # Last topic, moving to wrap-up
            decision_prompt = f"""After responding, decide if you have enough information about {TOPIC_METADATA[current_topic]['label']} to move on.

Respond with JSON:
{{
    "response": "your conversational response here",
    "move_to_next_topic": true/false,
    "key_insights": "brief summary of what you learned from this response"
}}

If move_to_next_topic is TRUE, your response should wrap up the interview:
1. Acknowledge their last point
2. Ask if there's anything else important to share about {st.session_state.employee_name}
3. Be warm and appreciative

ALL IN ONE MESSAGE.

Set it to true if you have enough on this topic."""
    else:
        # Already wrapping up - this is the FINAL response
        decision_prompt = f"""This is your FINAL statement to close the interview about {st.session_state.employee_name}.

Respond with JSON:
{{
    "response": "your final closing statement",
    "move_to_next_topic": false,
    "key_insights": "brief summary of this final point"
}}

Your response should:
1. Acknowledge what they just shared (especially if it's important feedback)
2. Thank them for their time and insights
3. Close definitively - NO questions, NO invitation to continue

Make it warm but final. This is the last thing you'll say in this interview."""
    
    try:
        r = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                *history[:-1],
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
        return "Thanks for sharing that. What else can you tell me?", False, user_text
