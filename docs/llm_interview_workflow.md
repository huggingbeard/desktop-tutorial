# LLM Interview Workflow Design

## Objectives
- Collect comprehensive qualitative feedback about an individual from colleagues, managers, and the individual themselves.
- Ensure the conversation begins with open-ended reflection before guiding the interviewer through any remaining required topics.
- Provide a repeatable structure that can be reused across interviews while staying flexible and human-centered.

## High-Level Architecture
1. **Frontend Conversation Interface**
   - Web-based chat UI allowing interviewers to interact with the LLM.
   - Displays the interview context, progress, and outstanding topics.
   - Captures free-form notes and allows manual overrides or skips.
2. **LLM Orchestration Layer**
   - Maintains conversation state (topics covered, interviewee role, persona data).
   - Uses system prompts and dynamic context to guide the LLM.
   - Calls evaluation functions to determine when topics are sufficiently discussed.
3. **Knowledge Base / Config Store**
   - Interview templates describing required topics, sample probing questions, and completion criteria.
   - Persona dossiers (role, tenure, known achievements) collected beforehand.
4. **Persistence & Reporting**
   - Logs conversation transcripts.
   - Outputs structured summaries mapped to the required topics.

## Conversation Flow Logic
1. **Initialization**
   - Select interview type (self, peer, manager) and relevant template.
   - Generate a session-specific system prompt describing objectives, interviewee details, and tone guidelines.
2. **Open Reflection Phase**
   - LLM invites interviewer to share broad reflections (e.g., "Tell me about their recent contributions and overall performance.").
   - The interviewer can type or paste notes; the LLM responds with clarifying questions but avoids checklist prompts.
3. **Coverage Assessment**
   - After each exchange, the orchestration layer evaluates which required topics were mentioned using lightweight NLP (keyword/semantic matching) or LLM-based classification.
   - Topics marked "covered" are summarized; uncovered topics remain in a queue.
4. **Guided Topic Phase**
   - For remaining topics, the LLM introduces them gradually: "We haven't touched on collaboration yet—could you describe how they work with cross-functional teams?"
   - Provide optional probing suggestions and allow the interviewer to skip or mark "not applicable." Provide rationale prompts if skipped.
5. **Closing and Summary**
   - Summarize key points per topic and highlight follow-up items.
   - Generate a final structured report or export to internal systems.

## Implementation Considerations
- **LLM Prompting**: Use role-based system messages ensuring confidentiality, empathy, and avoidance of bias. Few-shot examples can demonstrate desired tone.
- **State Management**: Track topic coverage using a combination of deterministic tags and LLM feedback. Consider storing state in a small database keyed by session ID.
- **Topic Templates**: Represent topics as JSON with fields such as `name`, `description`, `priority`, `sample_questions`, and `completion_criteria` to drive orchestration.
- **Safety & Compliance**: Include guardrails to prevent inappropriate questions and remind interviewers about privacy policies.
- **Human Oversight**: Allow interviewers to edit summaries before finalizing to ensure accuracy and context.

## Tech Stack Suggestions
- **Frontend**: React + Tailwind (or similar) chat UI with a progress indicator.
- **Backend**: Node.js/Express or Python/FastAPI managing session state and LLM calls.
- **LLM Provider**: OpenAI GPT-4o or similar, with function calling to manage state updates.
- **Storage**: PostgreSQL or Firebase for session logs and templates.
- **Analytics**: Track completion rates and topic coverage to improve prompts over time.

## Next Steps
1. Draft interview templates for each persona type.
2. Build a prototype conversation flow using mocked data to validate user experience.
3. Iterate on prompt engineering and coverage detection based on pilot interviews.
