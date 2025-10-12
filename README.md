# Interview Assistant Mock-up

This repository contains a Streamlit prototype for an LLM-guided interview workflow. The mock-up supports two participant personas (Boss and Colleague) and walks through three topics that must be covered during the conversation: leadership, technical competence, and team orientation. OpenAI's API is used to guide the conversation, capture topical coverage, produce a summary, and export notes.

## Repository layout

All of the runnable code lives at the root of the project:

```
├── app.py            # Streamlit front-end and OpenAI workflow
├── README.md         # This guide
├── requirements.txt  # Python dependencies
└── docs/             # Background notes on the workflow
```

You can clone this repository or copy these files into a directory of your choice—there
is no need to create extra sub-folders. The instructions below assume you are operating
from the project root shown above.

## Getting started (command line)

1. **Create a virtual environment (optional but recommended).**
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   ```

   On Windows PowerShell use:
   ```powershell
   python -m venv .venv
   .venv\Scripts\Activate.ps1
   ```

2. **Set the OpenAI API key.** Either export it as an environment variable or add it to Streamlit secrets.
   ```bash
   export OPENAI_API_KEY="sk-your-key"
   ```

   On Windows PowerShell run:
   ```powershell
   $Env:OPENAI_API_KEY = "sk-your-key"
   ```

   Alternatively, create a `.streamlit/secrets.toml` file and include:
   ```toml
   OPENAI_API_KEY = "sk-your-key"
   ```

3. **Install dependencies.**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the Streamlit app from the command line.**
   ```bash
   streamlit run app.py
   ```

5. **Open the app in your browser.**
   Streamlit will print a local URL (usually `http://localhost:8501`). Open it in any modern browser to access the interactive interview flow.

> **Note:** The `python -m compileall app.py` command shown in the testing section only checks that the file compiles; it does not launch the application. Use `streamlit run app.py` to start the interactive UI.

### Running from an IDE

If you prefer an IDE such as VS Code or PyCharm:

1. Open the project folder in your IDE (again, no extra folders are required).
2. Create/activate the same virtual environment inside the IDE terminal, install dependencies, and set `OPENAI_API_KEY` as described above.
3. Use the IDE terminal to execute `streamlit run app.py`, or configure a run configuration that launches that command. Streamlit will still expose the web UI via your browser.

## Using the prototype

- Choose the participant persona (Boss or Colleague) from the dropdown.
- Read the assistant's open-ended kickoff question and respond in the chat input.
- The sidebar tracker shows which topics have been covered based on the conversation.
- The assistant uses OpenAI to craft prompts and the app uses the API to classify which topics were addressed in each response.
- Select **Finalize interview and generate summary** to create an LLM-produced recap and download a CSV of captured notes grouped by topic.
