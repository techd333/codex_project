# codex_project

This repository contains a simple RAG-based chatbot built with Python and the OpenAI API.

## Usage

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Set your OpenAI API key:
   ```bash
   export OPENAI_API_KEY=your_key
   ```
3. Start the chatbot:
   ```bash
   python rag_chatbot.py
   ```

The bot stores conversation history in `chat_history.json` and reuses previous
answers when similar questions are asked again.
