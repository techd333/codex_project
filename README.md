# JsonRAGChatbot

JsonRAGChatbot is a simple RAG-based chatbot built with Python and the OpenAI API.

## Usage

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Set your OpenAI API key:
   ```bash
   export OPENAI_API_KEY=your_key
   ```
3. Launch the Streamlit app:
   ```bash
   streamlit run streamlit_app.py
   ```

The app will open in your browser and store conversation history in
`chat_history.json`, reusing previous answers when similar questions are asked
again.

> The original terminal chatbot can still be started with
> `python json_rag_chatbot.py` if desired.
