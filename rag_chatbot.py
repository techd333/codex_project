"""Simple RAG-based chatbot using OpenAI API.

The bot keeps a local JSON history of past questions and answers along with
embeddings. When a new question arrives, it searches the history for the most
similar question using cosine similarity. If the similarity is above a
threshold, the previous answer is reused; otherwise, the question is sent to the
OpenAI Chat API and the resulting answer is stored for future reuse.

To run:
  1. Install dependencies: `pip install -r requirements.txt`
  2. Set your API key: `export OPENAI_API_KEY=...`
  3. Start the bot: `python rag_chatbot.py`

The conversation history is stored in `chat_history.json` in the current
working directory.
"""

from __future__ import annotations

import json
import os
from typing import Dict, List, Optional

import numpy as np
from openai import OpenAI

HISTORY_FILE = "chat_history.json"
EMBED_MODEL = "text-embedding-3-small"
CHAT_MODEL = "gpt-3.5-turbo"
THRESHOLD = 0.9

client = OpenAI()


def load_history() -> List[Dict]:
    """Load stored question/answer pairs from disk."""
    if os.path.exists(HISTORY_FILE):
        with open(HISTORY_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return []


def save_history(history: List[Dict]) -> None:
    """Persist conversation history to disk."""
    with open(HISTORY_FILE, "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)


def embed_text(text: str) -> List[float]:
    """Create an embedding for the supplied text using the OpenAI API."""
    response = client.embeddings.create(model=EMBED_MODEL, input=text)
    return response.data[0].embedding


def cosine_similarity(a: List[float], b: List[float]) -> float:
    """Compute cosine similarity between two embedding vectors."""
    vec_a = np.array(a)
    vec_b = np.array(b)
    return float(np.dot(vec_a, vec_b) / (np.linalg.norm(vec_a) * np.linalg.norm(vec_b)))


def search_similar(question: str, history: List[Dict]) -> tuple[Optional[Dict], float, List[float]]:
    """Find the most similar past question to the current one.

    Returns a tuple of (record, similarity, question_embedding).
    """
    q_emb = embed_text(question)
    best: Optional[Dict] = None
    best_sim = -1.0
    for record in history:
        sim = cosine_similarity(q_emb, record["embedding"])
        if sim > best_sim:
            best = record
            best_sim = sim
    return best, best_sim, q_emb


def generate_answer(question: str) -> str:
    """Query the OpenAI Chat API for an answer."""
    response = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[{"role": "user", "content": question}],
    )
    return response.choices[0].message.content.strip()


def chat() -> None:
    """Run an interactive chat session."""
    history = load_history()
    try:
        while True:
            question = input("You: ")
            if not question:
                continue
            if question.lower() in {"quit", "exit"}:
                break
            record, sim, q_emb = search_similar(question, history)
            if record and sim >= THRESHOLD:
                print(f"Bot (retrieved): {record['answer']}")
            else:
                answer = generate_answer(question)
                print(f"Bot: {answer}")
                history.append({"question": question, "answer": answer, "embedding": q_emb})
                save_history(history)
    except KeyboardInterrupt:
        print("\nBye!")


if __name__ == "__main__":
    chat()
