import streamlit as st
from rag_chatbot import (
    load_history,
    save_history,
    search_similar,
    generate_answer,
    THRESHOLD,
)

st.title("RAG Chatbot")

if "history" not in st.session_state:
    st.session_state.history = load_history()

# Display previous conversation
for item in st.session_state.history:
    with st.chat_message("user"):
        st.write(item["question"])
    with st.chat_message("assistant"):
        st.write(item["answer"])

if question := st.chat_input("Ask something"):
    with st.chat_message("user"):
        st.write(question)
    record, sim, q_emb = search_similar(question, st.session_state.history)
    if record and sim >= THRESHOLD:
        answer = record["answer"]
        with st.chat_message("assistant"):
            st.write(answer)
    else:
        answer = generate_answer(question)
        with st.chat_message("assistant"):
            st.write(answer)
        st.session_state.history.append({
            "question": question,
            "answer": answer,
            "embedding": q_emb,
        })
        save_history(st.session_state.history)
