# app_streamlit.py

import streamlit as st
from load_data import load_all_data
from vectorstore import load_vectorstore
from query_transformer import rewrite_query_with_tracking, split_rewritten_query
from retriever import get_dynamic_retriever, retrieve_for_each_subquestion, merge_contexts
from generator import generator_chain
from memory import add_message_to_history, initialize_user_session
from utils import generate_user_id
from logger import log_event
from prompt_builder import build_final_prompt
import time

# --- Streamlit UI ---
st.set_page_config(page_title="Bignalytics RAG Chatbot 🔎", page_icon="🔎")

st.title("🔎 Bignalytics RAG Chatbot")
st.write("Ask questions about Bignalytics courses, fees, placements, batches!")

# Sidebar: Settings
st.sidebar.title("Settings")
top_k = st.sidebar.slider("Top-K Chunks to Retrieve", 1, 10, 5)
memory_turns = st.sidebar.slider("Conversation Memory Turns", 1, 5, 3)
use_query_transformer = st.sidebar.checkbox("Use Query Rewriting", True)

# Dummy IP/User-Agent for now (Streamlit doesn't expose real ones easily)
ip_address = "127.0.0.1"
user_agent = "Streamlit-Test-User"

# Initialize session
if "user_id" not in st.session_state:
    st.session_state.user_id = generate_user_id(ip_address, user_agent)
    initialize_user_session(st.session_state.user_id)

# User input box
user_query = st.text_input("Type your question:", key="user_input")

if st.button("Ask"):
    if user_query:
        st.subheader("🔹 User Question")
        st.write(user_query)

        user_id = st.session_state.user_id

        start_time = time.time()

        # Step 1: Rewrite Query
        if use_query_transformer:
            rewritten_query = rewrite_query_with_tracking(user_query, user_id)
            sub_questions = split_rewritten_query(rewritten_query)
        else:
            rewritten_query = user_query
            sub_questions = [user_query]

        # Step 2: Retriever
        retriever = get_dynamic_retriever(rewritten_query)
        retrieved_contexts = retrieve_for_each_subquestion(sub_questions, retriever, top_k=top_k)
        final_context = merge_contexts(retrieved_contexts)

        # Step 3: Build final prompt (system + memory + context + question)
        final_prompt = build_final_prompt(user_id, final_context, user_query)

        # Step 4: Generator call (Groq/Ollama dynamic)
        generated_answer = generator_chain.invoke({"input": final_prompt})

        # Save conversation
        add_message_to_history(user_id, user_query, generated_answer)

        end_time = time.time()

        # Step 5: Display answer
        st.subheader("🔹 Bignalytics Chatbot Answer")
        st.success(generated_answer)

        st.info(f"🕐 Response generated in {end_time-start_time:.2f} seconds")

        # Log event
        log_event(f"USER {user_id} asked: {user_query} | Answered successfully.")
    
    else:
        st.warning("Please type a question before submitting!")

# Optional: Show conversation history
if st.sidebar.checkbox("Show Conversation History"):
    from memory import get_conversation_history

    history = get_conversation_history(st.session_state.user_id)
    st.subheader("🗂️ Conversation History")
    for role, message in history:
        st.write(f"**{role.capitalize()}**: {message}")
