# app_streamlit.py

import streamlit as st
import time
from rag_pipeline import process_query_detailed
from utils import generate_user_id
from memory import initialize_user_session, get_conversation_history

# --- Streamlit Settings ---
st.set_page_config(page_title="Bignalytics RAG Chatbot 🔎", page_icon="🔎", layout="wide")

# --- Main Titles ---
st.title("🔎 Bignalytics RAG Chatbot")
st.caption("Ask anything about our courses, fees, placement support, batches!")

# Dummy IP/User-Agent for now (Streamlit doesn't expose real ones easily)
ip_address = "127.0.0.1"
user_agent = "Streamlit-Test-User"

if "user_id" not in st.session_state:
    st.session_state.user_id = generate_user_id(ip_address, user_agent)
    initialize_user_session(st.session_state.user_id)

# Layout: Chat (left) + Process Visualization (right)
col1, col2 = st.columns([3, 1])

# 💬 Main Chat Interface
with col1:
    st.header("💬 Welcome to BigAnalytics Chat")

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    user_query = st.text_input("Ask your question:", key="user_input")

    if st.button("Submit"):
        if user_query:
            # Save user question to history
            st.session_state.chat_history.append(("user", user_query))
            
            with st.spinner("🔄 Processing your question..."):
                # Full RAG Pipeline
                result = process_query_detailed(user_query, ip_address, user_agent)

                # Save result for metadata
                st.session_state.last_result = result

                # Save generated bot answer to history
                st.session_state.chat_history.append(("bot", result["generated_answer"]))

        else:
            st.warning("⚠️ Please enter a question before submitting.")

    # Show Full Chat History
    for role, message in st.session_state.chat_history:
        if role == "user":
            with st.chat_message("user"):
                st.markdown(f"**You:** {message}")
        else:
            with st.chat_message("assistant"):
                st.markdown(f"**Bignalytics Bot:** {message}")

# 📊 Process Visualization + Metadata
with col2:
    st.header("📊 Process Debugging")

    if "last_result" in st.session_state:
        result = st.session_state.last_result

        # Step 1: Query Rewriting
        with st.expander("1️⃣ Query Rewriting ✅"):
            st.markdown(f"**Rewritten Query:**\n\n{result['rewritten_query']}")

        # Step 2: Sub-Question Expansion
        with st.expander("2️⃣ Sub-Questions ✅"):
            for idx, q in enumerate(result['sub_questions']):
                st.markdown(f"**{idx+1}.** {q}")

        # Step 3: Retrieval from Vectorstore
        with st.expander("3️⃣ Retrieval ✅"):
            st.markdown(f"**Chunks Retrieved:** {len(result['retrieved_chunks'])}")
            for idx, chunk in enumerate(result['retrieved_chunks'][:3]):  # Limit to top 3 previews
                st.markdown(f"**Chunk {idx+1}:** {chunk[:300]}...")

        # Step 4: Prompt Construction
        with st.expander("4️⃣ Prompt Context ✅"):
            st.markdown(f"**Context Passed to LLM:**\n\n{result['final_prompt'][:500]}...")  # Truncated

        # Step 5: Answer Generation
        with st.expander("5️⃣ Final Answer ✅"):
            st.markdown(f"**Answer Generated:**\n\n{result['generated_answer']}")

        # Step 6: Performance Tracking
        with st.expander("6️⃣ Performance Metrics ✅"):
            st.markdown(f"**Input Tokens:** {result['input_tokens']}")
            st.markdown(f"**Output Tokens:** {result['output_tokens']}")
            st.markdown(f"**Total Time Taken:** {result['total_time']:.2f} seconds")

    else:
        st.info("ℹ️ Ask your first question to see the full processing steps.")
