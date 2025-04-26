import streamlit as st
from embedding import load_embedding_model, create_embeddings
from query_transformer import load_query_transformer, transform_query

st.title("🔎 RAG Pipeline Visualizer")

# Sidebar controls
st.sidebar.title("Settings")
embedding_model_name = st.sidebar.selectbox("Embedding Model", ["all-MiniLM-L6-v2", "intfloat/e5-small-v2"])
query_model_name = st.sidebar.selectbox("Query Transformer Model", ["microsoft/phi-3-mini-128k-instruct", "microsoft/phi-2"])
top_k = st.sidebar.slider("Top-K Documents", 1, 10, 5)
chunk_size = st.sidebar.slider("Chunk Size (tokens)", 100, 1000, 500)
use_query_transformer = st.sidebar.checkbox("Use Query Transformer", True)

# Load models
embedding_model = load_embedding_model(embedding_model_name)
query_pipeline = load_query_transformer(query_model_name)

# User input
query = st.text_input("Ask your question:")

if st.button("Run Retrieval"):
    if query:
        st.subheader("🔹 Original Query")
        st.write(query)

        # Step 1: Query Transformation
        if use_query_transformer:
            transformed_query = transform_query(query_pipeline, query)
            st.subheader("🔹 Transformed Query")
            st.write(transformed_query)
        else:
            transformed_query = query

        # Step 2: Retrieval (Dummy - you can link real retriever)
        dummy_chunks = [f"Document chunk {i} relevant to '{transformed_query}'" for i in range(1, top_k+1)]
        st.subheader("🔹 Retrieved Chunks")
        for chunk in dummy_chunks:
            st.write(f"- {chunk}")

        from logger import save_log

        # Save log
        save_log({
            "user_query": query,
            "transformed_query": transformed_query,
            "retrieved_chunks": dummy_chunks,
            "embedding_model": embedding_model_name,
            "query_transformer_model": query_model_name,
            "top_k": top_k,
            "chunk_size": chunk_size,
            "status": "success"
        })


    else:
        st.warning("Please enter a question!")

