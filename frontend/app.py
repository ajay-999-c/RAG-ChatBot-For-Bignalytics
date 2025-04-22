import streamlit as st, requests

st.title("Bignalytics Chatbot")

q = st.text_input("Ask me anything about Bignalytics")
if st.button("Send") and q:
    resp = requests.post("http://localhost:8000/rag", json={"question": q}).json()
    st.markdown(resp["answer"])
    with st.expander("Show sources"):
        st.json(resp["sources"])
