from langchain_community.llms import Ollama
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableSequence
from vectorstore import load_vectorstore

# Phi model for section classification
classifier_llm = Ollama(model="phi:2.7b-chat-v2-q4_0", base_url="http://localhost:11434", temperature=0.0)

classification_prompt = PromptTemplate.from_template("""
Given a user's query, classify into:
- fee_structure
- placement_info
- course_overview
- hands_on_training
- eligibility_criteria
- course_duration
- batch_timing
- discount_offer
- general_info

Respond ONLY with section_type.

Examples...
User Query: {user_query}
Section Type:
""")

classification_chain = classification_prompt | classifier_llm

def classify_query_section(user_query: str):
    response = classification_chain.invoke({"user_query": user_query})
    return response.strip()

def get_dynamic_retriever(user_query: str, vectorstore_path="./faiss_index"):
    vectorstore = load_vectorstore(vectorstore_path)
    section_type = classify_query_section(user_query)

    if section_type and section_type != "general_info":
        return vectorstore.as_retriever(search_kwargs={"k": 5, "filter": {"section_type": section_type}})
    return vectorstore.as_retriever(search_kwargs={"k": 5})
