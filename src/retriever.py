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

# def get_dynamic_retriever(user_query: str, vectorstore_path="./faiss_index"):
#     vectorstore = load_vectorstore(vectorstore_path)
#     section_type = classify_query_section(user_query)

#     if section_type and section_type != "general_info":
#         return vectorstore.as_retriever(search_kwargs={"k": 5, "filter": {"section_type": section_type}})
#     return vectorstore.as_retriever(search_kwargs={"k": 5})

def get_dynamic_retriever(user_query: str, vectorstore_path="./faiss_index", db_type="faiss"):
    vectorstore = load_vectorstore(vectorstore_path)
    
    section_type = classify_query_section(user_query)

    retriever = vectorstore.as_retriever(search_kwargs={"k": 10})

    # ✅ Now return BOTH retriever and section_type
    return retriever, section_type



def retrieve_for_each_subquestion(sub_questions, retriever, top_k=5):
    """
    Retrieve documents for each sub-question separately.
    """
    all_retrieved_docs = []
    for sub_q in sub_questions:
        retrieved = retriever.get_relevant_documents(sub_q)
        all_retrieved_docs.extend(retrieved)
    return all_retrieved_docs

def merge_contexts(docs):
    """
    Merge retrieved documents into a single context string.
    """
    context_text = "\n\n".join(doc.page_content for doc in docs)
    return context_text



def filter_retrieved_docs(docs, target_section_type, db_type="faiss"):
    """Manually filter retrieved docs by section_type for FAISS. No filtering needed for Chroma/Qdrant."""
    if db_type != "faiss":
        return docs  # Already filtered inside retrieval if db supports metadata filters

    if not target_section_type or target_section_type == "general_info":
        return docs  # No need to filter

    # Manual filtering for FAISS
    return [doc for doc in docs if doc.metadata.get("section_type") == target_section_type]
