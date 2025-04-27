# retrieval.py

def retrieve_for_each_subquestion(sub_questions: list, retriever, top_k: int = 2):
    """
    Retrieves documents separately for each sub-question.

    Args:
      sub_questions: A list of sub-questions.
      retriever: A retriever object (FAISS, Chroma, etc).
      top_k: Number of top documents to retrieve per sub-question.

    Returns:
      A list of unique retrieved document texts.
    """
    retrieved_docs = []
    for subq in sub_questions:
        docs = retriever.get_relevant_documents(subq)[:top_k]
        retrieved_docs.extend(docs)
    
    # De-duplicate by content
    unique_contexts = list({doc.page_content for doc in retrieved_docs})
    return unique_contexts

def merge_contexts(contexts: list) -> str:
    """
    Merge multiple retrieved contexts into a single string.

    Args:
      contexts: A list of document texts.

    Returns:
      A single large context string for the Generator LLM.
    """
    return "\n\n".join(contexts)
