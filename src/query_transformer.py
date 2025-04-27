from langchain_community.llms import Ollama
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableSequence
from retrieval import retrieve_for_each_subquestion, merge_contexts


# 1. Connect to your Ollama-hosted Phi model
re_write_llm = Ollama(
    model="phi:2.7b-chat-v2-q4_0",
    temperature=0.0,
    base_url="http://localhost:11434"  # adjust if needed
)

# 2. Create the PromptTemplate
query_rewrite_prompt = PromptTemplate.from_template("""
You are an AI assistant tasked with reformulating user queries
to improve retrieval in a RAG system. 

Given the original query, rewrite it by breaking it into multiple clear and concise sub-questions,
each focusing on a specific aspect of the original query.

Format the rewritten output as a numbered list, where each sub-question is a standalone question.

Original query:
{original_query}

Rewritten sub-questions:
1.
2.
3.
""")

# 3. Set up RunnableSequence (prompt | llm)
query_rewriter = query_rewrite_prompt | re_write_llm

# 4. Function to rewrite query
def rewrite_query(original_query: str) -> str:
    """
    Rewrite the original query into detailed sub-questions.

    Args:
      original_query: The user’s raw question.

    Returns:
      A rewritten query as a numbered list.
    """
    response = query_rewriter.invoke({"original_query": original_query})
    if isinstance(response, str):
        return response.strip()
    if isinstance(response, dict) and "text" in response:
        return response["text"].strip()
    return str(response).strip()

# 5. Function to split rewritten query into list of sub-questions
def split_rewritten_query(rewritten_query: str) -> list:
    """
    Split a numbered list of sub-questions into a clean Python list.

    Args:
      rewritten_query: The output from rewrite_query().

    Returns:
      A list of sub-question strings.
    """
    lines = rewritten_query.strip().split("\n")
    questions = [line.lstrip("1234567890. ").strip() for line in lines if line.strip() and line.strip()[0].isdigit()]
    return questions

# — example usage —
if __name__ == "__main__":
    user_query = "fee structure of your Data Science course"
    rewritten = rewrite_query(user_query)
    sub_questions = split_rewritten_query(rewritten)

    print("Rewritten expanded query:\n", rewritten)
    print("\nExtracted sub-questions:")
    for idx, q in enumerate(sub_questions, start=1):
        print(f" {q}")


    # Assume you already have:
    # - sub_questions: list of sub-questions
    # - retriever: your FAISS/Chroma retriever object

    retrieved_contexts = retrieve_for_each_subquestion(sub_questions, retriever, top_k=2)
    final_context = merge_contexts(retrieved_contexts)

    print(final_context)
