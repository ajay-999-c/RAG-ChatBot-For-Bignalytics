
import time
from vectorstore import load_vectorstore
from query_transformer import rewrite_query_with_tracking, split_rewritten_query
from retriever import get_dynamic_retriever, retrieve_for_each_subquestion, merge_contexts
from generator import generator_chain
from memory import add_message_to_history, initialize_user_session
from utils import generate_user_id
from prompt_builder import build_final_prompt
from logger import log_pipeline_step

# Load vectorstore once (reuse it for all users)
VECTORSTORE_PATH = "./faiss_index"
vectorstore = load_vectorstore(VECTORSTORE_PATH)


def process_query_detailed(user_query: str, ip_address: str, user_agent: str) -> dict:
    user_id = generate_user_id(ip_address, user_agent)
    initialize_user_session(user_id)

    start_pipeline = time.time()

    # Rewrite Query
    rewritten_query = rewrite_query_with_tracking(user_query, user_id)
    sub_questions = split_rewritten_query(rewritten_query)

    # Retrieval
    retriever = get_dynamic_retriever(rewritten_query)
    retrieved_contexts = retrieve_for_each_subquestion(sub_questions, retriever, top_k=5)
    final_context = merge_contexts(retrieved_contexts)

    # Build final prompt
    final_prompt = build_final_prompt(user_id, final_context, user_query)

    # Generate Answer
    generated_answer = generator_chain.invoke({"input": final_prompt})

    # Save conversation
    add_message_to_history(user_id, user_query, generated_answer)

    end_pipeline = time.time()

    return {
        "rewritten_query": rewritten_query,
        "sub_questions": sub_questions,
        "retrieved_chunks": [doc.page_content for doc in retrieved_contexts],
        "generated_answer": generated_answer,
        "input_tokens": len(user_query.split()),  # rough estimation
        "output_tokens": len(generated_answer.split()),  # rough estimation
        "total_time": end_pipeline - start_pipeline,
    }
