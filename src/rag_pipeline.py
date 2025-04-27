
import time
from vectorstore import load_vectorstore
from query_transformer import rewrite_query_with_tracking, split_rewritten_query
from retriever import get_dynamic_retriever, retrieve_for_each_subquestion, merge_contexts, filter_retrieved_docs
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

    # Step 1: Rewrite Query
    rewritten_query = rewrite_query_with_tracking(user_query, user_id)
    sub_questions = split_rewritten_query(rewritten_query)

    # Step 2: Retrieval
    retriever, section_type = get_dynamic_retriever(rewritten_query)
    retrieved_docs = retriever.get_relevant_documents(rewritten_query)
    retrieved_docs = filter_retrieved_docs(retrieved_docs, section_type, db_type="faiss")

    # Step 2.5: Fallback Retrieval if 0 Chunks
    if len(retrieved_docs) == 0:
        print("⚠️ No documents found after rewritten query. Falling back to original query retrieval...")
        retriever, section_type = get_dynamic_retriever(user_query)
        retrieved_docs = retriever.get_relevant_documents(user_query)
        retrieved_docs = filter_retrieved_docs(retrieved_docs, section_type, db_type="faiss")

    # Step 3: Build Final Context
    final_context = merge_contexts(retrieved_docs)

    # Step 4: Generate Answer
    generated_answer = generator_chain.invoke({
        "context": final_context,
        "question": user_query
    })

    end_pipeline = time.time()

    # Save Conversation Memory
    add_message_to_history(user_id, user_query, generated_answer)

    return {
        "rewritten_query": rewritten_query,
        "sub_questions": sub_questions,
        "retrieved_chunks": [doc.page_content for doc in retrieved_docs],
        "final_prompt": final_context,
        "generated_answer": generated_answer,
        "input_tokens": len(user_query.split()),
        "output_tokens": len(generated_answer.split()),
        "total_time": end_pipeline - start_pipeline
    }
