# query_transformer.py

from langchain_community.llms import Ollama
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableSequence
import time
from utils import count_tokens
from logger import log_pipeline_step

print("\n🔵 [System] Initializing Query Rewriting LLM (Gemma3-1b) from Ollama...\n")

rewrite_llm = Ollama(
    model="gemma3:1b",
    base_url="http://localhost:11434",
    temperature=0.0
)

print("✅ [Success] Query Rewrite LLM loaded successfully.\n")

# Two different prompt templates

# 1. Reword-only prompt (for specific queries)
reword_prompt = PromptTemplate.from_template("""
You are a helpful assistant at Bignalytics Educational Institute.

The user query is already specific. 
Your job is to simply **rephrase** it slightly to improve clarity, without changing the original meaning.

Original Query:
{original_query}

Reworded Query:
""")

# 2. Expand-only prompt (for general queries)
expand_prompt = PromptTemplate.from_template("""
You are an AI assistant at Bignalytics Educational Institute.

Expand the broad user query into 2–3 specific sub-questions to help in retrieving relevant information about courses, fees, batches, placements.

STRICT RULES:
- Preserve the original meaning.
- Stay focused only on Bignalytics education topics.
- Do not invent unrelated new topics.

Original Query:
{original_query}

Expanded Sub-Questions:
1.
2.
3.
""")

print("✅ [Success] Built Reword + Expand Prompt Templates.\n")

# Build chains
reword_chain = reword_prompt | rewrite_llm
expand_chain = expand_prompt | rewrite_llm
print("✅ [Success] Chains ready.\n")

# Helper: Detect specificity
def is_query_specific(user_query: str) -> bool:
    """Heuristic: Detect if query is already specific."""
    specific_keywords = ["fee", "fees", "placement", "batch", "duration", "discount", "EMI", "installment", "admission"]
    word_count = len(user_query.split())

    if word_count <= 12:
        return True

    for keyword in specific_keywords:
        if keyword.lower() in user_query.lower():
            return True

    return False

# Main rewrite function
def rewrite_query_with_tracking(user_query: str, user_id: str):
    print(f"🔵 [Transform] Starting smart query transformation for user_id: {user_id}...\n")
    input_tokens = count_tokens(user_query)
    start_time = time.time()

    # Detect specificity
    if is_query_specific(user_query):
        print("🧠 Detected Specific Query. Only rewording...\n")
        rewritten = reword_chain.invoke({"original_query": user_query})
    else:
        print("🧠 Detected General Query. Expanding into sub-questions...\n")
        rewritten = expand_chain.invoke({"original_query": user_query})

    end_time = time.time()
    output_tokens = count_tokens(rewritten)

    print(f"✅ [Transform] Query transformation completed.\n")
    print(f"🧠 Input Tokens: {input_tokens} | Output Tokens: {output_tokens}")
    print(f"🕐 Time Taken: {end_time - start_time:.2f} seconds\n")

    # Logging
    log_pipeline_step("Query Transformation", user_query, input_tokens, output_tokens, end_time - start_time, user_id=user_id)

    return rewritten

# Helper: Split expanded query into sub-questions
def split_rewritten_query(rewritten_query: str) -> list:
    """Split expanded sub-questions if they exist."""
    lines = rewritten_query.strip().split("\n")
    questions = [line.lstrip("1234567890. ").strip() for line in lines if line.strip() and line.strip()[0].isdigit()]
    return questions
