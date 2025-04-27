# query_transformer.py

from langchain_community.llms import Ollama
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableSequence
import time
from utils import count_tokens
from logger import log_pipeline_step

print("\n🔵 [System] Initializing Query Rewriting LLM (Gemma3-1b) from Ollama...\n")

rewrite_llm = Ollama(
    model="gemma3:1b",  # ✅ Switched to Gemma3-1b
    base_url="http://localhost:11434",
    temperature=0.0
)
print("✅ [Success] Query Rewrite LLM loaded successfully.\n")

# Stronger Query Rewriting Prompt
print("🔵 [System] Building Stronger Query Rewrite Prompt Template...\n")

query_rewrite_prompt = PromptTemplate.from_template("""
You are an AI assistant designed to support the Bignalytics Educational Institute.

You must help reformulate user queries to improve retrieval in a RAG system.

STRICT INSTRUCTIONS:
- You are helping answer questions about: courses, fees, batches, placements at Bignalytics.
- Expand into 2–4 sub-questions only if meaningful.
- **Strictly preserve** original user intent.
- **Stay focused** on the user's topic (fees, courses, placements, batches).
- Do NOT invent or hallucinate unrelated questions.
- If the user query is already specific, just rephrase slightly.

Original Query:
{original_query}

Expanded Sub-Questions:
1.
2.
3.
""")

print("✅ [Success] Stronger Query Rewrite Prompt ready.\n")

# Build Rewrite Chain
rewrite_chain = query_rewrite_prompt | rewrite_llm
print("✅ [Success] Query Rewrite Chain ready.\n")

def rewrite_query_with_tracking(user_query: str, user_id: str):
    print(f"🔵 [Transform] Starting query transformation for user_id: {user_id}...\n")
    input_tokens = count_tokens(user_query)
    start_time = time.time()

    rewritten = rewrite_chain.invoke({"original_query": user_query})

    end_time = time.time()
    output_tokens = count_tokens(rewritten)

    print(f"✅ [Transform] Query transformation completed.\n")
    print(f"🧠 Input Tokens: {input_tokens} | Output Tokens: {output_tokens}")
    print(f"🕐 Time Taken: {end_time - start_time:.2f} seconds\n")

    log_pipeline_step("Query Transformation", user_query, input_tokens, output_tokens, end_time - start_time, user_id=user_id)

    return rewritten

def split_rewritten_query(rewritten_query: str) -> list:
    lines = rewritten_query.strip().split("\n")
    questions = [line.lstrip("1234567890. ").strip() for line in lines if line.strip() and line.strip()[0].isdigit()]
    return questions
