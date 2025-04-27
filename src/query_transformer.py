# query_transformer.py

from langchain_community.llms import Ollama
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableSequence
import time
from utils import count_tokens
from logger import log_pipeline_step

# Initialize Query Rewrite LLM
print("🔵 Initializing Query Rewriting LLM (Phi 2.7b) from Ollama...")

rewrite_llm = Ollama(
    model="phi:2.7b-chat-v2-q4_0",
    base_url="http://localhost:11434",
    temperature=0.0
)

print("✅ Query Rewrite LLM loaded successfully.")

# Create Prompt Template
print("🔵 Building Query Rewrite Prompt Template...")

query_rewrite_prompt = PromptTemplate.from_template("""
You are tasked with expanding the user's query into multiple clear and concise sub-questions for better retrieval.
Format output as numbered list.

Original Query:
{original_query}

Expanded Sub-Questions:
1.
2.
3.
""")

print("✅ Query Rewrite Prompt Template ready.")

# Build Rewrite Chain
rewrite_chain = query_rewrite_prompt | rewrite_llm
print("✅ Query Rewrite Chain ready.")

def rewrite_query_with_tracking(user_query: str, user_id: str):
    print(f"🔵 Starting query transformation for user_id: {user_id}...")
    input_tokens = count_tokens(user_query)
    start_time = time.time()

    # Actual Rewriting
    rewritten = rewrite_chain.invoke({"original_query": user_query})

    end_time = time.time()
    output_tokens = count_tokens(rewritten)

    print(f"✅ Query transformation completed.")
    print(f"🧠 Input Tokens: {input_tokens} | Output Tokens: {output_tokens}")
    print(f"🕐 Time Taken: {end_time - start_time:.2f} seconds")

    # Logging
    log_pipeline_step("Query Transformation", user_query, input_tokens, output_tokens, end_time-start_time, user_id=user_id)

    return rewritten

def split_rewritten_query(rewritten_query: str) -> list:
    lines = rewritten_query.strip().split("\n")
    questions = [line.lstrip("1234567890. ").strip() for line in lines if line.strip() and line.strip()[0].isdigit()]
    return questions
