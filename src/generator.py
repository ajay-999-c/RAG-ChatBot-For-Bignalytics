# generator.py

import os
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableSequence
from logger import log_event  # For logging events into file

print("\n🔵 [System] Starting Generator Initialization...\n")

# Load optional environment config
USE_GROQ = os.getenv("USE_GROQ", "false").lower() == "true"  # Default False

try:
    if USE_GROQ:
        print("🔵 [Loading] Attempting to load Groq LLM model (mixtral-8x7b-32768)...")
        from langchain_groq import ChatGroq

        generator_llm = ChatGroq(
            groq_api_key=os.getenv("GROQ_API_KEY"),
            model_name="mixtral-8x7b-32768",  # Or "llama3-70b-8192"
            temperature=0.2
        )
        log_event("✅ Initialized Groq ChatGroq LLM: mixtral-8x7b-32768")
        print("✅ [Success] Groq model loaded successfully.\n")

    else:
        print("🔵 [Loading] Attempting to load Ollama LLM model (gemma3:1b)...")
        from langchain_community.llms import Ollama

        generator_llm = Ollama(
            model="gemma3:1b",  # ✅ Correct model name!
            base_url="http://localhost:11434",
            temperature=0.2
        )
        log_event("✅ Initialized Ollama local LLM: gemma3:1b")
        print("✅ [Success] Ollama local model loaded successfully.\n")

except Exception as e:
    log_event(f"❌ Failed to initialize LLM: {str(e)}")
    print(f"❌ [Error] Failed to initialize LLM: {str(e)}")
    raise e

# Build Prompt Template
print("🔵 [Building] Creating Generation Prompt Template...")
generation_prompt = PromptTemplate.from_template("""
You are an AI assistant helping users.

Use the provided context to answer the question accurately.
If the context is insufficient, politely respond "Sorry, not enough information."

Context:
{context}

Question:
{question}
""")
print("✅ [Success] Generation Prompt Template ready.\n")

# Build Generator Chain
print("🔵 [Building] Combining Prompt and LLM into Generator Chain...")
generator_chain = generation_prompt | generator_llm
print("✅ [Success] Generator Chain ready for use.\n")

print("✅ [System] Generator initialization completed.\n")
