# generator.py

from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableSequence
import os
from logger import log_event  # Import logger to log loading status

# Load optional environment config
USE_GROQ = os.getenv("USE_GROQ", "false").lower() == "true"  # Default is False

try:
    if USE_GROQ:
        print("🔵 Loading Groq LLM model: mixtral-8x7b-32768...")
        from langchain_groq import ChatGroq

        generator_llm = ChatGroq(
            groq_api_key=os.getenv("GROQ_API_KEY"),
            model_name="mixtral-8x7b-32768",  # Or "llama3-70b-8192"
            temperature=0.2
        )
        log_event("✅ Initialized Groq ChatGroq LLM: mixtral-8x7b-32768")
        print("✅ Successfully initialized Groq model.")

    else:
        print("🔵 Loading local Ollama LLM model: gemma-2b-it...")
        from langchain_community.llms import Ollama

        generator_llm = Ollama(
            model="gemma-2b-it",  # Adjust to gemma-3b if you pulled that, or gemma-1b-it for 1B
            base_url="http://localhost:11434",
            temperature=0.2
        )
        log_event("✅ Initialized Ollama local LLM: gemma-2b-it")
        print("✅ Successfully initialized Ollama model.")

except Exception as e:
    log_event(f"❌ Failed to initialize LLM: {str(e)}")
    print(f"❌ Error initializing LLM: {str(e)}")
    raise e

# Prompt Template
print("🔵 Building Generation Prompt Template...")
generation_prompt = PromptTemplate.from_template("""
You are an AI assistant helping users.

Use the provided context to answer the question accurately.
If the context is insufficient, politely respond "Sorry, not enough information."

Context:
{context}

Question:
{question}
""")
print("✅ Prompt Template ready.")

# Build Generator Chain
generator_chain = generation_prompt | generator_llm
print("✅ Generator Chain ready for use.")
