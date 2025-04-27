# Bignalytics RAG Chatbot 🚀

An intelligent Retrieval-Augmented Generation (RAG) based chatbot for Bignalytics Educational Institute, focused on answering user questions about courses, fees, batches, and placements.

---

## 🚀 Setup Instructions

### 1. Clone the Repository

```bash
git clone ""
cd bignalytics-rag-chatbot

# Create new environment (example: biganalytics-env)
conda create -n biganalytics python=3.10 -y

# Activate the environment
conda activate biganalytics-env

# Install Python Dependencies
pip install -r requirements.txt

# Install Ollama
# Ollama is needed to run LLMs like Gemma 3 1b locally.

# Install Ollama following the instructions for your OS:
# 👉 https://ollama.com/download

# Pull Gemma 3 1b Model
# Once Ollama is installed and running, open terminal and run:

# Open terminal and run this command
ollama run gemma3:1b
# ✅ This will download the Gemma 3 1b model required for query transformation and answer generation.

## install phi2 model using ollama
ollama pull phi:2.7b-chat-v2-q4_0

