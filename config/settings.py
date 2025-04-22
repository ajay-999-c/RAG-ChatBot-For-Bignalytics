import os
from dotenv import load_dotenv
load_dotenv()                                    # reads .env at project root

EMBED_MODEL_NAME = os.getenv("EMBED_MODEL", "intfloat/e5-small-v2")
RERANK_MODEL_NAME = os.getenv("RERANK_MODEL", "BAAI/bge-reranker-v2-m3")
GEN_MODEL_NAME = os.getenv("GEN_MODEL", "mistralai/Mistral-7B-Instruct-v0.1")

VECTORSTORE_PATH = os.getenv("VECTORSTORE_PATH", "data/faiss_index")
DOCSTORE_PATH = os.getenv("DOCSTORE_PATH", "data/docstore.pkl")

MAX_CONTEXT_CHARS = 1500                        # compression threshold
TOP_K = 6                                       # retriever candidates
