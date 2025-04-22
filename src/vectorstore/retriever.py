import faiss, pickle
import numpy as np
from sentence_transformers import SentenceTransformer
from config.settings import (VECTORSTORE_PATH, DOCSTORE_PATH,
                             EMBED_MODEL_NAME, TOP_K)

class Retriever:
    def __init__(self):
        self.docs = pickle.load(open(DOCSTORE_PATH, "rb"))
        self.index = faiss.read_index(VECTORSTORE_PATH)
        self.embedder = SentenceTransformer(EMBED_MODEL_NAME, device="cpu")

    def query(self, question: str):
        q_emb = self.embedder.encode([question])
        D, I = self.index.search(np.array(q_emb, dtype="float32"), TOP_K)
        return [self.docs[i] for i in I[0]], D[0]

if __name__ == "__main__":
    r = Retriever()
    docs, scores = r.query("What is the fee of Masters DS & Analytics?")
    print(list(zip([d.page_content[:50] for d in docs], scores)))
