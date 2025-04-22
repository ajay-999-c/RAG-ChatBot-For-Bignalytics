from sentence_transformers import SentenceTransformer
import faiss, pickle, os
from pathlib import Path
from config.settings import (EMBED_MODEL_NAME,
                             VECTORSTORE_PATH, DOCSTORE_PATH)
from ingestion.loader import ingest
from ingestion.chunker import chunk

def build_index():
    docs = chunk(ingest())
    model = SentenceTransformer(EMBED_MODEL_NAME, device="cpu")
    vecs = model.encode([d.page_content for d in docs], show_progress_bar=True)

    index = faiss.IndexFlatL2(vecs.shape[1])
    index.add(vecs)

    Path(os.path.dirname(VECTORSTORE_PATH)).mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, VECTORSTORE_PATH)

    with open(DOCSTORE_PATH, "wb") as f:
        pickle.dump(docs, f)

    print(f"Indexed {len(docs)} chunks → {VECTORSTORE_PATH}")

if __name__ == "__main__":
    build_index()
