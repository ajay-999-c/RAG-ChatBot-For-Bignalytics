# embedding.py

from typing import List
from langchain_community.embeddings import OllamaEmbeddings
from langchain_core.pydantic_v1 import Field
import math

class BatchingOllamaEmbeddings(OllamaEmbeddings):
    batch_size: int = Field(default=10, description="Batch size for embeddings.")

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Batch embed documents for efficiency."""
        all_embeddings = []
        total = len(texts)
        num_batches = math.ceil(total / self.batch_size)

        print(f"⚡ Total {total} chunks to embed in {num_batches} batches of {self.batch_size} each.")

        for batch_idx in range(num_batches):
            batch_start = batch_idx * self.batch_size
            batch_end = min((batch_idx + 1) * self.batch_size, total)
            batch_texts = texts[batch_start:batch_end]

            batch_embeddings = super().embed_documents(batch_texts)

            all_embeddings.extend(batch_embeddings)
            print(f"✅ Embedded batch {batch_idx+1}/{num_batches} [{batch_start}-{batch_end}]")

        return all_embeddings

def get_embedding_model():
    return BatchingOllamaEmbeddings(
        model="nomic-embed-text",  # match your Ollama pulled model
        base_url="http://localhost:11434",
        batch_size=6  # Safe value, you can tune
    )
