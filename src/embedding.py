from sentence_transformers import SentenceTransformer

def load_embedding_model(model_name="all-MiniLM-L6-v2"):
    return SentenceTransformer(model_name)

def create_embeddings(model, texts):
    return model.encode(texts, show_progress_bar=True)
