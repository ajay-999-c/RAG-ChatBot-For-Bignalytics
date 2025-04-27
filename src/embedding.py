from langchain_community.embeddings import OllamaEmbeddings

def get_embedding_model():
    return OllamaEmbeddings(
        model="nomic-embed-text",  # FIXED model name!
        base_url="http://localhost:11434"
    )

