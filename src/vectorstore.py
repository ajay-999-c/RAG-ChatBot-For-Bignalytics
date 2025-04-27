
import time
from langchain_community.vectorstores import FAISS
from embedding import get_embedding_model
from semantic_chunker import chunk_documents_with_metadata

def create_vectorstore_from_documents(documents, save_path: str):
    total_start = time.time()

    print("✅ Step 1: Starting semantic chunking...")
    start = time.time()
    chunked_documents = chunk_documents_with_metadata(documents)
    end = time.time()
    print(f"✅ Step 1 Complete: {len(chunked_documents)} chunks created. ⏱️ {end-start:.2f} seconds")

    print("✅ Step 2: Loading embedding model...")
    start = time.time()
    embedding_function = get_embedding_model()
    end = time.time()
    print(f"✅ Step 2 Complete: Embedding model loaded. ⏱️ {end-start:.2f} seconds")

    print("✅ Step 3: Creating FAISS vectorstore from chunks...")
    start = time.time()
    vectorstore = FAISS.from_documents(chunked_documents, embedding_function)
    end = time.time()
    print(f"✅ Step 3 Complete: Vectorstore created. ⏱️ {end-start:.2f} seconds")

    print(f"✅ Step 4: Saving vectorstore to {save_path}...")
    start = time.time()
    vectorstore.save_local(save_path)
    end = time.time()
    print(f"✅ Step 4 Complete: Vectorstore saved successfully! ⏱️ {end-start:.2f} seconds")

    total_end = time.time()
    print(f"🎯 Total Time Taken: {total_end-total_start:.2f} seconds")

    return vectorstore

def load_vectorstore(save_path: str):
    embedding_function = get_embedding_model()
    return FAISS.load_local(save_path, embedding_function, allow_dangerous_deserialization=True)
