from langchain_community.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

_SPLITTER = RecursiveCharacterTextSplitter(
    chunk_size=400,
    chunk_overlap=60,
    separators=["\n\n", "\n", ".", "?"],
)

def chunk(docs: list[Document]) -> list[Document]:
    out = []
    for doc in docs:
        # Preserve entire FAQ row as one chunk
        if doc.metadata.get("type") == "faq":
            out.append(doc)
        else:
            out.extend(_SPLITTER.split_documents([doc]))
    return out
