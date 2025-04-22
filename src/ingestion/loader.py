from pathlib import Path
import pandas as pd
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_core.documents import Document

DATA_DIR = Path(__file__).parents[2] / "data"

def load_csv(csv_path: Path) -> list[Document]:
    df = pd.read_csv(csv_path)
    docs = []
    for _, row in df.iterrows():
        text = f"Q: {row['Question']}\nA: {row['Reply']}"
        meta = {"type": "faq", "row_id": int(_)}
        docs.append(Document(page_content=text, metadata=meta))
    return docs

def load_pdf(pdf_path: Path) -> list[Document]:
    loader = PyMuPDFLoader(str(pdf_path))
    raw_pages = loader.load()
    for p in raw_pages:
        p.metadata.update({"type": "brochure"})
    return raw_pages

def ingest() -> list[Document]:
    csv_docs = load_csv(DATA_DIR / "FINAL-FAQ-FIXED.csv")
    pdf_docs = load_pdf(DATA_DIR / "Dataset.pdf")
    return csv_docs + pdf_docs

if __name__ == "__main__":
    print(f"Ingested {len(ingest())} raw docs")
