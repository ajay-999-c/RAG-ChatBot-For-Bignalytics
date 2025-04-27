from langchain_community.document_loaders import UnstructuredPDFLoader, CSVLoader

def load_pdf(file_path: str):
    loader = UnstructuredPDFLoader(file_path)
    docs = loader.load()
    for doc in docs:
        doc.metadata["source_file"] = file_path
        doc.metadata["document_type"] = "pdf"
    return docs

def load_csv(file_path: str):
    loader = CSVLoader(file_path=file_path)
    docs = loader.load()
    for doc in docs:
        doc.metadata["source_file"] = file_path
        doc.metadata["document_type"] = "csv"
    return docs

def load_all_data(pdf_paths: list, csv_paths: list):
    all_docs = []
    for pdf in pdf_paths:
        all_docs.extend(load_pdf(pdf))
    for csv in csv_paths:
        all_docs.extend(load_csv(csv))
    return all_docs
