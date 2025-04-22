from fastapi import FastAPI
from pydantic import BaseModel
from pipeline.rag import ask

app = FastAPI()

class Query(BaseModel):
    question: str

@app.post("/rag")
def rag_endpoint(q: Query):
    answer, docs = ask(q.question)
    return {"answer": answer, "sources": [d.metadata for d in docs]}
