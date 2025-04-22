from vectorstore.retriever import Retriever
from vectorstore.rerank import Reranker
from generator.llm import Generator

from generator.groq_llm import answer_with_groq

from typing import Tuple

ret = Retriever()
rerank = Reranker()
# gen = Generator()


def ask(question: str):
    rough_docs, _ = ret.query(question)
    docs = rerank.score(question, rough_docs)[:3]
    answer = answer_with_groq(question, docs)
    return answer, docs
# def ask(question: str) -> Tuple[str, list]:
#     rough_docs, _ = ret.query(question)
#     docs = rerank.score(question, rough_docs)[:3]  # take top‑3
#     answer = gen.answer(question, docs)
#     return answer, docs
