from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch, torch.nn.functional as F
from config.settings import RERANK_MODEL_NAME

class Reranker:
    def __init__(self):
        tok = AutoTokenizer.from_pretrained(RERANK_MODEL_NAME)
        model = AutoModelForSequenceClassification.from_pretrained(RERANK_MODEL_NAME)
        self.tokenizer, self.model = tok, model.eval()

    def score(self, query: str, docs):
        pairs = [(query, d.page_content) for d in docs]
        toks = self.tokenizer.batch_encode_plus(
            pairs, truncation=True, padding=True, return_tensors="pt"
        )
        with torch.no_grad():
            logits = self.model(**toks).logits.squeeze(-1)
        probs = F.softmax(logits, dim=0)
        scored = sorted(zip(docs, probs.tolist()), key=lambda x: x[1], reverse=True)
        return [d for d, _ in scored]
