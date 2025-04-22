from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import torch
from config.settings import GEN_MODEL_NAME

class Generator:
    def __init__(self):
        self.tok = AutoTokenizer.from_pretrained(GEN_MODEL_NAME)
        self.lm = AutoModelForCausalLM.from_pretrained(
            GEN_MODEL_NAME, device_map="auto", torch_dtype=torch.float16
        )
        self.ppl = pipeline(
            "text-generation", model=self.lm, tokenizer=self.tok,
            max_new_tokens=256, temperature=0, do_sample=False
        )

    def answer(self, question: str, context_docs):
        context = "\n\n---\n\n".join([d.page_content for d in context_docs])
        prompt = (
            f"Answer the question using only the context. "
            f"Add a 'SOURCES:' section listing chunk indices.\n\n"
            f"CONTEXT:\n{context}\n\nQUESTION:\n{question}\n\nANSWER:\n"
        )
        return self.ppl(prompt)[0]["generated_text"][len(prompt):]
