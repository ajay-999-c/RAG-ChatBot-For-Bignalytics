from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

def load_query_transformer(model_name="microsoft/phi-3-mini-128k-instruct"):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    nlp_pipeline = pipeline("text-generation", model=model, tokenizer=tokenizer, max_length=128)
    return nlp_pipeline

def transform_query(pipeline, query):
    prompt = f"Rewrite the following question for better search: {query}"
    result = pipeline(prompt)
    return result[0]['generated_text']
