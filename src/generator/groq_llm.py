# generator/groq_llm.py
import os
import openai

openai.api_key = os.getenv("GROQ_API_KEY")
openai.api_base = "https://api.groq.com/openai/v1"


def answer_with_groq(question, context_docs):
    context = "\n\n".join([d.page_content for d in context_docs])
    prompt = (
        f"Answer the question using only the context below.\n"
        f"Context:\n{context}\n\n"
        f"Question: {question}\nAnswer:"
    )

    response = openai.ChatCompletion.create(
        model="mixtral-8x7b-instruct",  # or llama3-70b-8192
        messages=[
            {"role": "system", "content": "You are a helpful assistant for Bignalytics queries."},
            {"role": "user", "content": prompt},
        ],
        temperature=0,
        max_tokens=512,
    )

    return response["choices"][0]["message"]["content"]
