import ollama
from config import LLM_MODEL, OLLAMA_BASE_URL


def generate(query, chunks):
    """Generate a grounded answer using Ollama LLM with retrieved context."""
    if not chunks:
        return "Answer not found in indexed sources.", []

    context = "\n\n".join(
        [f"[{m['modality'].upper()}] {t}" for t, m in chunks]
    )
    prompt = f"""You are a helpful assistant. Answer ONLY using the context below.
If the answer is not in the context, say "Not found in indexed sources."

Context:
{context}

Question: {query}
Answer:"""

    try:
        client = ollama.Client(host=OLLAMA_BASE_URL)
        response = client.chat(
            model=LLM_MODEL,
            messages=[{"role": "user", "content": prompt}],
        )
    except Exception as e:
        return (
            f"Could not reach Ollama at {OLLAMA_BASE_URL} ({e}). "
            f"Start Ollama and ensure model `{LLM_MODEL}` is pulled.",
            chunks,
        )
    msg = response["message"]
    content = msg["content"] if isinstance(msg, dict) else msg.content
    return content, chunks
