import ollama
from config import LLM_MODEL, OLLAMA_BASE_URL


def generate(query, chunks):
    """Generate a grounded answer using Ollama LLM with numbered context and inline citations."""
    if not chunks:
        return "Not found in indexed sources.", []

    # Build numbered context with rich metadata
    numbered_parts = []
    for i, (text, meta) in enumerate(chunks, 1):
        mod = meta.get("modality", "text").upper()
        src = meta.get("source", "unknown")
        page = meta.get("page", "?")

        if mod == "AUDIO":
            header = f"[{i}] (AUDIO: {src} | Timestamp: {page})"
        else:
            header = f"[{i}] ({mod}: {src} | Page: {page})"

        numbered_parts.append(f"{header}\n{text}")

    context = "\n\n---\n\n".join(numbered_parts)

    prompt = f"""You are a precise, citation-aware assistant. Follow these rules strictly:

1. Answer ONLY using the numbered context chunks below.
2. For EVERY factual claim, cite the source using [chunk_number].
3. If multiple chunks support a claim, cite all: [1][3].
4. If the answer is not in any chunk, say "Not found in indexed sources."
5. Never add information beyond what the chunks contain.

Context:
{context}

Question: {query}

Answer (every claim must have a citation):"""

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
