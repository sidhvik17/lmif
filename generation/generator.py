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
        elif mod == "IMAGE":
            header = f"[{i}] (IMAGE — OCR/caption may contain errors: {src} | Page: {page})"
        else:
            header = f"[{i}] ({mod}: {src} | Page: {page})"

        numbered_parts.append(f"{header}\n{text}")

    context = "\n\n---\n\n".join(numbered_parts)

    prompt = f"""You are a helpful assistant that answers questions using ONLY the context below.

RULES:
- Cite every claim with [chunk_number], e.g. [1] or [1][3].
- IMAGE chunks were extracted via OCR and may have garbled text, wrong characters, or missing spaces. Interpret them as best you can — e.g. "SaTISH" means "SATISH", "Bomk" means "Bank", "W991" could be a misread of digits.
- AUDIO chunks were auto-transcribed and may have errors too.
- Always attempt an answer from the context. Only say "Not found in indexed sources." if the context is truly irrelevant.

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
