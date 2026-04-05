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
            header = f"[{i}] (AUDIO TRANSCRIPT from: {src} | Timestamp: {page})"
        elif mod == "IMAGE":
            header = f"[{i}] (IMAGE OCR/caption — may contain OCR errors: {src} | Page: {page})"
        else:
            header = f"[{i}] (TEXT from: {src} | Page: {page})"

        numbered_parts.append(f"{header}\nContent: {text}")

    context = "\n\n---\n\n".join(numbered_parts)

    prompt = f"""You are a helpful assistant that answers questions using ONLY the context below.

RULES:
- Cite every claim with [chunk_number], e.g. [1] or [1][3].
- Each chunk has a "Content:" line — that IS the actual extracted text. Read it carefully.
- AUDIO TRANSCRIPT chunks contain speech-to-text transcriptions. The text after "Content:" is what was spoken.
- IMAGE OCR chunks were extracted via OCR and may have garbled text. Interpret as best you can — e.g. "SaTISH" = "SATISH", "Bomk" = "Bank".
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
