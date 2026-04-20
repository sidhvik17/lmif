"""HyDE (Hypothetical Document Embeddings) for query expansion.

Generates a hypothetical answer via the LLM, then embeds it alongside
the original query for improved retrieval. Inspired by both LUMA-RAG's
multi-signal retrieval and MMORE's hybrid approach.
"""

import numpy as np
import ollama
from config import (
    LLM_MODEL, OLLAMA_BASE_URL,
    LLM_NUM_CTX, HYDE_TEMPERATURE, HYDE_NUM_PREDICT,
)
from pipeline.embedder import embed_query


def generate_hypothetical_answer(query):
    """Ask the LLM to generate a short hypothetical passage answering the query.

    The prompt is designed to produce text that resembles a real document passage,
    which helps the embedding align with actual indexed content.
    """
    prompt = (
        f"Write a short, factual paragraph (3-4 sentences) that directly answers "
        f"this question as if you are reading from a document, audio transcript, "
        f"or image description. Do not hedge or say 'I don't know'. Just write "
        f"the answer as if it were extracted from a real source:\n\n{query}"
    )
    try:
        client = ollama.Client(host=OLLAMA_BASE_URL)
        response = client.chat(
            model=LLM_MODEL,
            messages=[{"role": "user", "content": prompt}],
            options={
                "temperature": HYDE_TEMPERATURE,
                "num_predict": HYDE_NUM_PREDICT,
                "num_ctx": LLM_NUM_CTX,
            },
        )
        msg = response["message"]
        return msg["content"] if isinstance(msg, dict) else msg.content
    except Exception as e:
        print(f"[HyDE] LLM unavailable ({e}), falling back to original query.")
        return ""


def hyde_embed(query):
    """Return an embedding that averages the query vector with a hypothetical answer vector.

    If LLM is unavailable, falls back to standard query embedding.
    Uses weighted average: 40% query + 60% hypothetical, since the hypothetical
    answer is closer in form to the indexed documents.
    """
    query_vec = embed_query(query)

    hypo_answer = generate_hypothetical_answer(query)
    if not hypo_answer.strip():
        return query_vec

    hypo_vec = embed_query(hypo_answer)

    # Weighted average: hypothetical doc gets more weight since it resembles
    # stored passages more closely than the short query
    combined = 0.4 * np.array(query_vec, dtype=np.float32) + 0.6 * np.array(hypo_vec, dtype=np.float32)
    # L2-normalize for consistent cosine similarity
    norm = np.linalg.norm(combined)
    if norm > 0:
        combined = combined / norm
    return combined.tolist()
