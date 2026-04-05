"""HyDE (Hypothetical Document Embeddings) for query expansion.

Generates a hypothetical answer via the LLM, then embeds it alongside
the original query for improved retrieval. Inspired by both LUMA-RAG's
multi-signal retrieval and MMORE's hybrid approach.
"""

import numpy as np
import ollama
from config import LLM_MODEL, OLLAMA_BASE_URL
from pipeline.embedder import embed_query


def generate_hypothetical_answer(query):
    """Ask the LLM to generate a short hypothetical passage answering the query."""
    prompt = (
        f"Write a short, factual paragraph (3-4 sentences) that directly answers "
        f"this question. Do not hedge or say 'I don't know'. Just write the answer "
        f"as if it were from a document:\n\n{query}"
    )
    try:
        client = ollama.Client(host=OLLAMA_BASE_URL)
        response = client.chat(
            model=LLM_MODEL,
            messages=[{"role": "user", "content": prompt}],
        )
        msg = response["message"]
        return msg["content"] if isinstance(msg, dict) else msg.content
    except Exception as e:
        print(f"[HyDE] LLM unavailable ({e}), falling back to original query.")
        return ""


def hyde_embed(query):
    """Return an embedding that averages the query vector with a hypothetical answer vector.

    If LLM is unavailable, falls back to standard query embedding.
    """
    query_vec = embed_query(query)

    hypo_answer = generate_hypothetical_answer(query)
    if not hypo_answer.strip():
        return query_vec

    hypo_vec = embed_query(hypo_answer)

    # Average the two embeddings (query + hypothetical) for broader recall
    combined = (np.array(query_vec) + np.array(hypo_vec)) / 2.0
    # L2-normalize
    norm = np.linalg.norm(combined)
    if norm > 0:
        combined = combined / norm
    return combined
