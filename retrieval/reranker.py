from sentence_transformers import CrossEncoder

# Load once at module level (singleton)
_reranker = None


def get_reranker():
    global _reranker
    if _reranker is None:
        _reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    return _reranker


def rerank(query: str, chunks: list, top_k: int = 5) -> list:
    """
    Rerank retrieved chunks using a cross-encoder for much better precision.
    chunks: list of (text, metadata) tuples
    """
    if not chunks:
        return []

    reranker = get_reranker()
    pairs = [(query, text) for text, _ in chunks]
    scores = reranker.predict(pairs)

    # Sort by reranker score descending
    scored = sorted(zip(scores, chunks), key=lambda x: x[0], reverse=True)
    return [chunk for _, chunk in scored[:top_k]]
