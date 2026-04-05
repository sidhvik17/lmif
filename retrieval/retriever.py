from pipeline.embedder import embed_query
from vectorstore.store import search
from retrieval.reranker import rerank
from retrieval.bm25_index import bm25_search
from retrieval.hyde import hyde_embed
from config import TOP_K


def _merge_results(dense_results, sparse_results, dense_weight=0.6):
    """Merge dense and sparse results using weighted reciprocal rank fusion.

    Inspired by MMORE's hybrid dense+sparse retrieval strategy.
    """
    # Two parallel dicts: one for scores, one for doc data
    fused_scores = {}
    doc_data = {}

    def _key(text, meta):
        return (text[:200], meta.get("source", ""), meta.get("page", ""))

    # Score dense results by reciprocal rank
    for rank, (text, meta) in enumerate(dense_results):
        k = _key(text, meta)
        fused_scores[k] = fused_scores.get(k, 0.0) + dense_weight / (rank + 1)
        doc_data[k] = (text, meta)

    # Score sparse results by reciprocal rank
    sparse_weight = 1.0 - dense_weight
    for rank, (text, meta, _bm25_score) in enumerate(sparse_results):
        k = _key(text, meta)
        fused_scores[k] = fused_scores.get(k, 0.0) + sparse_weight / (rank + 1)
        if k not in doc_data:
            doc_data[k] = (text, meta)

    # Sort by fused score descending
    ranked = sorted(fused_scores.keys(), key=lambda k: fused_scores[k], reverse=True)
    return [doc_data[k] for k in ranked]


def retrieve(query, modality_filter=None, use_hyde=False):
    """Hybrid retrieval: dense + BM25 sparse, then cross-encoder rerank.

    Args:
        query: The search query string.
        modality_filter: Optional filter - "text", "image", "audio", or None for all.
        use_hyde: If True, use HyDE (Hypothetical Document Embeddings) for better recall.
    """
    qvec = hyde_embed(query) if use_hyde else embed_query(query)

    # Dense retrieval: cast wide net of 20 candidates
    raw_dense = search(qvec, k=20, modality_filter=modality_filter)

    # Sparse BM25 retrieval: 20 candidates
    raw_sparse = bm25_search(query, k=20)
    # Apply modality filter to sparse results if needed
    if modality_filter:
        raw_sparse = [
            (t, m, s) for t, m, s in raw_sparse
            if m.get("modality") == modality_filter
        ]

    # Merge with reciprocal rank fusion
    merged = _merge_results(raw_dense, raw_sparse)

    # Rerank with cross-encoder for precision
    reranked = rerank(query, merged[:30], top_k=TOP_K)
    return reranked
