from sentence_transformers import CrossEncoder
from config import (
    RERANKER_MODEL, RERANKER_MIN_SCORE,
    USE_MMR_RERANK, MMR_LAMBDA,
)

_reranker = None


def get_reranker():
    global _reranker
    if _reranker is None:
        _reranker = CrossEncoder(RERANKER_MODEL)
    return _reranker


def _mmr_select(chunks, scores, top_k, lambda_):
    """Greedy Maximal Marginal Relevance.

    MMR = λ · rel(d) − (1−λ) · max_{d' ∈ selected} sim(d, d')

    Reduces near-duplicates and promotes modality/source diversity.
    Uses BGE dense embeddings (cheap since model already loaded) for
    pairwise similarity.
    """
    if len(chunks) <= 1:
        return list(range(len(chunks)))

    # Normalize relevance scores to [0,1] so they compose with cosine sim.
    import numpy as np
    s = np.asarray(scores, dtype=np.float32)
    s_min, s_max = float(s.min()), float(s.max())
    s_range = s_max - s_min
    rel = (s - s_min) / s_range if s_range > 1e-6 else np.zeros_like(s)

    # Pairwise cosine similarity via shared embedder (lazy, normalized vectors).
    from pipeline.embedder import _get_model
    model = _get_model()
    texts = [t for t, _ in chunks]
    embs = model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
    embs = np.asarray(embs, dtype=np.float32)
    sim = embs @ embs.T   # n x n cosine (embs already L2-normed)

    n = len(chunks)
    selected: list[int] = []
    remaining = set(range(n))

    # Seed with highest-relevance chunk
    first = int(np.argmax(rel))
    selected.append(first)
    remaining.discard(first)

    while remaining and len(selected) < top_k:
        best_idx, best_val = -1, -float("inf")
        for i in remaining:
            max_sim_to_sel = max(float(sim[i, j]) for j in selected)
            mmr = lambda_ * float(rel[i]) - (1.0 - lambda_) * max_sim_to_sel
            if mmr > best_val:
                best_val, best_idx = mmr, i
        selected.append(best_idx)
        remaining.discard(best_idx)

    return selected


def rerank(
    query: str,
    chunks: list,
    top_k: int = 5,
    use_mmr: bool | None = None,
    mmr_lambda: float | None = None,
) -> list:
    """Rerank retrieved chunks using a cross-encoder for better precision.

    chunks: list of (text, metadata) tuples.

    If `use_mmr` is enabled, final selection uses Maximal Marginal Relevance
    over the cross-encoder scores — promotes diversity for broad queries.
    """
    if not chunks:
        return []

    reranker = get_reranker()
    pairs = [(query, text) for text, _ in chunks]
    scores = reranker.predict(pairs)

    # Apply permissive min-score filter. Guarantee enough chunks to honor top_k
    # even when cross-encoder scores are low (common for broad/generic queries
    # where CE can't bind to a specific answer).
    min_keep = min(max(top_k, 3), len(chunks))
    scored = list(zip(scores, chunks))
    filtered = [(s, c) for s, c in scored if s > RERANKER_MIN_SCORE]
    if len(filtered) < min_keep:
        filtered = sorted(scored, key=lambda x: x[0], reverse=True)[:min_keep]

    # Sort by relevance descending; keep up to 2x top_k as MMR candidate pool
    filtered.sort(key=lambda x: x[0], reverse=True)
    pool_size = min(len(filtered), max(top_k * 2, top_k + 3))
    pool = filtered[:pool_size]

    pool_chunks = [c for _, c in pool]
    pool_scores = [s for s, _ in pool]

    mmr_on = USE_MMR_RERANK if use_mmr is None else use_mmr
    if mmr_on and len(pool_chunks) > top_k:
        lam = MMR_LAMBDA if mmr_lambda is None else mmr_lambda
        idxs = _mmr_select(pool_chunks, pool_scores, top_k, lam)
        return [pool_chunks[i] for i in idxs]

    return pool_chunks[:top_k]
