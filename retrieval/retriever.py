import hashlib
import re
from typing import Any, Dict, List, Optional, Tuple

from pipeline.embedder import embed_query
from vectorstore.store import search, get_count
from retrieval.reranker import rerank
from retrieval.bm25_index import bm25_search
from retrieval.hyde import hyde_embed
from config import (
    TOP_K, BROAD_TOP_K_MULTIPLIER, BROAD_FETCH_K, NORMAL_FETCH_K,
)
from logging_config import get_logger

log = get_logger(__name__)

ChunkResult = Tuple[str, Dict[str, Any]]

# Metadata fields supported as Chroma-side pre-filters (indexed, cheap).
# Everything else is applied in Python as post-filter.
_PRE_FILTER_FIELDS = {"modality", "file_ext", "source", "file_name"}


def _split_filters(
    filters: Optional[Dict[str, Any]],
) -> Tuple[Optional[Dict[str, Any]], Dict[str, Any]]:
    """Split user filters into (pre_filter, post_filter).

    Pre-filter = applied at ANN search time via Chroma `where` (fast).
    Post-filter = applied in Python after retrieval (flexible, e.g. callables,
    ranges on non-indexed fields).
    """
    if not filters:
        return None, {}
    pre: Dict[str, Any] = {}
    post: Dict[str, Any] = {}
    for k, v in filters.items():
        if v is None:
            continue
        if k in _PRE_FILTER_FIELDS or isinstance(v, dict):
            pre[k] = v
        else:
            post[k] = v
    return (pre or None), post


_OPERATOR_MATCHERS = {
    "$eq":  lambda got, v: got == v,
    "$ne":  lambda got, v: got != v,
    "$gt":  lambda got, v: got is not None and got > v,
    "$gte": lambda got, v: got is not None and got >= v,
    "$lt":  lambda got, v: got is not None and got < v,
    "$lte": lambda got, v: got is not None and got <= v,
    "$in":  lambda got, v: got in v,
    "$nin": lambda got, v: got not in v,
}


def _meta_matches(meta: Dict[str, Any], filters: Dict[str, Any]) -> bool:
    """Evaluate a filter dict against metadata. Supports scalars, callables,
    and Chroma-style operator dicts ($eq/$ne/$gt/$gte/$lt/$lte/$in/$nin).
    """
    for k, v in filters.items():
        if v is None:
            continue
        got = meta.get(k)
        if callable(v):
            if not v(got):
                return False
        elif isinstance(v, dict):
            for op, val in v.items():
                matcher = _OPERATOR_MATCHERS.get(op)
                if matcher is None or not matcher(got, val):
                    return False
        else:
            if got != v:
                return False
    return True


def _apply_filter(
    results: List[ChunkResult], filters: Dict[str, Any]
) -> List[ChunkResult]:
    if not filters:
        return results
    return [(t, m) for (t, m) in results if _meta_matches(m, filters)]


def _merge_results(
    dense_results: List[ChunkResult],
    sparse_results: List[Tuple[str, Dict[str, Any], float]],
    dense_weight: float = 0.6,
) -> List[ChunkResult]:
    """Reciprocal rank fusion of dense + sparse results.

    Content-hash key for robust dedup across retrievers.
    """
    fused_scores: Dict[Any, float] = {}
    doc_data: Dict[Any, ChunkResult] = {}

    def _key(text: str, meta: Dict[str, Any]):
        text_hash = hashlib.md5(text.encode("utf-8", errors="replace")).hexdigest()
        return (text_hash, meta.get("source", ""), meta.get("page", ""))

    for rank, (text, meta) in enumerate(dense_results):
        k = _key(text, meta)
        fused_scores[k] = fused_scores.get(k, 0.0) + dense_weight / (rank + 1)
        doc_data[k] = (text, meta)

    sparse_weight = 1.0 - dense_weight
    for rank, (text, meta, _score) in enumerate(sparse_results):
        k = _key(text, meta)
        fused_scores[k] = fused_scores.get(k, 0.0) + sparse_weight / (rank + 1)
        doc_data.setdefault(k, (text, meta))

    ranked = sorted(fused_scores.keys(), key=lambda k: fused_scores[k], reverse=True)
    return [doc_data[k] for k in ranked]


def _is_broad_query(query: str) -> bool:
    """Detect queries that need ALL content, not just top-k."""
    q = query.lower().strip()
    broad_patterns = [
        r"\b(summarize|summary|overview|all|everything|every)\b",
        r"\b(what (do|did) you (have|know))\b",
        r"\b(list|show|tell me) (all|everything)\b",
        r"\b(what (information|data|content|documents?))\b.*\b(have|indexed|stored|available)\b",
    ]
    return any(re.search(p, q) for p in broad_patterns)


def _validate_query(query: str) -> str:
    """Strip, reject empty, cap at 2000 chars (prevents pathological input)."""
    if not isinstance(query, str):
        raise TypeError("Query must be a string.")
    q = query.strip()
    if not q:
        raise ValueError("Query is empty.")
    if len(q) > 2000:
        log.warning("Query truncated from %d to 2000 chars.", len(q))
        q = q[:2000]
    return q


def retrieve(
    query: str,
    modality_filter: Optional[str] = None,
    use_hyde: bool = False,
    filters: Optional[Dict[str, Any]] = None,
) -> List[ChunkResult]:
    """Hybrid retrieval: dense + BM25 sparse, merged via RRF, then cross-encoder rerank.

    Staged filtering (production pattern):
      Stage 1 — Pre-filter on indexed metadata (modality/file_ext/source) via Chroma.
      Stage 2 — ANN vector search + BM25 sparse on pre-filtered set.
      Stage 3 — Post-filter on free-form metadata (e.g. word_count range) in Python.
      Stage 4 — Cross-encoder rerank for precision.

    Args:
        query: search string.
        modality_filter: shortcut for filters={"modality": ...}.
        use_hyde: enable HyDE query expansion for improved recall on indirect queries.
        filters: metadata constraints. Values may be scalars (equality),
                 Chroma operator dicts ({"$gte": 50}), or callables (post-filter only).
    """
    query = _validate_query(query)

    if get_count() == 0:
        log.warning("Collection empty — no documents indexed.")
        return []

    # Combine shortcut and generic filters
    combined: Dict[str, Any] = {}
    if modality_filter:
        combined["modality"] = modality_filter
    if filters:
        combined.update(filters)

    pre, post = _split_filters(combined or None)

    broad = _is_broad_query(query)
    fetch_k = BROAD_FETCH_K if broad else NORMAL_FETCH_K
    final_top_k = TOP_K * BROAD_TOP_K_MULTIPLIER if broad else TOP_K

    qvec = hyde_embed(query) if use_hyde else embed_query(query)

    raw_dense = search(qvec, k=fetch_k, filters=pre)

    raw_sparse = bm25_search(query, k=fetch_k)
    # BM25 results bypass Chroma's `where`, so re-validate pre-filters
    # here to keep operator filters (e.g. $gte) correct after RRF merge.
    if pre:
        raw_sparse = [(t, m, s) for t, m, s in raw_sparse if _meta_matches(m, pre)]

    merged = _merge_results(raw_dense, raw_sparse)

    # Post-filter covers BOTH: any free-form fields AND dict-op fields that
    # BM25 may have leaked in. Cheap re-validation on small merged set.
    if post or pre:
        before = len(merged)
        full_filters = {**(pre or {}), **post}
        merged = _apply_filter(merged, full_filters)
        if len(merged) != before:
            log.info("Filter validation: %d -> %d chunks", before, len(merged))

    if not merged:
        log.warning("No chunks survived filtering.")
        return []

    reranked = rerank(query, merged[:40], top_k=final_top_k)
    return reranked
