import hashlib
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import chromadb

from config import CHROMA_DB_PATH, COLLECTION_NAME
from logging_config import get_logger

log = get_logger(__name__)
client = chromadb.PersistentClient(path=CHROMA_DB_PATH)


def _get_collection():
    """Always fetch a fresh collection reference to avoid stale reads.

    Uses cosine distance because BGE embeddings are normalized and cosine
    similarity is the standard measure for retrieval tasks.
    """
    return client.get_or_create_collection(
        COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )


def _normalize_embeddings(vectors) -> List[List[float]]:
    """L2-normalize embedding vectors before storage."""
    arr = np.asarray(vectors, dtype=np.float32)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    norms = np.where(norms > 0, norms, 1.0)
    return (arr / norms).tolist()


def _deterministic_id(text: str, source: str, page: Any) -> str:
    """Deterministic chunk ID from content + source + page.

    Re-ingesting the same file produces the same IDs, so upsert correctly
    deduplicates instead of creating duplicates.
    """
    key = f"{source}::{page}::{text[:500]}"
    return hashlib.sha256(key.encode("utf-8")).hexdigest()[:16]


def _sanitize_metadata(meta: dict) -> dict:
    """Chroma metadata values must be str, int, float, or bool."""
    out = {}
    for k, v in meta.items():
        if v is None:
            continue
        if isinstance(v, (str, int, float, bool)):
            out[k] = v
        else:
            out[k] = str(v)
    return out


def add_chunks(chunks: List[Dict[str, Any]], vectors) -> None:
    """Upsert chunks with embeddings and metadata into ChromaDB."""
    if not chunks:
        log.info("No chunks to store.")
        return
    collection = _get_collection()
    ids = [
        _deterministic_id(
            c["text"],
            c["metadata"].get("source", ""),
            str(c["metadata"].get("page", "")),
        )
        for c in chunks
    ]
    texts = [c["text"] for c in chunks]
    metas = [_sanitize_metadata(c["metadata"]) for c in chunks]
    embeddings = _normalize_embeddings(vectors)
    if len(embeddings) != len(chunks):
        raise ValueError("Embeddings count must match chunks count.")
    collection.upsert(
        ids=ids,
        documents=texts,
        embeddings=embeddings,
        metadatas=metas,
    )
    log.info("Stored %d chunks. Collection size: %d", len(chunks), collection.count())


def get_count() -> int:
    return _get_collection().count()


def clear_collection() -> None:
    try:
        client.delete_collection(COLLECTION_NAME)
        log.info("Collection '%s' deleted.", COLLECTION_NAME)
    except Exception:
        pass
    _get_collection()
    log.info("Collection '%s' recreated (empty).", COLLECTION_NAME)


def delete_by_source(source: str) -> int:
    """Remove all chunks originating from a given source path. Returns deleted count.

    Call before re-ingesting a file that has shrunk, to prevent orphan chunks
    (chunks whose deterministic IDs no longer match the new content).
    """
    collection = _get_collection()
    got = collection.get(where={"source": source}, include=[])
    ids = got.get("ids") or []
    if not ids:
        log.info("No chunks found for source: %s", source)
        return 0
    collection.delete(ids=ids)
    log.info("Deleted %d chunks for source: %s", len(ids), source)
    return len(ids)


def modality_breakdown() -> Dict[str, int]:
    """Count chunks grouped by modality. Useful for UI dashboards + CLI stats."""
    collection = _get_collection()
    got = collection.get(include=["metadatas"])
    metas = got.get("metadatas") or []
    counts: Dict[str, int] = {}
    for m in metas:
        mod = (m or {}).get("modality", "unknown")
        counts[mod] = counts.get(mod, 0) + 1
    return counts


def _build_where(filters: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """Build Chroma `where` clause from a filters dict.

    Single-key → pass through (Chroma supports simple {field: value}).
    Multi-key → wrap with $and per Chroma syntax.
    Values may be scalars (equality) or dicts with operators like {"$gte": 100}.
    """
    if not filters:
        return None
    clauses = []
    for k, v in filters.items():
        if v is None:
            continue
        clauses.append({k: v})
    if not clauses:
        return None
    if len(clauses) == 1:
        return clauses[0]
    return {"$and": clauses}


def search(
    query_vector,
    k: int = 5,
    filters: Optional[Dict[str, Any]] = None,
) -> List[Tuple[str, Dict[str, Any]]]:
    """Search top-k most similar chunks with optional pre-filter.

    Adaptive fallback: if filtered query fails or yields nothing, retries
    with smaller k, then drops filter as last resort so user always gets
    some context.
    """
    collection = _get_collection()
    q = np.asarray(query_vector, dtype=np.float32)
    if q.ndim == 1:
        q = q.reshape(1, -1)
    n = collection.count()
    if n == 0:
        log.warning("Collection empty — nothing indexed yet.")
        return []

    where = _build_where(filters)
    effective_k = min(k, n)
    log.info("Search: n=%d k=%d where=%s", n, effective_k, where)

    query_kwargs: Dict[str, Any] = {
        "query_embeddings": q.tolist(),
        "n_results": effective_k,
        "include": ["documents", "metadatas"],
    }
    if where:
        query_kwargs["where"] = where

    try:
        results = collection.query(**query_kwargs)
    except Exception as e:
        log.warning("Query failed (%s); retrying with reduced k.", e)
        try:
            query_kwargs["n_results"] = min(effective_k, 5)
            results = collection.query(**query_kwargs)
        except Exception as e2:
            log.warning("Retry failed (%s); dropping filter.", e2)
            query_kwargs.pop("where", None)
            query_kwargs["n_results"] = min(effective_k, 10)
            results = collection.query(**query_kwargs)

    docs = results.get("documents") or [[]]
    metas = results.get("metadatas") or [[]]
    if not docs[0]:
        log.warning("No docs returned (collection has %d items).", n)
        return []
    log.info("Retrieved %d results.", len(docs[0]))
    return list(zip(docs[0], metas[0]))
