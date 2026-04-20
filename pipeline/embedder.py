from functools import lru_cache

import numpy as np
from sentence_transformers import SentenceTransformer

from config import EMBEDDING_MODEL

_model = None

_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

# Cache ceiling: 128 query strings covers a full interactive session with LRU
# eviction. Memory cost ≈ 128 * 384 * 4B ≈ 200KB. Bump only if you run long
# batch jobs where even that tail matters; unbounded caches are never worth it.
_QUERY_CACHE_SIZE = 128


def _get_model():
    """Lazy-load the embedding model. First call pays the ~400MB cost;
    later calls are free. Keeps `--help` / import-only tool paths fast.
    """
    global _model
    if _model is None:
        _model = SentenceTransformer(EMBEDDING_MODEL)
    return _model


def embed_chunks(chunks):
    """Generate dense vector embeddings for a list of text chunks."""
    model = _get_model()
    if not chunks:
        dim = model.get_sentence_embedding_dimension()
        return np.zeros((0, dim), dtype=np.float32)
    texts = [c["text"] for c in chunks]
    vectors = model.encode(
        texts,
        normalize_embeddings=True,
        show_progress_bar=len(texts) > 8,
    )
    return vectors


@lru_cache(maxsize=_QUERY_CACHE_SIZE)
def _embed_query_cached(query: str) -> tuple:
    """Cached query encoder. Returns an immutable tuple so downstream mutation
    of the returned array cannot corrupt the cached value — np.ndarray is mutable
    and lru_cache stores by reference, so caching arrays directly is a latent bug.
    """
    model = _get_model()
    vec = model.encode(f"{_QUERY_PREFIX}{query}", normalize_embeddings=True)
    return tuple(vec.tolist())


def embed_query(query: str):
    """Embed query with BGE instruction prefix. Returns a fresh L2-normalized np.ndarray."""
    if not isinstance(query, str) or not query.strip():
        raise ValueError("Cannot embed empty query.")
    cached = _embed_query_cached(query.strip())
    return np.array(cached, dtype=np.float32)
