"""BM25 sparse index for keyword-based retrieval (inspired by MMORE hybrid search)."""

import re
from rank_bm25 import BM25Okapi
from vectorstore.store import _get_collection


_bm25 = None
_bm25_docs = None
_bm25_count = None


def _tokenize(text):
    """Simple whitespace + punctuation tokenizer."""
    return re.findall(r"\w+", text.lower())


def _build_index():
    """Build/rebuild BM25 index from all documents in ChromaDB."""
    global _bm25, _bm25_docs, _bm25_count
    collection = _get_collection()
    n = collection.count()
    if n == 0:
        _bm25 = None
        _bm25_docs = []
        _bm25_count = 0
        return

    # Fetch all documents and metadata from ChromaDB
    all_data = collection.get(include=["documents", "metadatas"])
    docs = all_data.get("documents") or []
    metas = all_data.get("metadatas") or []

    _bm25_docs = list(zip(docs, metas))
    tokenized = [_tokenize(d) for d in docs]
    _bm25 = BM25Okapi(tokenized)
    _bm25_count = n
    print(f"[BM25] Built sparse index over {n} documents.")


def bm25_search(query, k=10):
    """Search using BM25 sparse retrieval. Returns list of (text, metadata, score)."""
    global _bm25, _bm25_docs, _bm25_count
    collection = _get_collection()
    current_count = collection.count()

    # Rebuild if collection changed or first time
    if _bm25 is None or _bm25_count != current_count:
        _build_index()

    if not _bm25 or not _bm25_docs:
        return []

    tokens = _tokenize(query)
    scores = _bm25.get_scores(tokens)

    # Get top-k indices
    ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
    results = []
    for idx in ranked:
        if scores[idx] > 0:
            doc, meta = _bm25_docs[idx]
            results.append((doc, meta, float(scores[idx])))
    return results
