"""BM25 sparse index for keyword-based retrieval (inspired by MMORE hybrid search)."""

import re
from rank_bm25 import BM25Okapi
from vectorstore.store import _get_collection
from logging_config import get_logger

log = get_logger(__name__)


_bm25 = None
_bm25_docs = None
_bm25_count = None

# Common English stopwords that dilute BM25 scores for indirect queries
_STOPWORDS = frozenset({
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "shall", "can", "to", "of", "in", "for",
    "on", "with", "at", "by", "from", "as", "into", "through", "during",
    "before", "after", "above", "below", "between", "out", "off", "over",
    "under", "again", "further", "then", "once", "here", "there", "when",
    "where", "why", "how", "all", "each", "every", "both", "few", "more",
    "most", "other", "some", "such", "no", "nor", "not", "only", "own",
    "same", "so", "than", "too", "very", "just", "about", "and", "but",
    "or", "if", "while", "that", "this", "it", "its", "i", "me", "my",
    "we", "our", "you", "your", "he", "him", "his", "she", "her", "they",
    "them", "their", "what", "which", "who", "whom",
})


def _tokenize(text):
    """Tokenize with stopword removal for better BM25 precision."""
    tokens = re.findall(r"\w+", text.lower())
    return [t for t in tokens if t not in _STOPWORDS]


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
    log.info("Built sparse index over %d documents.", n)


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
