import uuid
import numpy as np
import chromadb
from config import CHROMA_DB_PATH, COLLECTION_NAME

client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
collection = client.get_or_create_collection(COLLECTION_NAME)


def _normalize_embeddings(vectors):
    arr = np.asarray(vectors, dtype=np.float32)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    return arr.tolist()


def add_chunks(chunks, vectors):
    """Upsert chunks with their embeddings and metadata into ChromaDB."""
    if not chunks:
        print("[DB] No chunks to store (nothing to index).")
        return
    ids = [str(uuid.uuid4()) for _ in chunks]
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
    print(f"[DB] Stored {len(chunks)} chunks.")


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


def search(query_vector, k=5):
    """Search ChromaDB for top-k most similar chunks to a query vector."""
    q = np.asarray(query_vector, dtype=np.float32)
    if q.ndim == 1:
        q = q.reshape(1, -1)
    n = collection.count()
    if n == 0:
        print("[SEARCH] Collection is empty — no documents have been indexed yet.")
        return []
    print(f"[SEARCH] Searching {n} indexed chunks (k={k})...")
    k = min(k, max(n, 1))
    results = collection.query(
        query_embeddings=q.tolist(),
        n_results=k,
        include=["documents", "metadatas"],
    )
    docs = results.get("documents") or [[]]
    metas = results.get("metadatas") or [[]]
    if not docs[0]:
        print(f"[SEARCH] Query returned no documents (collection has {n} items).")
        return []
    return list(zip(docs[0], metas[0]))
