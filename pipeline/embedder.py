import numpy as np
from sentence_transformers import SentenceTransformer
from config import EMBEDDING_MODEL

model = SentenceTransformer(EMBEDDING_MODEL)


def embed_chunks(chunks):
    """Generate dense vector embeddings for a list of text chunks."""
    if not chunks:
        dim = model.get_sentence_embedding_dimension()
        return np.zeros((0, dim), dtype=np.float32)
    texts = [c["text"] for c in chunks]
    vectors = model.encode(texts, show_progress_bar=len(texts) > 8)
    return vectors


def embed_query(query: str):
    """Embed a query with BGE instruction prefix for retrieval-aware encoding."""
    prefixed = f"Represent this sentence for searching relevant passages: {query}"
    return model.encode(prefixed)
