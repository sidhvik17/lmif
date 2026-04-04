from pipeline.embedder import embed_query
from vectorstore.store import search
from retrieval.reranker import rerank
from config import TOP_K


def retrieve(query):
    """Embed query with BGE prefix, retrieve top-20 candidates, rerank to top-k."""
    qvec = embed_query(query)
    # Cast wide net: retrieve 20 candidates
    raw_results = search(qvec, k=20)
    # Rerank with cross-encoder for precision
    reranked = rerank(query, raw_results, top_k=TOP_K)
    return reranked
