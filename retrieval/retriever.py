from pipeline.embedder import model as embed_model
from vectorstore.store import search
from config import TOP_K


def retrieve(query):
    """Embed query and retrieve top-k relevant chunks from vector store."""
    qvec = embed_model.encode(query)
    return search(qvec, k=TOP_K)
