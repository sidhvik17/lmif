from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import util
from config import CHUNK_SIZE, CHUNK_OVERLAP
import torch

splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
)


def chunk_documents(raw_chunks):
    """Split raw extracted text chunks into smaller overlapping chunks."""
    final = []
    for item in raw_chunks:
        text = (item.get("text") or "").strip()
        if not text:
            continue
        splits = splitter.split_text(item["text"])
        for idx, s in enumerate(splits):
            s = (s or "").strip()
            if s:
                meta = {**item["metadata"], "chunk_index": idx}
                final.append({"text": s, "metadata": meta})
    return final


def deduplicate_chunks(chunks, embeddings, threshold=0.95):
    """
    Remove near-duplicate chunks based on embedding similarity.
    Keeps the chunk with richer metadata (prefer text > OCR > audio).
    """
    if len(chunks) <= 1:
        return chunks, embeddings

    MODALITY_PRIORITY = {"text": 3, "image": 2, "audio": 1}

    emb_tensor = torch.tensor(embeddings)
    sim_matrix = util.cos_sim(emb_tensor, emb_tensor)

    to_remove = set()
    for i in range(len(chunks)):
        if i in to_remove:
            continue
        for j in range(i + 1, len(chunks)):
            if j in to_remove:
                continue
            if sim_matrix[i][j] > threshold:
                # Keep the one with higher modality priority
                pri_i = MODALITY_PRIORITY.get(chunks[i]["metadata"].get("modality", "text"), 0)
                pri_j = MODALITY_PRIORITY.get(chunks[j]["metadata"].get("modality", "text"), 0)
                to_remove.add(j if pri_i >= pri_j else i)

    kept_indices = [i for i in range(len(chunks)) if i not in to_remove]
    kept_chunks = [chunks[i] for i in kept_indices]
    kept_embeddings = [embeddings[i] for i in kept_indices]
    return kept_chunks, kept_embeddings
