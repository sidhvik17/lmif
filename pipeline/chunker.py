from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import util
from config import (
    CHUNK_SIZE, CHUNK_OVERLAP, CHUNK_MIN_CHARS,
    IMAGE_MIN_CHARS, AUDIO_MIN_CHARS, DEDUP_THRESHOLD,
)
import torch

splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
)

# Modality prefixes help the embedding model distinguish content types,
# improving retrieval for queries like "what was said" (audio) vs "what does the document say" (text).
_MODALITY_PREFIX = {
    "audio": "Audio transcript of spoken content: ",
    "image": "Text extracted from image: ",
}

_MIN_CHARS_BY_MODALITY = {
    "text": CHUNK_MIN_CHARS,
    "image": IMAGE_MIN_CHARS,
    "audio": AUDIO_MIN_CHARS,
}


def chunk_documents(raw_chunks):
    """Split raw extracted text chunks into smaller overlapping chunks.

    Adds modality-context prefixes to audio/image chunks so their embeddings
    carry semantic signal about the content type. This helps indirect queries
    like 'what did the speaker say?' retrieve audio chunks even when the
    transcript text alone doesn't match the query.
    """
    final = []
    for item in raw_chunks:
        text = (item.get("text") or "").strip()
        if not text:
            continue

        modality = item.get("metadata", {}).get("modality", "text")
        prefix = _MODALITY_PREFIX.get(modality, "")
        min_chars = _MIN_CHARS_BY_MODALITY.get(modality, CHUNK_MIN_CHARS)

        splits = splitter.split_text(item["text"])
        for idx, s in enumerate(splits):
            s = (s or "").strip()
            if s and len(s) >= min_chars:
                enriched = f"{prefix}{s}" if prefix else s
                meta = {**item["metadata"], "chunk_index": idx}
                final.append({"text": enriched, "metadata": meta})
    return final


def deduplicate_chunks(chunks, embeddings, threshold=DEDUP_THRESHOLD):
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
