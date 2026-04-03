from langchain_text_splitters import RecursiveCharacterTextSplitter
from config import CHUNK_SIZE, CHUNK_OVERLAP

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
        for s in splits:
            s = (s or "").strip()
            if s:
                final.append({"text": s, "metadata": item["metadata"]})
    return final
