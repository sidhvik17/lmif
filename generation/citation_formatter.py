import os


def format_citations(chunks):
    """Format source citations from retrieved chunk metadata."""
    citations = []
    for _, meta in chunks:
        mod = meta.get("modality", "text")
        src = os.path.basename(str(meta.get("source", "unknown")))
        page = meta.get("page", "")
        if mod == "audio":
            citations.append(f"[AUDIO] Source: {src}, Timestamp: {page}")
        else:
            citations.append(f"[{mod.upper()}] Source: {src}, Page: {page}")
    return list(dict.fromkeys(citations))  # deduplicate
