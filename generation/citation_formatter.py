import re


def format_citations(answer: str, chunks: list) -> str:
    """
    Post-process the LLM answer to convert [chunk_number] references
    into human-readable citations with a footer.
    """
    citation_map = {}
    for i, (text, meta) in enumerate(chunks, 1):
        mod = meta.get("modality", "text")
        src = meta.get("source", "unknown")
        page = meta.get("page", "")

        if mod == "audio":
            citation_map[i] = f"{src} (Audio, {page})"
        elif mod == "image":
            citation_map[i] = f"{src} (OCR, Region {page})"
        else:
            citation_map[i] = f"{src} (Page {page})"

    # Extract all referenced chunk numbers from the answer
    referenced = set()
    for match in re.finditer(r'\[(\d+)\]', answer):
        num = int(match.group(1))
        if num in citation_map:
            referenced.add(num)

    # Build citation footer
    if referenced:
        footer = "\n\n---\n**Sources:**\n"
        for num in sorted(referenced):
            footer += f"  [{num}] {citation_map[num]}\n"
        return answer + footer

    return answer


def verify_citations(answer: str, chunks: list) -> dict:
    """
    Verify that cited chunks actually support the claims.
    Returns a quality report.
    """
    referenced = set()
    for match in re.finditer(r'\[(\d+)\]', answer):
        referenced.add(int(match.group(1)))

    total_chunks = len(chunks)
    cited_chunks = len(referenced)

    # Check for hallucinated citations (citing non-existent chunks)
    valid_range = set(range(1, total_chunks + 1))
    invalid_citations = referenced - valid_range

    return {
        "total_chunks_retrieved": total_chunks,
        "chunks_cited": cited_chunks,
        "citation_coverage": cited_chunks / total_chunks if total_chunks > 0 else 0,
        "invalid_citations": list(invalid_citations),
        "has_any_citation": cited_chunks > 0,
    }
