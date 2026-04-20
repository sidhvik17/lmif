import os
from datetime import datetime, timezone
from typing import Any, Dict, List, NamedTuple, Optional

from config import SUPPORTED_EXTS
from ingestion.asr_engine import transcribe_audio
from ingestion.cache_manager import compute_file_hash, load_cached, save_cached
from ingestion.ocr_engine import extract_text_from_image
from ingestion.text_parser import parse_docx, parse_pdf, parse_plain_text
from logging_config import get_logger

log = get_logger(__name__)

SKIP_DIR_NAMES = {"venv", ".venv", "chroma_db", "cache", "__pycache__", ".git", "node_modules"}
MAX_FILE_SIZE_MB = 500


class IngestResult(NamedTuple):
    """Return type for ingest_file.

    NamedTuple (not bare list) so the API can grow — from_cache, source, and any
    future fields — without breaking callers. Callers that only want chunks use
    `.chunks`; legacy list-iteration patterns should be migrated to that.
    """
    chunks: List[Dict[str, Any]]
    from_cache: bool
    source: str


def detect_type(filepath: str) -> Optional[str]:
    """Detect file modality based on extension."""
    ext = os.path.splitext(filepath)[1].lower()
    for ftype, exts in SUPPORTED_EXTS.items():
        if ext in exts:
            return ftype
    return None


def _enrich_metadata(chunks: List[Dict[str, Any]], filepath: str) -> List[Dict[str, Any]]:
    """Attach production metadata: ingested_at, word_count, file_ext, file_name.

    Enables post-ingest filtering by time range, content length, file type
    without requiring a re-index. Word count also serves as a quality signal
    (very short chunks often carry little info).
    """
    ingested_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    file_ext = os.path.splitext(filepath)[1].lower()
    file_name = os.path.basename(filepath)
    for c in chunks:
        meta = c.setdefault("metadata", {})
        text = c.get("text", "") or ""
        # Always refresh ingested_at so re-ingestion (cached or not) updates
        # the provenance timestamp even though the parser output is reused.
        meta["ingested_at"] = ingested_at
        meta.setdefault("word_count", len(text.split()))
        meta.setdefault("file_ext", file_ext)
        meta.setdefault("file_name", file_name)
    return chunks


def _parse_file(filepath: str, ftype: str) -> List[Dict[str, Any]]:
    """Dispatch to the correct parser. Raises on hard parser failure; caller catches."""
    if ftype == "text":
        ext = os.path.splitext(filepath)[1].lower()
        if ext == ".pdf":
            return parse_pdf(filepath)
        if ext == ".docx":
            return parse_docx(filepath)
        return parse_plain_text(filepath)
    if ftype == "image":
        return extract_text_from_image(filepath)
    if ftype == "audio":
        return transcribe_audio(filepath)
    return []


def ingest_file(filepath: str) -> IngestResult:
    """Ingest single file → IngestResult(chunks, from_cache, source).

    Validates existence and size. Consults the content-hash cache before
    dispatching to parsers. Parser errors are caught so a single bad file
    doesn't abort directory ingestion.
    """
    empty = IngestResult([], False, filepath)

    if not os.path.isfile(filepath):
        log.warning("Skipping — not a file: %s", filepath)
        return empty

    try:
        size_mb = os.path.getsize(filepath) / (1024 * 1024)
    except OSError as e:
        log.warning("Cannot stat %s: %s", filepath, e)
        return empty
    if size_mb > MAX_FILE_SIZE_MB:
        log.warning("Skipping %s — %.1f MB exceeds %d MB cap", filepath, size_mb, MAX_FILE_SIZE_MB)
        return empty

    ftype = detect_type(filepath)
    if ftype is None:
        log.info("Skipping unsupported file: %s", filepath)
        return empty

    # Cache lookup — keyed by file content hash. The cached value is the
    # pre-chunking parser output (text/OCR/ASR), which is the expensive step.
    try:
        file_hash = compute_file_hash(filepath)
    except OSError as e:
        log.warning("Could not hash %s (%s); skipping cache.", filepath, e)
        file_hash = None

    if file_hash:
        cached = load_cached(file_hash)
        if cached is not None:
            log.info("Cache hit: %s (%d segment(s), skipping re-parse)",
                     os.path.basename(filepath), len(cached))
            return IngestResult(
                chunks=_enrich_metadata(cached, filepath),
                from_cache=True,
                source=filepath,
            )

    try:
        chunks = _parse_file(filepath, ftype)
    except Exception as e:
        log.error("Failed to process %s: %s", os.path.basename(filepath), e)
        return empty

    if file_hash and chunks:
        save_cached(file_hash, chunks, filepath)

    return IngestResult(
        chunks=_enrich_metadata(chunks, filepath),
        from_cache=False,
        source=filepath,
    )


def ingest_directory(dirpath: str) -> List[IngestResult]:
    """Ingest all supported files under dirpath (recursive).

    Returns one IngestResult per file so callers can surface per-file cache hits.
    Legacy callers that want a flat chunk list can do:
        all_chunks = [c for r in ingest_directory(p) for c in r.chunks]
    """
    results: List[IngestResult] = []
    for root, dirnames, files in os.walk(dirpath):
        dirnames[:] = [d for d in dirnames if d not in SKIP_DIR_NAMES]
        for fname in files:
            fpath = os.path.join(root, fname)
            if not os.path.isfile(fpath):
                continue
            rel = os.path.relpath(fpath, dirpath)
            result = ingest_file(fpath)
            results.append(result)
            tag = "cached" if result.from_cache else "parsed"
            log.info("%s -> %d raw segment(s) [%s]", rel, len(result.chunks), tag)
    return results
