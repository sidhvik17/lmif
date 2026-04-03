import os
from config import SUPPORTED_EXTS
from ingestion.text_parser import parse_pdf, parse_docx, parse_plain_text
from ingestion.ocr_engine import extract_text_from_image
from ingestion.asr_engine import transcribe_audio

SKIP_DIR_NAMES = {"venv", ".venv", "chroma_db", "__pycache__", ".git", "node_modules"}


def detect_type(filepath):
    """Detect file modality based on extension."""
    ext = os.path.splitext(filepath)[1].lower()
    for ftype, exts in SUPPORTED_EXTS.items():
        if ext in exts:
            return ftype
    return None


def ingest_file(filepath):
    """Ingest a single file and return list of chunks with metadata."""
    ftype = detect_type(filepath)
    if ftype == "text":
        ext = os.path.splitext(filepath)[1].lower()
        if ext == ".pdf":
            return parse_pdf(filepath)
        if ext == ".docx":
            return parse_docx(filepath)
        return parse_plain_text(filepath)
    elif ftype == "image":
        return extract_text_from_image(filepath)
    elif ftype == "audio":
        return transcribe_audio(filepath)
    else:
        print(f"[SKIP] Unsupported file: {filepath}")
        return []


def ingest_directory(dirpath):
    """Ingest all supported files under a directory (recursive)."""
    all_chunks = []
    for root, dirnames, files in os.walk(dirpath):
        dirnames[:] = [d for d in dirnames if d not in SKIP_DIR_NAMES]
        for fname in files:
            fpath = os.path.join(root, fname)
            if not os.path.isfile(fpath):
                continue
            rel = os.path.relpath(fpath, dirpath)
            try:
                chunks = ingest_file(fpath)
                all_chunks.extend(chunks)
                print(f"[OK] {rel} -> {len(chunks)} raw segment(s)")
            except Exception as e:
                print(f"[ERR] {rel}: {e}")
    return all_chunks
