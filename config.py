# config.py
import os
import shutil
import warnings

TESSERACT_PATH = os.environ.get("TESSERACT_CMD") or shutil.which("tesseract") or (
    r"C:\Program Files\Tesseract-OCR\tesseract.exe"
)


def _validate_tesseract() -> None:
    """Warn once if Tesseract binary is not reachable.

    EasyOCR still works without it; the warning prevents silent OCR degradation
    from going unnoticed when Tesseract was expected to be present.
    """
    if not TESSERACT_PATH or not os.path.isfile(TESSERACT_PATH):
        warnings.warn(
            f"Tesseract not found at {TESSERACT_PATH!r}. "
            "OCR will rely on EasyOCR only. Set TESSERACT_CMD or install Tesseract-OCR "
            "to enable dual-OCR extraction.",
            RuntimeWarning,
            stacklevel=2,
        )


_validate_tesseract()


EMBEDDING_MODEL   = "BAAI/bge-small-en-v1.5"
CHROMA_DB_PATH    = os.path.join(os.path.dirname(__file__), "chroma_db")
COLLECTION_NAME   = "lmif_collection"

CHUNK_SIZE        = 700
CHUNK_OVERLAP     = 70
CHUNK_MIN_CHARS   = 40     # drop very short text chunks (headers, page nums, junk)
IMAGE_MIN_CHARS   = 20     # image/OCR chunks keep lower bar than text
AUDIO_MIN_CHARS   = 20     # audio transcript chunks, likewise
DEDUP_THRESHOLD   = 0.92   # cosine-sim threshold for near-duplicate removal

TOP_K             = 7
BROAD_TOP_K_MULTIPLIER = 2  # broad/summary queries return TOP_K * this
NORMAL_FETCH_K    = 20      # ANN/BM25 fetch size for focused queries
BROAD_FETCH_K     = 40      # ANN/BM25 fetch size for broad/summary queries

LLM_MODEL         = "llama3"
OLLAMA_BASE_URL   = "http://localhost:11434"
LLM_NUM_CTX       = 8192       # context window; Ollama default 2048 silently truncates
LLM_TEMPERATURE   = 0.2        # low for factual RAG grounding
LLM_NUM_PREDICT   = 1024       # answer token budget
HYDE_TEMPERATURE  = 0.7        # slight creativity for hypothetical doc diversity
HYDE_NUM_PREDICT  = 200        # short hypothetical passage

WHISPER_MODEL     = "base"     # base / small / medium
WHISPER_MERGE_WINDOW = 30      # seconds per merged audio segment

RERANKER_MODEL    = "cross-encoder/ms-marco-MiniLM-L-6-v2"
RERANKER_MIN_SCORE = -5.0      # very permissive floor for cross-encoder logits

# Maximal Marginal Relevance — balances relevance vs diversity in final top-k.
# Higher lambda => more relevance; lower => more diversity.
USE_MMR_RERANK    = True
MMR_LAMBDA        = 0.6        # 0.6 tilts slightly toward diversity; good for multi-modality corpora

# Ingest cache: parsed parser output keyed by file content hash.
# Bump CACHE_SCHEMA_VERSION when the cached chunk dict shape changes
# (e.g. new required metadata fields) to force full re-parse.
CACHE_DIR            = os.path.join(os.path.dirname(__file__), "cache")
CACHE_SCHEMA_VERSION = 1

SUPPORTED_EXTS    = {
    "text":  [".pdf", ".docx", ".txt", ".md"],
    "image": [".jpg", ".jpeg", ".png", ".tiff", ".bmp", ".webp"],
    "audio": [".mp3", ".wav", ".m4a", ".flac", ".ogg"],
}
