# config.py
import os
import shutil

TESSERACT_PATH = os.environ.get("TESSERACT_CMD") or shutil.which("tesseract") or (
    r"C:\Program Files\Tesseract-OCR\tesseract.exe"
)

EMBEDDING_MODEL   = "all-MiniLM-L6-v2"
CHROMA_DB_PATH    = os.path.join(os.path.dirname(__file__), "chroma_db")
COLLECTION_NAME   = "lmif_collection"

CHUNK_SIZE        = 700
CHUNK_OVERLAP     = 70
TOP_K             = 7

LLM_MODEL         = "llama3"
OLLAMA_BASE_URL   = "http://localhost:11434"

WHISPER_MODEL     = "base"     # base / small / medium

SUPPORTED_EXTS    = {
    "text":  [".pdf", ".docx", ".txt", ".md"],
    "image": [".jpg", ".jpeg", ".png", ".tiff"],
    "audio": [".mp3", ".wav"],
}
