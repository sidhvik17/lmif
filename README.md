# LMIF — Local Multimodal Intelligence Framework

Offline RAG over text, images, and audio. Runs locally. No cloud calls at query time.

## What it does

- Ingests PDF/DOCX/TXT/MD, JPG/PNG/TIFF/BMP/WEBP, MP3/WAV/M4A/FLAC/OGG
- Extracts via PyMuPDF, python-docx, EasyOCR + Tesseract + BLIP, Whisper
- Embeds with BGE, stores in ChromaDB
- Hybrid retrieval: dense + BM25 + cross-encoder rerank, optional HyDE
- Answers via Ollama (llama3 by default) with chunk-level citations

## First-run: downloads ~2 GB from HuggingFace

"Offline" means at query time. First ingest/query pulls models:

- `BAAI/bge-small-en-v1.5` — ~130 MB (embeddings)
- `cross-encoder/ms-marco-MiniLM-L-6-v2` — ~90 MB (reranker)
- `Salesforce/blip-image-captioning-base` — ~990 MB (captions, only if ingesting images)
- Whisper `base` — ~140 MB (only if ingesting audio)
- EasyOCR English — ~60 MB (only if ingesting images)

## Prerequisites

- Python 3.11 or 3.12 (3.14 currently fails — LangChain uses Pydantic v1)
- [Tesseract OCR](https://github.com/UB-Mannheim/tesseract/wiki) — set `TESSERACT_CMD` env var if not on PATH
- [FFmpeg](https://ffmpeg.org/) — required by Whisper for non-WAV audio
- [Ollama](https://ollama.com/) — `ollama pull llama3`

## Install

```bash
python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # Linux/Mac
pip install -r requirements.txt
```

## CLI

```bash
python cli.py ingest ./data              # file or directory
python cli.py query "your question"
python cli.py query "summarize" --hyde   # HyDE query expansion
python cli.py query "q" --modality text  # filter by modality
python cli.py stats                      # collection size + modality breakdown
python cli.py forget /path/to/file.pdf   # remove all chunks from one file
python cli.py clear --yes                # wipe collection
```

## Streamlit UI

```bash
streamlit run ui/app.py
```

## Configuration

Edit `config.py` or set env vars:

| Var | Purpose |
|-----|---------|
| `TESSERACT_CMD` | Path to Tesseract binary |
| `LMIF_LOG_LEVEL` | `DEBUG` / `INFO` / `WARNING` / `ERROR` (default INFO) |

Key `config.py` constants:

- `EMBEDDING_MODEL` — swap to `bge-m3` for multilingual
- `LLM_MODEL` — Ollama model name
- `LLM_NUM_CTX` — Ollama context window (default 8192; raises memory)
- `CHUNK_SIZE`, `CHUNK_OVERLAP`, `CHUNK_MIN_CHARS`
- `TOP_K` — chunks passed to LLM after rerank

## Advanced: filter API

```python
from retrieval.retriever import retrieve
chunks = retrieve(
    "what was said",
    filters={
        "modality": "audio",
        "word_count": {"$gte": 50},   # Chroma operator
        "ingested_at": lambda v: v and v > "2026-01-01",  # callable
    },
)
```

Operators: `$eq $ne $gt $gte $lt $lte $in $nin`.

## Development

```bash
pip install -r requirements-dev.txt
pytest                                   # 49 unit tests
ruff check .                             # lint
```

## Troubleshooting

- **Ollama unreachable** — start it, then `ollama pull llama3`
- **Tesseract not found** — set `TESSERACT_CMD` or install to default path
- **HF download fails** — check internet on first ingest
- **Pydantic v1 error on Python 3.14** — use 3.11 or 3.12
- **Short/junk answers** — try `--hyde`, raise `TOP_K`, or re-ingest if chunks < min length filter

## Architecture

```
ingest → parse → chunk (+modality prefix) → embed → ChromaDB
query  → embed/HyDE → Chroma(dense) + BM25(sparse) → RRF merge
       → post-filter → cross-encoder rerank → Ollama → citations
```

## License

Local use. No warranty.
