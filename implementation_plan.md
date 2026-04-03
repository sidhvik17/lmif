# LMIF — Build All Project Modules

Build the complete Localized Multimodal Intelligence Framework from the user's PRD and step-by-step guide. All prerequisites (Python, Tesseract, FFmpeg, Ollama) are already installed. The folder structure exists with empty Python files.

## Proposed Changes

### Foundation

#### [MODIFY] [requirements.txt](file:///c:/Users/ASUS/OneDrive/Desktop/lmif/requirements.txt)
Fix the malformed last line — split `torch torchvision torchaudio` into separate entries and add the `--index-url` as a proper pip flag line. Also add `opencv-python` and `Pillow` (needed by OCR).

#### [MODIFY] [config.py](file:///c:/Users/ASUS/OneDrive/Desktop/lmif/config.py)
Write all configuration constants: paths, model names, chunk sizes, supported extensions.

---

### Ingestion Modules

#### [MODIFY] [text_parser.py](file:///c:/Users/ASUS/OneDrive/Desktop/lmif/ingestion/text_parser.py)
PDF parsing (PyMuPDF) and DOCX parsing (python-docx) with page-level metadata.

#### [MODIFY] [ocr_engine.py](file:///c:/Users/ASUS/OneDrive/Desktop/lmif/ingestion/ocr_engine.py)
Image preprocessing (grayscale, denoise, threshold via OpenCV) + Tesseract OCR extraction.

#### [MODIFY] [asr_engine.py](file:///c:/Users/ASUS/OneDrive/Desktop/lmif/ingestion/asr_engine.py)
Whisper-based audio transcription with timestamped segments.

#### [MODIFY] [ingest_manager.py](file:///c:/Users/ASUS/OneDrive/Desktop/lmif/ingestion/ingest_manager.py)
File type detection and routing to the correct parser. Supports single file and directory ingestion.

---

### Pipeline

#### [MODIFY] [chunker.py](file:///c:/Users/ASUS/OneDrive/Desktop/lmif/pipeline/chunker.py)
LangChain `RecursiveCharacterTextSplitter` for chunking text from all modalities.

#### [MODIFY] [embedder.py](file:///c:/Users/ASUS/OneDrive/Desktop/lmif/pipeline/embedder.py)
`sentence-transformers` (all-MiniLM-L6-v2) for dense vector embeddings.

---

### Vector Store

#### [MODIFY] [store.py](file:///c:/Users/ASUS/OneDrive/Desktop/lmif/vectorstore/store.py)
ChromaDB persistent client with upsert and cosine similarity search.

---

### Retrieval & Generation

#### [MODIFY] [retriever.py](file:///c:/Users/ASUS/OneDrive/Desktop/lmif/retrieval/retriever.py)
Embeds query and retrieves top-k chunks from ChromaDB.

#### [MODIFY] [generator.py](file:///c:/Users/ASUS/OneDrive/Desktop/lmif/generation/generator.py)
Ollama LLM call with context-grounded prompt template.

#### [MODIFY] [citation_formatter.py](file:///c:/Users/ASUS/OneDrive/Desktop/lmif/generation/citation_formatter.py)
Formats citations by modality (text→page, image→OCR Region, audio→timestamp).

---

### Entry Points

#### [MODIFY] [cli.py](file:///c:/Users/ASUS/OneDrive/Desktop/lmif/cli.py)
Typer CLI with `ingest` and `query` commands.

#### [MODIFY] [app.py](file:///c:/Users/ASUS/OneDrive/Desktop/lmif/ui/app.py)
Streamlit web UI with Ingest and Query tabs.

## Verification Plan

### Automated Tests
1. **Import checks** — Run individual module imports to verify no syntax errors:
   ```
   cd c:\Users\ASUS\OneDrive\Desktop\lmif
   venv\Scripts\activate
   python -c "from ingestion.ingest_manager import ingest_file; print('OK')"
   python -c "from pipeline.embedder import embed_chunks; print('OK')"
   python -c "from vectorstore.store import collection; print('OK')"
   python -c "from retrieval.retriever import retrieve; print('OK')"
   python -c "from generation.generator import generate; print('OK')"
   ```

### Manual Verification
1. **CLI ingest** — Place a test PDF in `data/`, run `python cli.py ingest ./data`, verify chunks are indexed
2. **CLI query** — Run `python cli.py query "test question"`, verify answer + citations appear
3. **Streamlit UI** — Run `streamlit run ui/app.py`, open browser, test upload and query
