# LMIF — Local Multimodal Information Fusion
# Reproducible Linux image. CPU-only by default; for CUDA, swap base image and torch wheels.
FROM python:3.12-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HOME=/app/.cache/huggingface \
    TRANSFORMERS_CACHE=/app/.cache/huggingface \
    LMIF_LOG_LEVEL=INFO

# System deps:
#   tesseract-ocr           - OCR engine
#   ffmpeg                  - audio decoding for whisper/pydub
#   libgl1 / libglib2.0-0   - OpenCV runtime deps
#   poppler-utils           - PDF rendering fallback
#   build-essential         - compile wheels that lack linux manylinux
RUN apt-get update && apt-get install -y --no-install-recommends \
        tesseract-ocr \
        ffmpeg \
        libgl1 \
        libglib2.0-0 \
        poppler-utils \
        build-essential \
        git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install deps first for better layer caching
COPY requirements.txt ./
RUN pip install --upgrade pip \
    && pip install -r requirements.txt

COPY . .

# Pre-warm model caches at build time is optional; commented out to keep the
# image small. Uncomment to bake models into the image (adds ~2GB).
# RUN python -c "from sentence_transformers import SentenceTransformer, CrossEncoder; \
#     SentenceTransformer('BAAI/bge-small-en-v1.5'); \
#     CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')"

EXPOSE 8501

# Default: launch the Streamlit UI. Override with `docker run ... python cli.py ...`
CMD ["streamlit", "run", "ui/app.py", "--server.port=8501", "--server.address=0.0.0.0"]
