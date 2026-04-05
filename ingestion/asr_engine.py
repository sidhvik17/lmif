import os
import whisper
from config import WHISPER_MODEL

# Load Whisper model once at module level
_model = None


def _get_model():
    global _model
    if _model is None:
        _model = whisper.load_model(WHISPER_MODEL)
    return _model


def transcribe_audio(filepath):
    """Transcribe audio file using OpenAI Whisper (fully offline)."""
    model = _get_model()
    result = model.transcribe(filepath, verbose=False)

    chunks = []
    segments = result.get("segments", [])
    if not segments:
        # Fallback: use full text as one chunk
        text = result.get("text", "").strip()
        if text:
            chunks.append({
                "text": text,
                "metadata": {
                    "source": filepath,
                    "page": "00:00 - full",
                    "modality": "audio",
                },
            })
        return chunks

    for seg in segments:
        text = seg.get("text", "").strip()
        if not text:
            continue
        start = int(seg["start"])
        end = int(seg["end"])
        start_m, start_s = divmod(start, 60)
        end_m, end_s = divmod(end, 60)
        chunks.append({
            "text": text,
            "metadata": {
                "source": filepath,
                "page": f"{start_m:02d}:{start_s:02d} - {end_m:02d}:{end_s:02d}",
                "modality": "audio",
            },
        })

    return chunks
