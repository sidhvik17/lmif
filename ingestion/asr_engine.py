import os
import whisper
from config import WHISPER_MODEL

# Load Whisper model once at module level
_model = None

# Merge segments into groups of this many seconds
MERGE_WINDOW = 30


def _get_model():
    global _model
    if _model is None:
        _model = whisper.load_model(WHISPER_MODEL)
    return _model


def _merge_segments(segments, window=MERGE_WINDOW):
    """Merge Whisper segments into larger chunks by time window.

    Prevents tiny 1-2 second chunks that lack context for the LLM.
    """
    if not segments:
        return []

    merged = []
    buf_texts = []
    buf_start = int(segments[0]["start"])

    for seg in segments:
        text = seg.get("text", "").strip()
        if not text:
            continue
        end = int(seg["end"])

        # If adding this segment exceeds window, flush buffer
        if buf_texts and (end - buf_start) > window:
            merged.append({
                "text": " ".join(buf_texts),
                "start": buf_start,
                "end": int(segments[segments.index(seg) - 1]["end"]) if segments.index(seg) > 0 else end,
            })
            buf_texts = []
            buf_start = int(seg["start"])

        buf_texts.append(text)

    # Flush remaining
    if buf_texts:
        merged.append({
            "text": " ".join(buf_texts),
            "start": buf_start,
            "end": int(segments[-1]["end"]),
        })

    return merged


def transcribe_audio(filepath):
    """Transcribe audio file using OpenAI Whisper (fully offline)."""
    model = _get_model()
    result = model.transcribe(filepath, verbose=False)

    segments = result.get("segments", [])
    if not segments:
        text = result.get("text", "").strip()
        if text:
            return [{
                "text": text,
                "metadata": {
                    "source": filepath,
                    "page": "00:00 - full",
                    "modality": "audio",
                },
            }]
        return []

    # Merge small segments into ~30s windows for better LLM context
    merged = _merge_segments(segments)

    chunks = []
    for group in merged:
        start_m, start_s = divmod(group["start"], 60)
        end_m, end_s = divmod(group["end"], 60)
        chunks.append({
            "text": group["text"],
            "metadata": {
                "source": filepath,
                "page": f"{start_m:02d}:{start_s:02d} - {end_m:02d}:{end_s:02d}",
                "modality": "audio",
            },
        })

    return chunks
