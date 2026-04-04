import os
import tempfile
import speech_recognition as sr
from pydub import AudioSegment


def convert_to_wav(filepath):
    """Convert any audio format to PCM WAV."""
    ext = os.path.splitext(filepath)[1].lower()
    if ext == ".wav":
        return filepath, False
    
    audio = AudioSegment.from_file(filepath)
    wav_path = os.path.join(tempfile.gettempdir(), "lmif_temp.wav")
    audio.export(wav_path, format="wav")
    return wav_path, True


def transcribe_audio(filepath):
    """Transcribe audio file using SpeechRecognition."""
    recognizer = sr.Recognizer()

    # Convert mp3/other formats to WAV
    wav_path, converted = convert_to_wav(filepath)

    chunks = []
    try:
        with sr.AudioFile(wav_path) as source:
            duration = source.DURATION
            # Process in 30-second segments
            segment_len = 30
            offset = 0.0
            while offset < duration:
                length = min(segment_len, duration - offset)
                audio_data = recognizer.record(source, duration=length)
                try:
                    text = recognizer.recognize_google(audio_data)
                except sr.UnknownValueError:
                    text = ""
                except sr.RequestError:
                    text = "[Speech recognition unavailable]"

                if text.strip():
                    start_m, start_s = divmod(int(offset), 60)
                    end_m, end_s = divmod(int(offset + length), 60)
                    chunks.append({
                        "text": text.strip(),
                        "metadata": {
                            "source": filepath,
                            "page": f"{start_m:02d}:{start_s:02d} - {end_m:02d}:{end_s:02d}",
                            "modality": "audio",
                        },
                    })
                offset += segment_len
    finally:
        if converted and os.path.exists(wav_path):
            os.remove(wav_path)

    return chunks
