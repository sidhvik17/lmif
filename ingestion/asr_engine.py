import os
import speech_recognition as sr


def transcribe_audio(filepath):
    """Transcribe audio file using SpeechRecognition."""
    recognizer = sr.Recognizer()

    # Convert mp3/other formats to wav if needed
    wav_path = filepath
    converted = False
    if not filepath.lower().endswith(".wav"):
        try:
            from pydub import AudioSegment
            audio = AudioSegment.from_file(filepath)
            wav_path = filepath + ".tmp.wav"
            audio.export(wav_path, format="wav")
            converted = True
        except ImportError:
            # If pydub not available, try direct load (may fail for non-wav)
            pass

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
