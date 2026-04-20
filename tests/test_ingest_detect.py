from ingestion.ingest_manager import detect_type


def test_detect_pdf():
    assert detect_type("doc.pdf") == "text"
    assert detect_type("DOC.PDF") == "text"


def test_detect_docx():
    assert detect_type("doc.docx") == "text"


def test_detect_markdown_and_txt():
    assert detect_type("notes.md") == "text"
    assert detect_type("notes.txt") == "text"


def test_detect_image_formats():
    for ext in [".jpg", ".jpeg", ".png", ".tiff", ".bmp", ".webp"]:
        assert detect_type(f"img{ext}") == "image", ext


def test_detect_audio_formats():
    for ext in [".mp3", ".wav", ".m4a", ".flac", ".ogg"]:
        assert detect_type(f"a{ext}") == "audio", ext


def test_unknown_extension_returns_none():
    assert detect_type("archive.zip") is None
    assert detect_type("noext") is None
