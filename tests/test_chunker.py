from pipeline.chunker import chunk_documents


def _raw(text, modality="text", source="x.txt", page=1):
    return {"text": text, "metadata": {"modality": modality, "source": source, "page": page}}


def test_short_text_chunk_dropped():
    """Text chunks below CHUNK_MIN_CHARS get filtered."""
    out = chunk_documents([_raw("ab")])
    assert out == []


def test_long_text_chunk_kept():
    out = chunk_documents([_raw("Hello world, this is a long enough chunk to survive.")])
    assert len(out) == 1
    assert "chunk_index" in out[0]["metadata"]


def test_audio_prefix_added():
    text = "This is a spoken transcript with enough body to clear the min filter."
    out = chunk_documents([_raw(text, modality="audio")])
    assert out, "audio chunk should survive"
    assert out[0]["text"].startswith("Audio transcript of spoken content:")


def test_image_prefix_added():
    text = "Extracted card text reading name address phone number here."
    out = chunk_documents([_raw(text, modality="image")])
    assert out
    assert out[0]["text"].startswith("Text extracted from image:")


def test_text_has_no_prefix():
    text = "Plain document text without any modality prefix attached to it."
    out = chunk_documents([_raw(text, modality="text")])
    assert out
    assert not out[0]["text"].startswith("Audio")
    assert not out[0]["text"].startswith("Text extracted")


def test_empty_input_returns_empty():
    assert chunk_documents([]) == []


def test_whitespace_only_skipped():
    assert chunk_documents([_raw("   \n\t  ")]) == []


def test_chunk_index_is_sequential():
    long_text = ("paragraph one is here " * 50) + "\n\n" + ("paragraph two is there " * 50)
    out = chunk_documents([_raw(long_text)])
    indices = [c["metadata"]["chunk_index"] for c in out]
    assert indices == list(range(len(indices)))
