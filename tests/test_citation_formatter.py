from generation.citation_formatter import format_citations, verify_citations


def _chunks():
    return [
        ("text 1", {"modality": "text", "source": "/p/doc.pdf", "page": 3}),
        ("text 2", {"modality": "audio", "source": "/p/a.mp3", "page": "00:12 - 00:25"}),
        ("text 3", {"modality": "image", "source": "/p/img.png", "page": "OCR Region"}),
    ]


def test_format_adds_footer_for_referenced_chunks():
    out = format_citations("claim [1] and also [3].", _chunks())
    assert "**Sources:**" in out
    assert "doc.pdf" in out
    assert "img.png" in out
    assert "a.mp3" not in out, "chunk 2 not cited, should be omitted"


def test_format_no_footer_when_no_citations():
    answer = "I said something without citations."
    out = format_citations(answer, _chunks())
    assert out == answer


def test_format_ignores_invalid_citation_numbers():
    out = format_citations("see [99]", _chunks())
    assert "**Sources:**" not in out


def test_verify_detects_invalid_citations():
    report = verify_citations("see [1] and [99]", _chunks())
    assert report["chunks_cited"] == 2
    assert report["invalid_citations"] == [99]
    assert report["has_any_citation"] is True


def test_verify_no_citations_flags_zero():
    report = verify_citations("plain text", _chunks())
    assert report["has_any_citation"] is False
    assert report["citation_coverage"] == 0
