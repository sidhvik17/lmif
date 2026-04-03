import fitz  # PyMuPDF
from docx import Document


def parse_pdf(filepath):
    """Extract text from PDF file, page by page."""
    chunks = []
    with fitz.open(filepath) as doc:
        for i, page in enumerate(doc):
            text = page.get_text()
            if text.strip():
                chunks.append({
                    "text": text,
                    "metadata": {
                        "source": filepath,
                        "page": i + 1,
                        "modality": "text",
                    },
                })
    return chunks


def parse_plain_text(filepath):
    """Extract text from a plain-text or markdown file."""
    with open(filepath, encoding="utf-8", errors="replace") as f:
        text = f.read()
    if not text.strip():
        return []
    return [{
        "text": text,
        "metadata": {
            "source": filepath,
            "page": 1,
            "modality": "text",
        },
    }]


def parse_docx(filepath):
    """Extract text from DOCX file."""
    doc = Document(filepath)
    full_text = "\n".join([p.text for p in doc.paragraphs if p.text.strip()])
    if not full_text.strip():
        return []
    return [{
        "text": full_text,
        "metadata": {
            "source": filepath,
            "page": 1,
            "modality": "text",
        },
    }]
