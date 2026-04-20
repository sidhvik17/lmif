import os
import re
import tempfile

import fitz  # PyMuPDF
from docx import Document

from logging_config import get_logger

log = get_logger(__name__)


def _ocr_pdf_page_images(page, filepath, page_num):
    """Extract text from images embedded in a PDF page using OCR.

    Uses EasyOCR (lazy-imported to avoid overhead when not needed).
    Returns a list of chunks for any images found on the page.
    """
    chunks = []
    image_list = page.get_images(full=True)
    if not image_list:
        return chunks

    try:
        from ingestion.ocr_engine import _easyocr_extract
    except ImportError:
        return chunks  # OCR not available, skip image extraction

    doc = page.parent
    for img_info in image_list:
        xref = img_info[0]
        tmp_path = None
        try:
            base_image = doc.extract_image(xref)
            if not base_image:
                continue
            image_bytes = base_image["image"]
            ext = base_image.get("ext", "png")

            with tempfile.NamedTemporaryFile(suffix=f".{ext}", delete=False) as tmp:
                tmp.write(image_bytes)
                tmp_path = tmp.name

            ocr_text = _easyocr_extract(tmp_path)
            if ocr_text and len(ocr_text.strip()) > 10:
                chunks.append({
                    "text": ocr_text.strip(),
                    "metadata": {
                        "source": filepath,
                        "page": f"{page_num} (embedded image)",
                        "modality": "image",
                    },
                })
        except Exception as e:
            log.warning("PDF-OCR: failed xref=%s page=%s: %s", xref, page_num, e)
        finally:
            if tmp_path and os.path.isfile(tmp_path):
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass
    return chunks


# ----- Optional Docling branch ---------------------------------------------

_DOCLING_AVAILABLE = None  # tri-state: None=unknown, True/False after first probe
_DOCLING_CONVERTER = None


def _try_import_docling():
    """Probe Docling availability once. Cached after first call."""
    global _DOCLING_AVAILABLE, _DOCLING_CONVERTER
    if _DOCLING_AVAILABLE is not None:
        return _DOCLING_AVAILABLE
    try:
        from docling.document_converter import DocumentConverter  # type: ignore
        _DOCLING_CONVERTER = DocumentConverter()
        _DOCLING_AVAILABLE = True
        log.info("Docling available — will prefer it for PDF parsing.")
    except ImportError:
        _DOCLING_AVAILABLE = False
    except Exception as e:
        # Docling may fail to construct (model download blocked, etc.) — degrade silently.
        log.warning("Docling import/init failed (%s); using PyMuPDF.", e)
        _DOCLING_AVAILABLE = False
    return _DOCLING_AVAILABLE


_HEADING_SPLIT_RE = re.compile(r"\n(?=#{1,6}\s)")


def _split_markdown_sections(md: str):
    """Split a Markdown document into chunks at heading boundaries.

    Keeps each heading with the content that follows it. Falls back to a single
    chunk if there are no headings.
    """
    parts = _HEADING_SPLIT_RE.split(md)
    return [p.strip() for p in parts if p.strip()]


def _docling_parse_pdf(filepath):
    """Parse a PDF via Docling. Returns chunks list or None on unavailability/error."""
    if not _try_import_docling():
        return None
    try:
        result = _DOCLING_CONVERTER.convert(filepath)
        md = result.document.export_to_markdown()
        if not md or not md.strip():
            return None
        sections = _split_markdown_sections(md)
        return [
            {
                "text": sec,
                "metadata": {
                    "source": filepath,
                    "page": f"section {i + 1}",
                    "modality": "text",
                },
            }
            for i, sec in enumerate(sections)
        ]
    except Exception as e:
        log.warning("Docling failed on %s (%s) — falling back to PyMuPDF.", filepath, e)
        return None


def _pymupdf_parse_pdf(filepath):
    """Extract text + embedded-image OCR from a PDF via PyMuPDF. Original path."""
    chunks = []
    with fitz.open(filepath) as doc:
        for i, page in enumerate(doc):
            page_num = i + 1
            try:
                text = page.get_text()
            except Exception as e:
                log.warning("PDF: page %s text extraction failed: %s", page_num, e)
                text = ""
            if text and text.strip():
                chunks.append({
                    "text": text,
                    "metadata": {
                        "source": filepath,
                        "page": page_num,
                        "modality": "text",
                    },
                })

            try:
                img_chunks = _ocr_pdf_page_images(page, filepath, page_num)
                chunks.extend(img_chunks)
            except Exception as e:
                log.warning("PDF-OCR: image extraction failed page %s: %s", page_num, e)

    return chunks


def parse_pdf(filepath):
    """Extract text from PDF. Prefers Docling (layout/tables) when installed."""
    docling_chunks = _docling_parse_pdf(filepath)
    if docling_chunks:
        log.info("Parsed %s via Docling (%d section(s))",
                 os.path.basename(filepath), len(docling_chunks))
        return docling_chunks
    return _pymupdf_parse_pdf(filepath)


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
    """Extract text from DOCX file.

    Splits into sections based on heading styles. If no headings are found,
    returns the full text as a single chunk.
    """
    doc = Document(filepath)
    paragraphs = [p for p in doc.paragraphs if p.text.strip()]
    if not paragraphs:
        return []

    chunks = []
    current_section = []
    section_num = 1

    for p in paragraphs:
        if p.style and p.style.name and p.style.name.startswith("Heading"):
            if current_section:
                chunks.append({
                    "text": "\n".join(current_section),
                    "metadata": {
                        "source": filepath,
                        "page": f"section {section_num}",
                        "modality": "text",
                    },
                })
                section_num += 1
                current_section = []
        current_section.append(p.text)

    if current_section:
        chunks.append({
            "text": "\n".join(current_section),
            "metadata": {
                "source": filepath,
                "page": f"section {section_num}" if section_num > 1 else 1,
                "modality": "text",
            },
        })

    return chunks
