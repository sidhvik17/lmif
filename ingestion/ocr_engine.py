import os
import numpy as np
import pytesseract
import cv2
import easyocr
from PIL import Image
from config import TESSERACT_PATH

if TESSERACT_PATH:
    pytesseract.pytesseract.tesseract_cmd = TESSERACT_PATH

# Lazy-loaded models
_easyocr_reader = None
_caption_model = None
_caption_processor = None


def _get_easyocr():
    global _easyocr_reader
    if _easyocr_reader is None:
        _easyocr_reader = easyocr.Reader(["en"], gpu=False)
    return _easyocr_reader


def _get_captioner():
    global _caption_model, _caption_processor
    if _caption_model is None:
        from transformers import BlipProcessor, BlipForConditionalGeneration
        _caption_processor = BlipProcessor.from_pretrained(
            "Salesforce/blip-image-captioning-base"
        )
        _caption_model = BlipForConditionalGeneration.from_pretrained(
            "Salesforce/blip-image-captioning-base"
        )
    return _caption_processor, _caption_model


def generate_caption(filepath):
    """Generate a natural-language caption for an image using BLIP."""
    try:
        processor, model = _get_captioner()
        image = Image.open(filepath).convert("RGB")
        inputs = processor(image, return_tensors="pt")
        out = model.generate(**inputs, max_new_tokens=80)
        caption = processor.decode(out[0], skip_special_tokens=True)
        return caption.strip()
    except Exception as e:
        print(f"[CAPTION] BLIP captioning failed for {filepath}: {e}")
        return ""


def _easyocr_extract(filepath):
    """Extract text using EasyOCR (better for photos, cards, handwriting)."""
    try:
        reader = _get_easyocr()
        results = reader.readtext(filepath)
        # Filter by confidence and join text
        lines = []
        for bbox, text, conf in results:
            if conf > 0.2 and text.strip():
                lines.append(text.strip())
        return " ".join(lines)
    except Exception as e:
        print(f"[EasyOCR] Failed for {filepath}: {e}")
        return ""


def _tesseract_extract(filepath):
    """Extract text using Tesseract OCR (better for clean documents)."""
    buf = np.fromfile(filepath, dtype=np.uint8)
    img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    if img is None:
        return ""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    denoised = cv2.fastNlMeansDenoising(gray, h=10)
    _, thresh = cv2.threshold(
        denoised, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    return (pytesseract.image_to_string(thresh) or "").strip()


def extract_text_from_image(filepath):
    """Extract text via dual OCR (EasyOCR + Tesseract) + BLIP caption."""
    chunks = []

    # 1. EasyOCR (primary — better for photos, cards, real-world images)
    easy_text = _easyocr_extract(filepath)

    # 2. Tesseract (secondary — better for clean scanned docs)
    tess_text = _tesseract_extract(filepath)

    # Use the longer result as primary, keep both if substantially different
    if len(easy_text) >= len(tess_text):
        primary_text = easy_text
        secondary_text = tess_text
    else:
        primary_text = tess_text
        secondary_text = easy_text

    if primary_text:
        chunks.append({
            "text": primary_text,
            "metadata": {
                "source": filepath,
                "page": "OCR Region",
                "modality": "image",
            },
        })

    # Add secondary OCR if it has substantially different content
    if secondary_text and len(secondary_text) > 20:
        overlap = sum(1 for w in secondary_text.split() if w in primary_text)
        if overlap < len(secondary_text.split()) * 0.5:
            chunks.append({
                "text": secondary_text,
                "metadata": {
                    "source": filepath,
                    "page": "OCR Region (alt)",
                    "modality": "image",
                },
            })

    # 3. BLIP caption (works for ALL images including photos/diagrams)
    caption = generate_caption(filepath)
    if caption:
        if not primary_text or caption.lower() not in primary_text.lower():
            chunks.append({
                "text": f"[Image description]: {caption}",
                "metadata": {
                    "source": filepath,
                    "page": "BLIP Caption",
                    "modality": "image",
                },
            })

    if not chunks:
        print(f"[OCR] Could not extract text or caption from: {filepath}")

    return chunks
