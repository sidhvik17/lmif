import os
import numpy as np
import pytesseract
import cv2
from PIL import Image
from config import TESSERACT_PATH

if TESSERACT_PATH:
    pytesseract.pytesseract.tesseract_cmd = TESSERACT_PATH

# Lazy-loaded BLIP captioning model
_caption_model = None
_caption_processor = None


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


def preprocess_image(filepath):
    """Preprocess image for better OCR: grayscale, denoise, threshold."""
    # Use np.fromfile + imdecode to handle paths with spaces/unicode (Windows)
    buf = np.fromfile(filepath, dtype=np.uint8)
    img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    if img is None:
        return None
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    denoised = cv2.fastNlMeansDenoising(gray, h=10)
    _, thresh = cv2.threshold(
        denoised, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    return thresh


def extract_text_from_image(filepath):
    """Extract text via OCR + generate BLIP caption for any image."""
    chunks = []

    # 1. Try OCR for text-heavy images
    processed = preprocess_image(filepath)
    ocr_text = ""
    if processed is not None:
        ocr_text = (pytesseract.image_to_string(processed) or "").strip()

    if ocr_text:
        chunks.append({
            "text": ocr_text,
            "metadata": {
                "source": filepath,
                "page": "OCR Region",
                "modality": "image",
            },
        })

    # 2. Generate BLIP caption (works for ALL images including photos/diagrams)
    caption = generate_caption(filepath)
    if caption:
        # Avoid adding caption if it's essentially the same as OCR text
        if not ocr_text or caption.lower() not in ocr_text.lower():
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
