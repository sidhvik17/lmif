import os
import numpy as np
import pytesseract
import cv2
from config import TESSERACT_PATH

if TESSERACT_PATH:
    pytesseract.pytesseract.tesseract_cmd = TESSERACT_PATH


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
    """Extract text from image file using Tesseract OCR."""
    processed = preprocess_image(filepath)
    if processed is None:
        print(f"[OCR] Could not read image: {filepath}")
        return []
    text = pytesseract.image_to_string(processed)
    if not (text or "").strip():
        return []
    return [{
        "text": text,
        "metadata": {
            "source": filepath,
            "page": "OCR Region",
            "modality": "image",
        },
    }]
