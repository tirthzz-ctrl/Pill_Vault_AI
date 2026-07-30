"""
OpenCV-based medication label scanner.

Pipeline: load image → preprocess (denoise, threshold, deskew) →
          OCR (EasyOCR) → parse structured medication data.

Requires: pip install opencv-python easyocr
"""

from __future__ import annotations

import os
import re
import time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

try:
    import easyocr
    _EASYOCR_AVAILABLE = True
    _EASYOCR_READER = None
except ImportError:
    _EASYOCR_AVAILABLE = False
    _EASYOCR_READER = None


def _get_easyocr_reader():
    global _EASYOCR_READER
    if _EASYOCR_READER is None and _EASYOCR_AVAILABLE:
        _EASYOCR_READER = easyocr.Reader(["en"], gpu=False, verbose=False)
    return _EASYOCR_READER


def preprocess_image(image: np.ndarray) -> np.ndarray:
    """Apply OpenCV preprocessing to make text more readable."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    denoised = cv2.fastNlMeansDenoising(gray, h=30)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(denoised)

    blurred = cv2.GaussianBlur(enhanced, (3, 3), 0)

    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    cleaned = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel)

    return cleaned


def deskew(image: np.ndarray) -> np.ndarray:
    """Correct skew in the image."""
    coords = np.column_stack(np.where(image > 0))
    if len(coords) < 10:
        return image
    angle = cv2.minAreaRect(coords)[-1]
    if angle < -45:
        angle = 90 + angle
    if abs(angle) < 0.5:
        return image
    h, w = image.shape[:2]
    center = (w // 2, h // 2)
    matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(
        image, matrix, (w, h),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE,
    )
    return rotated


def find_label_region(image: np.ndarray) -> np.ndarray:
    """Detect and crop to the label region using contour detection."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(blurred, 50, 150)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    dilated = cv2.dilate(edged, kernel, iterations=2)
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return image

    # Find the largest rectangular contour (likely the label)
    best = None
    best_area = 0
    for c in contours:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        if len(approx) == 4:
            area = cv2.contourArea(approx)
            if area > best_area:
                best_area = area
                best = approx

    if best is None or best_area < image.shape[0] * image.shape[1] * 0.05:
        return image

    x, y, w, h = cv2.boundingRect(best)
    margin = 10
    x = max(0, x - margin)
    y = max(0, y - margin)
    w = min(image.shape[1] - x, w + 2 * margin)
    h = min(image.shape[0] - y, h + 2 * margin)

    return image[y : y + h, x : x + w]


def extract_text(image: np.ndarray) -> str:
    """Extract text from preprocessed image using EasyOCR."""
    reader = _get_easyocr_reader()
    if reader is None:
        return ""

    # EasyOCR expects RGB or BGR
    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    results = reader.readtext(image, paragraph=False)
    lines = []
    for bbox, text, confidence in results:
        if confidence > 0.2:
            lines.append(text.strip())
    return "\n".join(lines)


def parse_medication_text(raw_text: str) -> dict:
    """Parse OCR text to extract structured medication data."""
    result = {
        "medication_name": "",
        "dosage_mg": "",
        "rx_number": "",
        "total_quantity": 0,
        "expiration_date": "",
        "doctor_name": "",
    }

    # Medication name: often the first capitalized line
    lines = [l.strip() for l in raw_text.split("\n") if l.strip()]
    if not lines:
        return result

    # Rx number pattern
    rx_pattern = re.compile(r"(?:rx|prescription|#)\s*[#:.\s]*([A-Za-z0-9\-]+)", re.IGNORECASE)
    for line in lines:
        m = rx_pattern.search(line)
        if m:
            result["rx_number"] = m.group(1).strip()
            break

    # Dosage pattern: number followed by mg/mcg/g/ml
    dose_pattern = re.compile(r"(\d+(?:\.\d+)?)\s*(mg|mcg|microgram|gram|g|ml)", re.IGNORECASE)
    for line in lines:
        m = dose_pattern.search(line)
        if m:
            val = float(m.group(1))
            unit = m.group(2).lower()
            if unit in ("mcg", "microgram"):
                val = val / 1000.0
            elif unit in ("g", "gram"):
                val = val * 1000.0
            result["dosage_mg"] = str(int(val)) if val == int(val) else f"{val:.3f}".rstrip("0").rstrip(".")
            break

    # Quantity: "Qty: N" or "N Tablets" or "N Capsules"
    qty_patterns = [
        re.compile(r"qty\s*[:\s]+(\d+)", re.IGNORECASE),
        re.compile(r"(\d+)\s*(tablets|capsules|tabs|caps|pills)", re.IGNORECASE),
        re.compile(r"count\s*[:\s]+(\d+)", re.IGNORECASE),
        re.compile(r"(\d+)\s*(ea|ct|count)", re.IGNORECASE),
    ]
    for pattern in qty_patterns:
        for line in lines:
            m = pattern.search(line)
            if m:
                result["total_quantity"] = int(m.group(1))
                break
        if result["total_quantity"]:
            break

    # Expiration date: various date formats
    exp_pattern = re.compile(
        r"(?:exp|expiry|expiration|expires|use by)\s*[:\s]*(\d{1,2}[/-]\d{2,4}(?:[/-]\d{2,4})?)",
        re.IGNORECASE,
    )
    for line in lines:
        m = exp_pattern.search(line)
        if m:
            result["expiration_date"] = m.group(1)
            break

    # Doctor name: often "Dr." or "Physician" after a keyword
    doc_pattern = re.compile(r"(?:dr|doctor|physician)\s*[.\s]*([A-Za-z\s\']+)", re.IGNORECASE)
    for line in lines:
        m = doc_pattern.search(line)
        if m:
            result["doctor_name"] = m.group(1).strip()
            break

    # Medication name: first line that isn't a known filler
    skip_words = {"rx", "prescription", "pharmacy", "drug", "medication",
                  "tablet", "capsule", "bottle", "label", "ndc", "qty",
                  "exp", "lot", "mfg", "dist", "keep", "do not", "warning"}
    for line in lines:
        cleaned = re.sub(r"^\d+\s*", "", line).strip()
        words = cleaned.lower().split()
        if not words or words[0] in skip_words or len(cleaned) < 3:
            continue
        result["medication_name"] = cleaned
        break

    return result


def capture_from_camera(camera_id: int = 0) -> Optional[str]:
    """Open webcam, show live preview, capture a frame on SPACE or Enter.

    Returns the file path to the saved captured image, or None on failure.

    Controls:
      SPACE / Enter  — capture frame and return
      ESC / Q        — abort return None
    """
    cap = cv2.VideoCapture(camera_id)
    if not cap.isOpened():
        print("  [cam] Could not open camera.")
        return None

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

    print("  [cam] Camera opened. Press SPACE to capture, ESC to cancel.")
    captured = None

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        display = frame.copy()
        cv2.putText(display, "SPACE: capture | ESC: cancel",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.imshow("PillVault - Medicine Label Scanner", display)

        key = cv2.waitKey(1) & 0xFF
        if key == 27 or key == ord("q"):  # ESC / q = cancel
            break
        elif key == 32 or key == 13:      # SPACE / Enter = capture
            captured = frame.copy()
            break

    cap.release()
    cv2.destroyAllWindows()

    if captured is None:
        print("  [cam] Capture cancelled.")
        return None

    out_dir = Path.home() / ".pillvault"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = str(out_dir / f"capture_{int(time.time())}.jpg")
    cv2.imwrite(out_path, captured)
    print(f"  [cam] Captured: {out_path}")
    return out_path


def scan_medication_label(image_path: str) -> dict:
    """Full pipeline: load → find label → preprocess → OCR → parse.

    Returns the same format as vision_parser() in pillvault_agent.py.
    """
    image = cv2.imread(image_path)
    if image is None:
        return {
            "medication_name": "",
            "dosage_mg": "",
            "rx_number": "",
            "total_quantity": 0,
            "expiration_date": "",
            "doctor_name": "",
        }

    original = image.copy()

    label = find_label_region(image)
    processed = preprocess_image(label)
    processed = deskew(processed)

    raw_text = extract_text(processed)
    if not raw_text:
        # Retry on original image if label crop failed
        processed_orig = preprocess_image(original)
        processed_orig = deskew(processed_orig)
        raw_text = extract_text(processed_orig)

    result = parse_medication_text(raw_text)
    return result
