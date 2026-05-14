"""
tesseract_engine.py — Tesseract OCR wrapper for medical documents
"""

import os
from pathlib import Path
from typing import Dict, Any, List, Union
import numpy as np
import pytesseract
from PIL import Image

from src.utils.config import TESSERACT_CMD, TESSERACT_LANG_STRING
from src.utils.logger import log

# Configure Tesseract path
if os.path.exists(TESSERACT_CMD):
    pytesseract.pytesseract.tesseract_cmd = TESSERACT_CMD


class TesseractEngine:
    """Wrapper around pytesseract for medical document OCR."""

    def __init__(self, lang: str = TESSERACT_LANG_STRING, config: str = ""):
        self.lang = lang
        self.config = config or "--oem 3 --psm 6"

    def extract(self, image: Union[np.ndarray, Image.Image, str, Path]) -> Dict[str, Any]:
        """
        Run Tesseract OCR and return structured output.
        Returns: {text, confidence, words, bounding_boxes}
        """
        pil_img = self._to_pil(image)

        try:
            # Full text
            text = pytesseract.image_to_string(
                pil_img, lang=self.lang, config=self.config
            ).strip()

            # Word-level data with confidence
            data = pytesseract.image_to_data(
                pil_img, lang=self.lang, config=self.config,
                output_type=pytesseract.Output.DICT
            )

            words, boxes, confidences = [], [], []
            for i, word in enumerate(data["text"]):
                conf = int(data["conf"][i])
                if conf > 0 and word.strip():
                    words.append(word)
                    confidences.append(conf)
                    boxes.append({
                        "word": word,
                        "x": data["left"][i],
                        "y": data["top"][i],
                        "w": data["width"][i],
                        "h": data["height"][i],
                        "confidence": conf,
                    })

            avg_conf = float(np.mean(confidences)) / 100.0 if confidences else 0.0

            log.info(f"Tesseract OCR: {len(words)} words, confidence={avg_conf:.2f}")
            return {
                "engine": "tesseract",
                "text": text,
                "confidence": round(avg_conf, 4),
                "word_count": len(words),
                "bounding_boxes": boxes,
            }

        except Exception as e:
            log.error(f"Tesseract OCR failed: {e}")
            return {"engine": "tesseract", "text": "", "confidence": 0.0,
                    "word_count": 0, "bounding_boxes": [], "error": str(e)}

    def _to_pil(self, image: Any) -> Image.Image:
        if isinstance(image, Image.Image):
            return image
        if isinstance(image, np.ndarray):
            if len(image.shape) == 2:
                return Image.fromarray(image, mode="L")
            import cv2
            return Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        return Image.open(str(image)).convert("RGB")
