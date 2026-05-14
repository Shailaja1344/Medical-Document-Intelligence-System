"""
easyocr_engine.py — EasyOCR wrapper for medical documents
"""

from typing import Dict, Any, List, Union
from pathlib import Path
import numpy as np

from src.utils.config import OCR_LANGUAGES
from src.utils.logger import log


class EasyOCREngine:
    """Wrapper around EasyOCR for multilingual medical document OCR."""

    def __init__(self, languages: List[str] = None, gpu: bool = False):
        self.languages = languages or ["en", "hi"]
        self.gpu = gpu
        self._reader = None

    def _get_reader(self):
        """Lazy-load EasyOCR reader (model download on first call)."""
        if self._reader is None:
            import easyocr
            log.info(f"Loading EasyOCR for languages: {self.languages}")
            self._reader = easyocr.Reader(self.languages, gpu=self.gpu)
        return self._reader

    def extract(self, image: Union[np.ndarray, str, Path]) -> Dict[str, Any]:
        """
        Run EasyOCR and return structured output.
        Returns: {text, confidence, bounding_boxes}
        """
        reader = self._get_reader()

        if isinstance(image, Path):
            image = str(image)

        try:
            results = reader.readtext(image, detail=1, paragraph=False)

            lines, boxes, confidences = [], [], []
            for (bbox, text, conf) in results:
                if text.strip():
                    lines.append(text)
                    confidences.append(conf)
                    boxes.append({
                        "word": text,
                        "bbox": bbox,
                        "confidence": round(conf, 4),
                    })

            full_text = "\n".join(lines)
            avg_conf = float(np.mean(confidences)) if confidences else 0.0

            log.info(f"EasyOCR: {len(lines)} lines, confidence={avg_conf:.2f}")
            return {
                "engine": "easyocr",
                "text": full_text,
                "confidence": round(avg_conf, 4),
                "word_count": len(lines),
                "bounding_boxes": boxes,
            }

        except Exception as e:
            log.error(f"EasyOCR failed: {e}")
            return {"engine": "easyocr", "text": "", "confidence": 0.0,
                    "word_count": 0, "bounding_boxes": [], "error": str(e)}
