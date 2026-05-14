"""
ocr_ensemble.py — Confidence-weighted ensemble of Tesseract + EasyOCR engines
"""

from typing import Dict, Any, Union
from pathlib import Path
import numpy as np

from src.ocr.tesseract_engine import TesseractEngine
from src.ocr.easyocr_engine import EasyOCREngine
from src.utils.config import OCR_CONFIDENCE_THRESHOLD
from src.utils.logger import log


class OCREnsemble:
    """
    Runs Tesseract and EasyOCR, selects best result by confidence.
    Falls back to second engine if primary confidence is below threshold.
    """

    def __init__(self, primary: str = "easyocr", gpu: bool = False):
        self.tesseract = TesseractEngine()
        self.easyocr = EasyOCREngine(gpu=gpu)
        self.primary = primary
        self.threshold = OCR_CONFIDENCE_THRESHOLD

    def extract(
        self,
        image: Union[np.ndarray, str, Path],
        force_both: bool = False,
    ) -> Dict[str, Any]:
        """
        Extract text using ensemble strategy.

        Args:
            image: preprocessed image
            force_both: if True, always run both engines

        Returns:
            Best result dict with added 'engine_used' and 'all_results' keys.
        """
        results = {}

        # Run primary engine first
        if self.primary == "easyocr":
            primary_result = self.easyocr.extract(image)
            results["easyocr"] = primary_result
        else:
            primary_result = self.tesseract.extract(image)
            results["tesseract"] = primary_result

        log.info(
            f"Primary ({self.primary}) confidence: {primary_result['confidence']:.3f}"
        )

        # Run secondary if confidence is low OR forced
        if primary_result["confidence"] < self.threshold or force_both:
            log.info("Low confidence — running secondary OCR engine...")
            if self.primary == "easyocr":
                secondary_result = self.tesseract.extract(image)
                results["tesseract"] = secondary_result
            else:
                secondary_result = self.easyocr.extract(image)
                results["easyocr"] = secondary_result

            log.info(
                f"Secondary confidence: {secondary_result['confidence']:.3f}"
            )
            best = self._pick_best(primary_result, secondary_result)
        else:
            best = primary_result

        best["all_results"] = results
        best["engine_used"] = best.get("engine", self.primary)
        return best

    def _pick_best(self, a: Dict, b: Dict) -> Dict:
        """Choose result with higher confidence and more text content."""
        score_a = a["confidence"] * max(a["word_count"], 1)
        score_b = b["confidence"] * max(b["word_count"], 1)
        winner = a if score_a >= score_b else b
        log.info(
            f"Ensemble winner: {winner['engine']} "
            f"(score_a={score_a:.3f}, score_b={score_b:.3f})"
        )
        return winner

    def compare(self, image: Union[np.ndarray, str, Path]) -> Dict[str, Any]:
        """Run both engines and return comparison metrics."""
        t_res = self.tesseract.extract(image)
        e_res = self.easyocr.extract(image)
        return {
            "tesseract": {
                "confidence": t_res["confidence"],
                "word_count": t_res["word_count"],
                "text_preview": t_res["text"][:200],
            },
            "easyocr": {
                "confidence": e_res["confidence"],
                "word_count": e_res["word_count"],
                "text_preview": e_res["text"][:200],
            },
            "winner": "easyocr" if e_res["confidence"] > t_res["confidence"] else "tesseract",
        }
