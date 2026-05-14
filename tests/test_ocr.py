"""
test_ocr.py — Unit tests for OCR engines and preprocessing
"""

import sys
import numpy as np
import pytest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parents[1]))


def make_blank_image(h=200, w=400, text_like=True):
    """Create a simple white image with black border for testing."""
    img = np.ones((h, w), dtype=np.uint8) * 255
    img[10:20, 10:390] = 0  # fake text line
    return img


class TestImageProcessor:
    def test_grayscale_conversion(self):
        from src.preprocessing.image_processor import ImageProcessor
        processor = ImageProcessor(apply_deskew=False, apply_denoise=False,
                                   apply_threshold=False, sharpen_handwriting=False)
        color_img = np.random.randint(0, 255, (100, 200, 3), dtype=np.uint8)
        result = processor.preprocess(color_img)
        assert result.ndim == 2, "Output should be grayscale (2D)"

    def test_preprocessed_shape(self):
        from src.preprocessing.image_processor import ImageProcessor
        processor = ImageProcessor()
        img = np.ones((300, 500, 3), dtype=np.uint8) * 200
        result = processor.preprocess(img)
        assert result.shape[0] == 300
        assert result.shape[1] == 500

    def test_invalid_path_raises(self):
        from src.preprocessing.image_processor import ImageProcessor
        processor = ImageProcessor()
        with pytest.raises(ValueError):
            processor._load_image("nonexistent_file.png")


class TestOCREnsemble:
    def test_ensemble_returns_dict(self):
        """Ensemble should return a dict with required keys even with blank image."""
        from src.ocr.ocr_ensemble import OCREnsemble
        ensemble = OCREnsemble()
        img = make_blank_image()
        result = ensemble.ocr.tesseract.extract(img)
        assert "text" in result
        assert "confidence" in result
        assert "engine" in result

    def test_confidence_in_range(self):
        from src.ocr.tesseract_engine import TesseractEngine
        engine = TesseractEngine()
        img = make_blank_image()
        result = engine.extract(img)
        assert 0.0 <= result["confidence"] <= 1.0


class TestMetrics:
    def test_cer_identical(self):
        from src.utils.metrics import compute_cer
        assert compute_cer("hello", "hello") == 0.0

    def test_wer_identical(self):
        from src.utils.metrics import compute_wer
        assert compute_wer("hello world", "hello world") == 0.0

    def test_wer_different(self):
        from src.utils.metrics import compute_wer
        wer = compute_wer("hello world", "hello")
        assert wer > 0.0

    def test_cer_partial(self):
        from src.utils.metrics import compute_cer
        cer = compute_cer("Paracetamol", "Paraeetamol")
        assert 0.0 < cer < 1.0
