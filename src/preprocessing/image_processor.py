"""
image_processor.py — Medical document image preprocessing pipeline
"""

import cv2
import numpy as np
from pathlib import Path
from typing import Union, Tuple
from PIL import Image, ImageEnhance, ImageFilter

from src.utils.logger import log


class ImageProcessor:
    """
    Preprocessing pipeline for medical document images.
    Handles scanned prescriptions, handwritten notes, and printed reports.
    """

    def __init__(
        self,
        target_dpi: int = 300,
        apply_deskew: bool = True,
        apply_denoise: bool = True,
        apply_threshold: bool = True,
        sharpen_handwriting: bool = True,
    ):
        self.target_dpi = target_dpi
        self.apply_deskew = apply_deskew
        self.apply_denoise = apply_denoise
        self.apply_threshold = apply_threshold
        self.sharpen_handwriting = sharpen_handwriting

    def preprocess(self, image_input: Union[str, Path, np.ndarray]) -> np.ndarray:
        """
        Full preprocessing pipeline.
        Returns: preprocessed grayscale numpy array ready for OCR.
        """
        img = self._load_image(image_input)
        log.debug(f"Loaded image: shape={img.shape}")

        img = self._to_grayscale(img)
        img = self._enhance_contrast(img)

        if self.apply_denoise:
            img = self._denoise(img)

        if self.apply_deskew:
            img = self._deskew(img)

        if self.sharpen_handwriting:
            img = self._sharpen(img)

        if self.apply_threshold:
            img = self._adaptive_threshold(img)

        log.debug(f"Preprocessing complete: shape={img.shape}")
        return img

    # ── Private Methods ───────────────────────────────────────────────────────

    def _load_image(self, source: Union[str, Path, np.ndarray]) -> np.ndarray:
        if isinstance(source, np.ndarray):
            return source.copy()
        path = str(source)
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError(f"Cannot read image: {path}")
        return img

    def _to_grayscale(self, img: np.ndarray) -> np.ndarray:
        if len(img.shape) == 3:
            return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return img

    def _enhance_contrast(self, img: np.ndarray) -> np.ndarray:
        """CLAHE contrast enhancement — improves faded scan text."""
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        return clahe.apply(img)

    def _denoise(self, img: np.ndarray) -> np.ndarray:
        """Bilateral filter preserves edges while removing noise."""
        return cv2.bilateralFilter(img, d=9, sigmaColor=75, sigmaSpace=75)

    def _deskew(self, img: np.ndarray) -> np.ndarray:
        """
        Deskew using Hough line transform to detect dominant text angle.
        Corrects rotation up to ±15 degrees.
        """
        try:
            edges = cv2.Canny(img, 50, 150, apertureSize=3)
            lines = cv2.HoughLines(edges, 1, np.pi / 180, threshold=100)
            if lines is None:
                return img

            angles = []
            for line in lines[:20]:
                rho, theta = line[0]
                angle = np.degrees(theta) - 90
                if -15 < angle < 15:
                    angles.append(angle)

            if not angles:
                return img

            median_angle = float(np.median(angles))
            h, w = img.shape[:2]
            M = cv2.getRotationMatrix2D((w // 2, h // 2), median_angle, 1.0)
            return cv2.warpAffine(
                img, M, (w, h), flags=cv2.INTER_CUBIC,
                borderMode=cv2.BORDER_REPLICATE
            )
        except Exception as e:
            log.warning(f"Deskew failed: {e}")
            return img

    def _sharpen(self, img: np.ndarray) -> np.ndarray:
        """Unsharp mask for handwriting clarity."""
        kernel = np.array([
            [0, -1,  0],
            [-1,  5, -1],
            [0, -1,  0]
        ], dtype=np.float32)
        return cv2.filter2D(img, -1, kernel)

    def _adaptive_threshold(self, img: np.ndarray) -> np.ndarray:
        """
        Adaptive thresholding — handles uneven illumination in scans.
        Uses Gaussian adaptive method for medical documents.
        """
        return cv2.adaptiveThreshold(
            img, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            blockSize=31,
            C=10,
        )

    # ── Utility Methods ───────────────────────────────────────────────────────

    @staticmethod
    def load_from_pdf(pdf_path: Union[str, Path], page: int = 0) -> np.ndarray:
        """Convert a PDF page to a numpy image array."""
        try:
            from pdf2image import convert_from_path
            pages = convert_from_path(str(pdf_path), dpi=300)
            if page >= len(pages):
                raise ValueError(f"Page {page} out of range (total: {len(pages)})")
            pil_img = pages[page]
            return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
        except ImportError:
            raise ImportError("pdf2image not installed. Run: pip install pdf2image")

    @staticmethod
    def numpy_to_pil(img: np.ndarray) -> Image.Image:
        if len(img.shape) == 2:
            return Image.fromarray(img, mode="L")
        return Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

    @staticmethod
    def save_processed(img: np.ndarray, output_path: Union[str, Path]) -> None:
        cv2.imwrite(str(output_path), img)
        log.info(f"Saved preprocessed image to {output_path}")
