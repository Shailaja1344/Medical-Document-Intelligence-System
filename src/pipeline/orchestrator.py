"""
orchestrator.py — Full pipeline orchestrator chaining all modules
"""

import time
from pathlib import Path
from typing import Any, Dict, Optional, Union
import numpy as np

from src.preprocessing.image_processor import ImageProcessor
from src.ocr.ocr_ensemble import OCREnsemble
from src.correction.spell_corrector import MedicalSpellCorrector
from src.classifier.tfidf_svm_classifier import TFIDFSVMClassifier
from src.ner.entity_extractor import EntityExtractor
from src.translation.translator import MedicalTranslator
from src.explanation.explainer import PatientExplainer
from src.utils.json_structurer import build_structured_json
from src.utils.logger import log
from src.utils.config import ALLOWED_EXTENSIONS, MAX_FILE_SIZE_MB


class MedicalDocumentPipeline:
    """
    End-to-end pipeline for medical document intelligence.

    Stages:
        1. Image preprocessing (if image/PDF input)
        2. OCR ensemble (Tesseract + EasyOCR)
        3. OCR error correction
        4. Document type classification
        5. Clinical NER entity extraction
        6. JSON structuring
        7. Multilingual translation (Hindi + Marathi)
        8. Patient-friendly explanation generation
    """

    def __init__(self, gpu: bool = False):
        log.info("Initializing Medical Document Pipeline...")
        self.preprocessor = ImageProcessor()
        self.ocr = OCREnsemble(gpu=gpu)
        self.corrector = MedicalSpellCorrector()
        self.classifier = TFIDFSVMClassifier()
        self.extractor = EntityExtractor()
        self.translator = MedicalTranslator()
        self.explainer = PatientExplainer()
        self._classifier_loaded = False
        log.info("Pipeline initialized successfully")

    def _ensure_classifier(self):
        if not self._classifier_loaded:
            try:
                self.classifier.load()
                self._classifier_loaded = True
            except FileNotFoundError:
                log.warning(
                    "No trained classifier found — using keyword-based fallback"
                )

    def process_file(
        self,
        file_path: Union[str, Path],
        target_langs: list = None,
    ) -> Dict[str, Any]:
        """
        Process a file (image/PDF) through the full pipeline.

        Args:
            file_path: path to medical document image or PDF
            target_langs: list of target languages, e.g. ['hindi', 'marathi']

        Returns:
            Full structured intelligence JSON.
        """
        path = Path(file_path)
        suffix = path.suffix.lower()

        if suffix not in ALLOWED_EXTENSIONS:
            raise ValueError(f"Unsupported file type: {suffix}")

        file_size_mb = path.stat().st_size / (1024 * 1024)
        if file_size_mb > MAX_FILE_SIZE_MB:
            raise ValueError(
                f"File too large: {file_size_mb:.1f} MB (max {MAX_FILE_SIZE_MB} MB)"
            )

        if suffix == ".pdf":
            image = ImageProcessor.load_from_pdf(path)
        else:
            import cv2
            image = cv2.imread(str(path))

        preprocessed = self.preprocessor.preprocess(image)
        return self.process_image(preprocessed, target_langs=target_langs)

    def process_image(
        self,
        image: np.ndarray,
        target_langs: list = None,
    ) -> Dict[str, Any]:
        """Process a preprocessed numpy image array."""
        return self._run_pipeline(image=image, target_langs=target_langs)

    def process_text(
        self,
        text: str,
        target_langs: list = None,
    ) -> Dict[str, Any]:
        """
        Process raw text (skip OCR stage).
        Useful when text has already been extracted externally.
        """
        return self._run_pipeline(text=text, target_langs=target_langs)

    def _run_pipeline(
        self,
        image: Optional[np.ndarray] = None,
        text: Optional[str] = None,
        target_langs: list = None,
    ) -> Dict[str, Any]:
        """Internal pipeline runner."""
        timings = {}
        target_langs = target_langs or ["hindi", "marathi"]

        total_start = time.perf_counter()

        # ── Stage 1+2: OCR ────────────────────────────────────────────────────
        t = time.perf_counter()
        if image is not None:
            ocr_result = self.ocr.extract(image)
            raw_text = ocr_result["text"]
            ocr_confidence = ocr_result["confidence"]
            engine_used = ocr_result.get("engine_used", "unknown")
        else:
            raw_text = text or ""
            ocr_confidence = 1.0
            engine_used = "text_input"
            ocr_result = {
                "text": raw_text, "confidence": 1.0,
                "engine_used": "text_input", "bounding_boxes": []
            }
        timings["ocr_seconds"] = round(time.perf_counter() - t, 3)
        log.info(f"OCR stage: {len(raw_text)} chars, conf={ocr_confidence:.3f}")

        # ── Stage 3: OCR Correction ───────────────────────────────────────────
        t = time.perf_counter()
        corrected_text = self.corrector.correct(raw_text)
        timings["correction_seconds"] = round(time.perf_counter() - t, 3)

        # ── Stage 4: Classification ───────────────────────────────────────────
        t = time.perf_counter()
        self._ensure_classifier()
        try:
            clf_result = self.classifier.predict(corrected_text)
            doc_type = clf_result["document_type"]
            clf_confidence = clf_result["confidence"]
        except Exception as e:
            log.warning(f"Classification failed: {e} — using keyword fallback")
            doc_type = self._keyword_classify(corrected_text)
            clf_confidence = 0.5
            clf_result = {"document_type": doc_type, "confidence": clf_confidence}
        timings["classification_seconds"] = round(time.perf_counter() - t, 3)
        log.info(f"Document type: {doc_type} (conf={clf_confidence:.3f})")

        # ── Stage 5: NER ──────────────────────────────────────────────────────
        t = time.perf_counter()
        try:
            entities = self.extractor.extract(corrected_text)
        except Exception as e:
            log.error(f"NER extraction failed: {e}")
            entities = {}
        timings["ner_seconds"] = round(time.perf_counter() - t, 3)

        # ── Stage 6: JSON Structuring (partial, for translation input) ────────
        t = time.perf_counter()
        partial_json = build_structured_json(
            document_type=doc_type,
            ocr_text=raw_text,
            corrected_text=corrected_text,
            entities=entities,
        )
        timings["structuring_seconds"] = round(time.perf_counter() - t, 3)

        # ── Stage 7: Translation ──────────────────────────────────────────────
        t = time.perf_counter()
        summary_for_translation = self._build_summary(partial_json)
        translations = {}
        for lang in target_langs:
            lang_code = {"hindi": "hi", "marathi": "mr"}.get(lang, lang)
            translations[lang] = self.translator.translate(
                summary_for_translation, lang_code
            )
        timings["translation_seconds"] = round(time.perf_counter() - t, 3)

        # ── Stage 8: Patient Explanation ──────────────────────────────────────
        t = time.perf_counter()
        explanation = self.explainer.explain_document(partial_json)
        timings["explanation_seconds"] = round(time.perf_counter() - t, 3)

        # ── Final Assembly ────────────────────────────────────────────────────
        total_time = round(time.perf_counter() - total_start, 3)

        final = build_structured_json(
            document_type=doc_type,
            ocr_text=raw_text,
            corrected_text=corrected_text,
            entities=entities,
            translation_hindi=translations.get("hindi", ""),
            translation_marathi=translations.get("marathi", ""),
            patient_explanation=explanation,
            processing_time=total_time,
        )

        final["pipeline_metadata"] = {
            "ocr_engine": engine_used,
            "ocr_confidence": ocr_confidence,
            "classifier_confidence": clf_confidence,
            "stage_timings": timings,
            "total_processing_seconds": total_time,
        }

        log.info(f"Pipeline complete in {total_time:.3f}s")
        return final

    @staticmethod
    def _keyword_classify(text: str) -> str:
        """Simple keyword-based document classification fallback."""
        text_lower = text.lower()
        if any(w in text_lower for w in ["tab.", "cap.", "rx", "prescription", "dose", "dosage"]):
            return "Prescription"
        if any(w in text_lower for w in ["lab", "report", "test", "result", "blood", "urine", "hb"]):
            return "Lab Report"
        if any(w in text_lower for w in ["discharge", "admitted", "ward", "icu", "surgery"]):
            return "Discharge Summary"
        if any(w in text_lower for w in ["bill", "invoice", "amount", "payment", "charges"]):
            return "Medical Bill"
        if any(w in text_lower for w in ["scan", "mri", "ct", "xray", "x-ray", "ultrasound", "usg"]):
            return "Diagnostic Scan Report"
        return "Prescription"

    @staticmethod
    def _build_summary(doc: Dict[str, Any]) -> str:
        """Build a plain text summary from structured JSON for translation."""
        lines = []
        entities = doc.get("entities", {})

        if entities.get("diagnosis"):
            lines.append("Diagnosis: " + ", ".join(entities["diagnosis"]))
        if entities.get("symptoms"):
            lines.append("Symptoms: " + ", ".join(entities["symptoms"]))
        for med in entities.get("medicines", []):
            if med.get("name"):
                med_str = f"Medicine: {med['name']}"
                if med.get("dosage"):
                    med_str += f", {med['dosage']}"
                if med.get("frequency"):
                    med_str += f", {med['frequency']}"
                lines.append(med_str)
        for test in entities.get("lab_tests", []):
            if test.get("test_name"):
                lines.append(
                    f"Test: {test['test_name']} = "
                    f"{test.get('value', 'N/A')} {test.get('unit', '')}"
                )
        return ". ".join(lines) if lines else doc.get("corrected_text", "")[:500]
