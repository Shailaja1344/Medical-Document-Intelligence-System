"""
routes.py — FastAPI route definitions for Medical Document Intelligence API
"""

import io
import time
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, File, UploadFile, HTTPException, Form, Depends
from fastapi.responses import JSONResponse
import numpy as np

from src.api.schemas import (
    TextRequest, TranslateRequest, ExplainRequest,
    OCRResponse, ClassificationResponse, EntityExtractionResponse,
    TranslationResponse, ExplanationResponse, FullPipelineResponse,
    HealthResponse,
)
from src.utils.config import ALLOWED_EXTENSIONS, API_VERSION
from src.utils.logger import log

router = APIRouter()

# Lazy singleton pipeline (loaded on first request)
_pipeline = None


def get_pipeline():
    global _pipeline
    if _pipeline is None:
        from src.pipeline.orchestrator import MedicalDocumentPipeline
        _pipeline = MedicalDocumentPipeline()
    return _pipeline


def _read_upload_image(file: UploadFile) -> np.ndarray:
    """Read an uploaded file into a numpy image array."""
    import cv2
    suffix = Path(file.filename or "").suffix.lower()
    content = file.file.read()

    if suffix == ".pdf":
        from src.preprocessing.image_processor import ImageProcessor
        import tempfile, os
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
            tmp.write(content)
            tmp_path = tmp.name
        try:
            img = ImageProcessor.load_from_pdf(tmp_path)
        finally:
            os.unlink(tmp_path)
        return img

    arr = np.frombuffer(content, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(status_code=400, detail="Cannot decode uploaded image")
    return img


# ── Health Check ──────────────────────────────────────────────────────────────

@router.get("/health", response_model=HealthResponse, tags=["System"])
async def health_check():
    """API health check endpoint."""
    return HealthResponse(
        status="healthy",
        version=API_VERSION,
        modules={
            "ocr": "EasyOCR + Tesseract ensemble",
            "classifier": "TF-IDF SVM + DistilBERT",
            "ner": "BioBERT token classification",
            "translation": "Helsinki-NLP MarianMT",
            "explanation": "Rule-based medical dictionary",
        },
    )


# ── OCR Extract ───────────────────────────────────────────────────────────────

@router.post("/ocr_extract", response_model=OCRResponse, tags=["OCR"])
async def ocr_extract(file: UploadFile = File(...)):
    """
    Extract raw text from a medical document image or PDF using the OCR ensemble.
    Returns text, confidence score, and word-level bounding boxes.
    """
    log.info(f"OCR request: {file.filename}")
    try:
        img = _read_upload_image(file)
        pipeline = get_pipeline()
        preprocessed = pipeline.preprocessor.preprocess(img)
        result = pipeline.ocr.extract(preprocessed)
        return OCRResponse(**{
            k: result.get(k, "") for k in
            ["engine", "text", "confidence", "word_count", "bounding_boxes"]
        })
    except Exception as e:
        log.error(f"OCR failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ── Classify Document ─────────────────────────────────────────────────────────

@router.post("/classify_document", response_model=ClassificationResponse, tags=["Classification"])
async def classify_document(request: TextRequest):
    """
    Classify document type: Prescription, Lab Report, Discharge Summary, etc.
    Accepts extracted or raw text.
    """
    try:
        pipeline = get_pipeline()
        pipeline._ensure_classifier()
        result = pipeline.classifier.predict(request.text)
        return ClassificationResponse(**result)
    except Exception as e:
        log.error(f"Classification failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ── Extract Entities ──────────────────────────────────────────────────────────

@router.post("/extract_entities", response_model=EntityExtractionResponse, tags=["NER"])
async def extract_entities(request: TextRequest):
    """
    Extract clinical entities: medicines, diagnoses, lab tests, dates, etc.
    Uses BioBERT-based token classification.
    """
    t = time.perf_counter()
    try:
        pipeline = get_pipeline()
        entities = pipeline.extractor.extract(request.text)
        elapsed = round(time.perf_counter() - t, 3)
        return EntityExtractionResponse(
            entities={k: v for k, v in entities.items()},
            entity_count=sum(len(v) for v in entities.values()),
            processing_time_seconds=elapsed,
        )
    except Exception as e:
        log.error(f"NER failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ── Translate ─────────────────────────────────────────────────────────────────

@router.post("/translate/{lang}", response_model=TranslationResponse, tags=["Translation"])
async def translate_text(lang: str, request: TranslateRequest):
    """
    Translate medical text to Hindi ('hindi' / 'hi') or Marathi ('marathi' / 'mr').
    Medical terminology is preserved during translation.
    """
    t = time.perf_counter()
    lang_map = {"hi": "hi", "hindi": "hi", "mr": "mr", "marathi": "mr"}
    lang_code = lang_map.get(lang.lower())
    if not lang_code:
        raise HTTPException(status_code=400, detail=f"Unsupported language: {lang}")
    try:
        pipeline = get_pipeline()
        translated = pipeline.translator.translate(request.text, lang_code)
        return TranslationResponse(
            original=request.text,
            translated=translated,
            target_lang=lang,
            processing_time_seconds=round(time.perf_counter() - t, 3),
        )
    except Exception as e:
        log.error(f"Translation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ── Patient Explanation ───────────────────────────────────────────────────────

@router.post("/explain", response_model=ExplanationResponse, tags=["Explanation"])
async def explain_term(request: ExplainRequest):
    """
    Generate a patient-friendly explanation for a medical term, abbreviation, or diagnosis.
    """
    try:
        pipeline = get_pipeline()
        explanation = pipeline.explainer.explain_single_term(request.text)
        return ExplanationResponse(explanation=explanation, term=request.text)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ── Full Pipeline ─────────────────────────────────────────────────────────────

@router.post("/full_pipeline", tags=["Pipeline"])
async def full_pipeline(file: UploadFile = File(...)):
    """
    Run the complete end-to-end pipeline on an uploaded medical document.

    Returns:
    - OCR extracted text
    - Corrected text
    - Document type classification
    - Clinical entity extraction
    - Hindi translation
    - Marathi translation
    - Patient-friendly explanation
    - Processing timings
    """
    log.info(f"Full pipeline request: {file.filename}")
    try:
        img = _read_upload_image(file)
        pipeline = get_pipeline()
        preprocessed = pipeline.preprocessor.preprocess(img)
        result = pipeline.process_image(preprocessed)
        return JSONResponse(content=result)
    except Exception as e:
        log.error(f"Full pipeline failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ── Upload Document (alias with preprocessing info) ───────────────────────────

@router.post("/upload_document", tags=["OCR"])
async def upload_document(file: UploadFile = File(...)):
    """
    Upload a document for OCR + basic preprocessing.
    Returns raw OCR output and preprocessing metadata.
    """
    log.info(f"Upload request: {file.filename}")
    try:
        img = _read_upload_image(file)
        pipeline = get_pipeline()
        preprocessed = pipeline.preprocessor.preprocess(img)
        result = pipeline.ocr.extract(preprocessed)
        result["filename"] = file.filename
        result["original_shape"] = list(img.shape)
        result["preprocessed_shape"] = list(preprocessed.shape)
        return JSONResponse(content=result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
