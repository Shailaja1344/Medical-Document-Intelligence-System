"""
schemas.py — Pydantic request/response models for the FastAPI backend
"""

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


# ── Request Models ────────────────────────────────────────────────────────────

class TextRequest(BaseModel):
    text: str = Field(..., description="Medical text to process", min_length=1)


class TranslateRequest(BaseModel):
    text: str = Field(..., description="English text to translate")
    target_lang: str = Field(
        "hindi",
        description="Target language: 'hindi' or 'marathi'"
    )


class ExplainRequest(BaseModel):
    text: str = Field(..., description="Medical term or text to explain")


# ── Sub-Response Models ───────────────────────────────────────────────────────

class BoundingBox(BaseModel):
    word: str
    confidence: float
    x: Optional[int] = None
    y: Optional[int] = None
    w: Optional[int] = None
    h: Optional[int] = None


class OCRResponse(BaseModel):
    engine: str
    text: str
    confidence: float
    word_count: int
    bounding_boxes: List[Dict[str, Any]] = []


class ClassificationResponse(BaseModel):
    document_type: str
    confidence: float
    all_scores: Dict[str, float] = {}
    model: str


class MedicineEntity(BaseModel):
    name: Optional[str] = None
    dosage: Optional[str] = None
    frequency: Optional[str] = None
    duration: Optional[str] = None
    route: Optional[str] = "Oral"


class LabTestEntity(BaseModel):
    test_name: Optional[str] = None
    value: Optional[str] = None
    unit: Optional[str] = None
    normal_range: Optional[str] = None
    status: Optional[str] = "Normal"


class EntitiesBlock(BaseModel):
    patient_name: Optional[str] = None
    doctor_name: Optional[str] = None
    hospital_name: Optional[str] = None
    date: Optional[str] = None
    diagnosis: List[str] = []
    symptoms: List[str] = []
    procedures: List[str] = []
    medicines: List[MedicineEntity] = []
    lab_tests: List[LabTestEntity] = []


class PipelineMetadata(BaseModel):
    ocr_engine: str
    ocr_confidence: float
    classifier_confidence: float
    stage_timings: Dict[str, float] = {}
    total_processing_seconds: float


# ── Full Pipeline Response ────────────────────────────────────────────────────

class FullPipelineResponse(BaseModel):
    schema_version: str = "1.0.0"
    timestamp: str
    document_type: str
    ocr_text: str
    corrected_text: str
    entities: EntitiesBlock
    translation_hindi: str = ""
    translation_marathi: str = ""
    patient_explanation: str = ""
    processing_time_seconds: float
    validation_warnings: List[str] = []
    pipeline_metadata: Optional[PipelineMetadata] = None


class EntityExtractionResponse(BaseModel):
    entities: Dict[str, List[str]]
    entity_count: int
    processing_time_seconds: float


class TranslationResponse(BaseModel):
    original: str
    translated: str
    target_lang: str
    processing_time_seconds: float


class ExplanationResponse(BaseModel):
    explanation: str
    term: Optional[str] = None


class HealthResponse(BaseModel):
    status: str = "healthy"
    version: str
    modules: Dict[str, str] = {}
