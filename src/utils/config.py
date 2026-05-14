"""
config.py — Central configuration for Medical Document Intelligence System
"""

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# ── Base Paths ────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
LOGS_DIR = BASE_DIR / "logs"
PROCESSED_DIR = DATA_DIR / "processed"
RAW_DIR = DATA_DIR / "raw_documents"
SYNTHETIC_DIR = DATA_DIR / "synthetic_annotations"

# Ensure directories exist
for _dir in [LOGS_DIR, PROCESSED_DIR, MODELS_DIR]:
    _dir.mkdir(parents=True, exist_ok=True)

# ── OCR Settings ──────────────────────────────────────────────────────────────
OCR_CONFIDENCE_THRESHOLD = float(os.getenv("OCR_CONFIDENCE_THRESHOLD", 0.6))
OCR_LANGUAGES = os.getenv("OCR_LANGUAGES", "en,hi,mr").split(",")
TESSERACT_LANG_STRING = os.getenv("TESSERACT_LANG", "eng+hin+mar")
TESSERACT_CMD = os.getenv("TESSERACT_CMD", r"C:\Program Files\Tesseract-OCR\tesseract.exe")

# ── Model Names (HuggingFace Hub) ─────────────────────────────────────────────
NER_MODEL_NAME = os.getenv("NER_MODEL_NAME", "dslim/bert-base-NER")
CLASSIFIER_MODEL_NAME = os.getenv("CLASSIFIER_MODEL_NAME", "distilbert-base-uncased")
CORRECTION_MODEL_NAME = os.getenv("CORRECTION_MODEL_NAME", "dmis-lab/biobert-base-cased-v1.2")
TRANSLATION_EN_HI_MODEL = os.getenv("TRANSLATION_EN_HI_MODEL", "Helsinki-NLP/opus-mt-en-hi")
TRANSLATION_EN_MR_MODEL = os.getenv("TRANSLATION_EN_MR_MODEL", "Helsinki-NLP/opus-mt-en-mr")

# ── Local Saved Model Paths ───────────────────────────────────────────────────
NER_MODEL_PATH = MODELS_DIR / "ner_model"
CLASSIFIER_MODEL_PATH = MODELS_DIR / "classifier_model"
TFIDF_SVM_MODEL_PATH = MODELS_DIR / "tfidf_svm_classifier.joblib"
TFIDF_VECTORIZER_PATH = MODELS_DIR / "tfidf_vectorizer.joblib"

# ── Document Classification Labels ───────────────────────────────────────────
DOCUMENT_CLASSES = [
    "Prescription",
    "Lab Report",
    "Discharge Summary",
    "Medical Bill",
    "Diagnostic Scan Report",
]

# ── NER Label Set ─────────────────────────────────────────────────────────────
NER_LABELS = [
    "O",
    "B-MEDICINE_NAME", "I-MEDICINE_NAME",
    "B-DOSAGE", "I-DOSAGE",
    "B-FREQUENCY", "I-FREQUENCY",
    "B-DURATION", "I-DURATION",
    "B-ROUTE", "I-ROUTE",
    "B-TEST_NAME", "I-TEST_NAME",
    "B-TEST_VALUE", "I-TEST_VALUE",
    "B-UNIT", "I-UNIT",
    "B-NORMAL_RANGE", "I-NORMAL_RANGE",
    "B-ABNORMAL_FLAG", "I-ABNORMAL_FLAG",
    "B-DIAGNOSIS", "I-DIAGNOSIS",
    "B-SYMPTOM", "I-SYMPTOM",
    "B-PROCEDURE", "I-PROCEDURE",
    "B-DOCTOR_NAME", "I-DOCTOR_NAME",
    "B-HOSPITAL_NAME", "I-HOSPITAL_NAME",
    "B-DATE", "I-DATE",
    "B-PATIENT_NAME", "I-PATIENT_NAME",
]

NER_LABEL2ID = {label: idx for idx, label in enumerate(NER_LABELS)}
NER_ID2LABEL = {idx: label for label, idx in NER_LABEL2ID.items()}

# ── Training Hyperparameters ──────────────────────────────────────────────────
NER_TRAIN_EPOCHS = int(os.getenv("NER_TRAIN_EPOCHS", 5))
NER_BATCH_SIZE = int(os.getenv("NER_BATCH_SIZE", 16))
NER_LR = float(os.getenv("NER_LR", 5e-5))
NER_MAX_LENGTH = int(os.getenv("NER_MAX_LENGTH", 256))

CLASSIFIER_TRAIN_EPOCHS = int(os.getenv("CLASSIFIER_TRAIN_EPOCHS", 5))
CLASSIFIER_BATCH_SIZE = int(os.getenv("CLASSIFIER_BATCH_SIZE", 16))
CLASSIFIER_LR = float(os.getenv("CLASSIFIER_LR", 2e-5))
CLASSIFIER_MAX_LENGTH = int(os.getenv("CLASSIFIER_MAX_LENGTH", 256))

# ── Translation Settings ──────────────────────────────────────────────────────
SUPPORTED_LANGUAGES = {"hindi": "hi", "marathi": "mr", "english": "en"}
MEDICAL_TERMS_TO_PRESERVE = {
    "Paracetamol", "Amoxicillin", "CBC", "HbA1c", "BP", "ECG", "MRI", "CT",
    "COVID", "RT-PCR", "ICU", "OPD", "IPD", "IV", "IM", "SC", "OD", "BD",
    "TDS", "SOS", "Tab", "Cap", "Inj", "Syp", "mg", "ml", "mcg",
}

# ── API Settings ──────────────────────────────────────────────────────────────
API_HOST = os.getenv("API_HOST", "0.0.0.0")
API_PORT = int(os.getenv("API_PORT", 8000))
API_TITLE = "Medical Document Intelligence API"
API_VERSION = "1.0.0"
API_DESCRIPTION = (
    "End-to-end AI system for healthcare document OCR, entity extraction, "
    "classification, translation, and patient-friendly explanation."
)

# ── Misc ──────────────────────────────────────────────────────────────────────
MAX_FILE_SIZE_MB = int(os.getenv("MAX_FILE_SIZE_MB", 20))
ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".pdf", ".tiff", ".bmp"}
