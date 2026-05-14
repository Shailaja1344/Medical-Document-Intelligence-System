"""
spell_corrector.py — Post-OCR medical text correction engine
"""

import re
from typing import List, Tuple, Optional
from spellchecker import SpellChecker
from fuzzywuzzy import process as fuzz_process

from src.correction.medical_abbreviations import (
    normalize_abbreviation, apply_ocr_corrections, ALL_ABBREVIATIONS
)
from src.utils.logger import log

# ── Medical word list (extends general dictionary) ────────────────────────────
MEDICAL_WORD_LIST = [
    "paracetamol", "amoxicillin", "azithromycin", "metformin", "atorvastatin",
    "omeprazole", "pantoprazole", "cetirizine", "montelukast", "salbutamol",
    "prednisolone", "dexamethasone", "ibuprofen", "diclofenac", "cefixime",
    "ciprofloxacin", "doxycycline", "metronidazole", "albendazole", "ivermectin",
    "amlodipine", "enalapril", "ramipril", "losartan", "telmisartan",
    "glimepiride", "sitagliptin", "vildagliptin", "empagliflozin", "insulin",
    "levothyroxine", "carbimazole", "hydrochlorothiazide", "furosemide",
    "spironolactone", "aspirin", "clopidogrel", "warfarin", "heparin",
    "esomeprazole", "ranitidine", "ondansetron", "domperidone", "metoclopramide",
    "multivitamin", "calcium", "vitamin", "ferrous", "folic", "zinc",
    "hemoglobin", "platelet", "leukocyte", "erythrocyte", "creatinine",
    "bilirubin", "cholesterol", "triglyceride", "glucose", "urea", "uric",
    "hypertension", "diabetes", "fever", "infection", "inflammation",
    "bronchitis", "pneumonia", "tuberculosis", "dengue", "malaria", "typhoid",
    "gastritis", "ulcer", "appendicitis", "cholecystitis", "pancreatitis",
    "nephritis", "urethritis", "cystitis", "pyelonephritis", "tonsillitis",
    "pharyngitis", "sinusitis", "otitis", "conjunctivitis", "dermatitis",
    "prescription", "diagnosis", "symptoms", "dosage", "frequency", "duration",
    "discharge", "admission", "surgery", "biopsy", "endoscopy", "colonoscopy",
]


class MedicalSpellCorrector:
    """
    Multi-stage OCR post-correction for medical text.

    Correction priority:
    1. Known OCR misread corrections (exact string replace)
    2. Medical abbreviation normalization
    3. Fuzzy matching against medical word list
    4. General spell checker (pyspellchecker)
    5. Optional BioBERT masked token suggestion (lazy)
    """

    def __init__(self, use_biobert: bool = False):
        self.use_biobert = use_biobert
        self._spell = SpellChecker()
        self._spell.word_frequency.load_words(MEDICAL_WORD_LIST)
        self._medical_set = set(MEDICAL_WORD_LIST)
        self._biobert_pipe = None
        log.info("MedicalSpellCorrector initialized")

    def correct(self, text: str) -> str:
        """Full correction pipeline — returns corrected text string."""
        if not text or not text.strip():
            return text

        # Stage 1: Known OCR corrections
        text = apply_ocr_corrections(text)

        # Stage 2: Abbreviation normalization
        text = normalize_abbreviation(text)

        # Stage 3: Word-level correction
        text = self._correct_words(text)

        return text.strip()

    def _correct_words(self, text: str) -> str:
        """Tokenize and correct each word."""
        tokens = re.findall(r'[\w.]+|[^\w\s]|\s+', text)
        corrected = []
        for tok in tokens:
            word = tok.strip()
            if not word or not word.isalpha() or len(word) < 3:
                corrected.append(tok)
                continue
            corrected.append(self._correct_token(word))
        return "".join(corrected)

    def _correct_token(self, word: str) -> str:
        """Correct a single word token."""
        lower = word.lower()

        # Skip if already a known medical term
        if lower in self._medical_set:
            return word

        # Fuzzy match against medical word list
        match, score = fuzz_process.extractOne(lower, MEDICAL_WORD_LIST)
        if score >= 90:
            log.debug(f"Fuzzy correction: {word} → {match} (score={score})")
            return match

        # General spell check
        corrected = self._spell.correction(lower)
        if corrected and corrected != lower:
            log.debug(f"Spell correction: {word} → {corrected}")
            return corrected

        return word

    def _load_biobert(self):
        """Lazy-load BioBERT fill-mask pipeline."""
        if self._biobert_pipe is None:
            from transformers import pipeline
            from src.utils.config import CORRECTION_MODEL_NAME
            log.info("Loading BioBERT fill-mask pipeline...")
            self._biobert_pipe = pipeline(
                "fill-mask",
                model=CORRECTION_MODEL_NAME,
                top_k=3,
            )
        return self._biobert_pipe

    def biobert_suggest(self, masked_text: str) -> List[Tuple[str, float]]:
        """
        Get BioBERT suggestions for a [MASK] token in text.
        Returns list of (token, score) tuples.
        """
        pipe = self._load_biobert()
        results = pipe(masked_text)
        return [(r["token_str"], round(r["score"], 4)) for r in results]

    def batch_correct(self, texts: List[str]) -> List[str]:
        """Correct a list of texts."""
        return [self.correct(t) for t in texts]
