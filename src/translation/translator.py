"""
translator.py — Multilingual translation service for medical content
Supports English → Hindi and English → Marathi via Helsinki-NLP MarianMT
"""

from typing import Dict, List, Optional
import re

from src.utils.config import (
    TRANSLATION_EN_HI_MODEL, TRANSLATION_EN_MR_MODEL,
    SUPPORTED_LANGUAGES, MEDICAL_TERMS_TO_PRESERVE
)
from src.utils.logger import log


class MedicalTranslator:
    """
    Translates structured medical content to Hindi and Marathi.
    Preserves medical terminology (drug names, test names, units).
    """

    def __init__(self):
        self._pipelines: Dict[str, any] = {}
        self._placeholder_map: Dict[str, str] = {}

    def _get_pipeline(self, lang_code: str):
        """Lazy-load translation pipeline for a language."""
        if lang_code not in self._pipelines:
            from transformers import pipeline as hf_pipeline
            model_map = {
                "hi": TRANSLATION_EN_HI_MODEL,
                "mr": TRANSLATION_EN_MR_MODEL,
            }
            model_name = model_map.get(lang_code)
            if not model_name:
                raise ValueError(f"Unsupported language code: {lang_code}")
            log.info(f"Loading translation model: {model_name}")
            self._pipelines[lang_code] = hf_pipeline(
                "translation",
                model=model_name,
                max_length=512,
            )
        return self._pipelines[lang_code]

    def translate(self, text: str, target_lang: str = "hi") -> str:
        """
        Translate text from English to target language.
        Medical terms are protected via placeholder substitution.

        Args:
            text: English source text
            target_lang: 'hi' (Hindi) or 'mr' (Marathi)

        Returns:
            Translated string with medical terms preserved.
        """
        if not text or not text.strip():
            return text

        lang_code = SUPPORTED_LANGUAGES.get(target_lang.lower(), target_lang)

        try:
            # Step 1: Replace medical terms with placeholders
            protected_text, placeholder_map = self._protect_terms(text)

            # Step 2: Split into sentences and translate
            sentences = self._split_sentences(protected_text)
            pipe = self._get_pipeline(lang_code)
            translated_parts = []
            for sent in sentences:
                if sent.strip():
                    result = pipe(sent.strip())
                    translated_parts.append(result[0]["translation_text"])

            translated = " ".join(translated_parts)

            # Step 3: Restore medical terms
            translated = self._restore_terms(translated, placeholder_map)
            log.info(f"Translated to {lang_code}: {len(text)} → {len(translated)} chars")
            return translated

        except Exception as e:
            log.error(f"Translation failed for lang={lang_code}: {e}")
            return f"[Translation unavailable: {str(e)}]"

    def translate_both(self, text: str) -> Dict[str, str]:
        """Translate to both Hindi and Marathi."""
        return {
            "hindi": self.translate(text, "hi"),
            "marathi": self.translate(text, "mr"),
        }

    def detect_language(self, text: str) -> str:
        """Detect language of input text."""
        try:
            from langdetect import detect
            return detect(text)
        except Exception:
            return "en"

    def _protect_terms(self, text: str):
        """Replace medical terms with unique placeholders."""
        placeholder_map = {}
        protected = text
        for i, term in enumerate(MEDICAL_TERMS_TO_PRESERVE):
            if term in protected:
                placeholder = f"MEDTERM{i:03d}"
                placeholder_map[placeholder] = term
                protected = protected.replace(term, placeholder)
        return protected, placeholder_map

    def _restore_terms(self, text: str, placeholder_map: Dict[str, str]) -> str:
        """Restore medical term placeholders."""
        for placeholder, term in placeholder_map.items():
            text = text.replace(placeholder, term)
        return text

    @staticmethod
    def _split_sentences(text: str) -> List[str]:
        """Split text into sentences for chunked translation."""
        sentences = re.split(r'(?<=[.!?])\s+', text)
        return [s.strip() for s in sentences if s.strip()]
