"""
entity_extractor.py — Post-processes raw NER token predictions into structured entities
"""

from typing import Dict, List, Any
from collections import defaultdict

from src.ner.ner_model import ClinicalNERModel
from src.utils.logger import log

# Entity group mappings
ENTITY_GROUP_MAP = {
    "MEDICINE_NAME": "MEDICINE_NAME",
    "DOSAGE": "DOSAGE",
    "FREQUENCY": "FREQUENCY",
    "DURATION": "DURATION",
    "ROUTE": "ROUTE",
    "TEST_NAME": "TEST_NAME",
    "TEST_VALUE": "TEST_VALUE",
    "UNIT": "UNIT",
    "NORMAL_RANGE": "NORMAL_RANGE",
    "ABNORMAL_FLAG": "ABNORMAL_FLAG",
    "DIAGNOSIS": "DIAGNOSIS",
    "SYMPTOM": "SYMPTOM",
    "PROCEDURE": "PROCEDURE",
    "DOCTOR_NAME": "DOCTOR_NAME",
    "HOSPITAL_NAME": "HOSPITAL_NAME",
    "DATE": "DATE",
    "PATIENT_NAME": "PATIENT_NAME",
    # Map common dslim/bert-base-NER labels to our schema
    "PER": "PATIENT_NAME",
    "ORG": "HOSPITAL_NAME",
    "LOC": "HOSPITAL_NAME",
    "MISC": "DIAGNOSIS",
}


class EntityExtractor:
    """Extracts and groups clinical entities from text using the NER model."""

    def __init__(self):
        self.model = ClinicalNERModel()

    def extract(self, text: str) -> Dict[str, List[str]]:
        """
        Extract all clinical entities from text.

        Returns:
            Dict mapping entity type → list of entity strings.
            e.g. {"MEDICINE_NAME": ["Paracetamol"], "DOSAGE": ["500mg"], ...}
        """
        if not text or not text.strip():
            log.warning("Empty text passed to EntityExtractor")
            return defaultdict(list)

        raw_entities = self.model.predict_tokens(text)
        grouped = self._group_entities(raw_entities)
        grouped = self._deduplicate(grouped)

        log.info(
            f"Extracted {sum(len(v) for v in grouped.values())} entities "
            f"across {len(grouped)} types"
        )
        return dict(grouped)

    def _group_entities(
        self, raw: List[Dict[str, Any]]
    ) -> Dict[str, List[str]]:
        """Map raw NER output to our entity schema."""
        grouped: Dict[str, List[str]] = defaultdict(list)
        for ent in raw:
            raw_label = ent.get("entity_group", ent.get("entity", "O"))
            # Strip BIO prefix if present
            if raw_label.startswith(("B-", "I-")):
                raw_label = raw_label[2:]
            mapped = ENTITY_GROUP_MAP.get(raw_label)
            if mapped and ent.get("score", 0) >= 0.5:
                word = ent.get("word", "").strip()
                if word and len(word) > 1:
                    grouped[mapped].append(word)
        return grouped

    def _deduplicate(
        self, grouped: Dict[str, List[str]]
    ) -> Dict[str, List[str]]:
        """Remove duplicates while preserving order."""
        return {k: list(dict.fromkeys(v)) for k, v in grouped.items()}

    def extract_with_confidence(
        self, text: str
    ) -> List[Dict[str, Any]]:
        """Return raw entity list with individual confidence scores."""
        raw = self.model.predict_tokens(text)
        results = []
        for ent in raw:
            raw_label = ent.get("entity_group", ent.get("entity", "O"))
            if raw_label.startswith(("B-", "I-")):
                raw_label = raw_label[2:]
            mapped = ENTITY_GROUP_MAP.get(raw_label)
            if mapped:
                results.append({
                    "entity_type": mapped,
                    "value": ent.get("word", "").strip(),
                    "confidence": round(ent.get("score", 0.0), 4),
                    "start": ent.get("start"),
                    "end": ent.get("end"),
                })
        return results
