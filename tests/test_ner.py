"""
test_ner.py — Unit tests for clinical entity extractor
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parents[1]))


class TestEntityExtractor:
    def test_extract_returns_dict(self):
        from src.ner.entity_extractor import EntityExtractor
        extractor = EntityExtractor()
        result = extractor.extract("Patient has Viral Fever. Tab. Paracetamol 500mg BD.")
        assert isinstance(result, dict)

    def test_empty_text(self):
        from src.ner.entity_extractor import EntityExtractor
        extractor = EntityExtractor()
        result = extractor.extract("")
        assert isinstance(result, dict)

    def test_confidence_filtering(self):
        from src.ner.entity_extractor import EntityExtractor
        extractor = EntityExtractor()
        raw_ents = [
            {"entity_group": "MEDICINE_NAME", "word": "Paracetamol", "score": 0.95, "start": 0, "end": 11},
            {"entity_group": "DOSAGE", "word": "500mg", "score": 0.3, "start": 12, "end": 17},
        ]
        grouped = extractor._group_entities(raw_ents)
        assert "MEDICINE_NAME" in grouped
        assert "DOSAGE" not in grouped  # filtered by 0.5 threshold

    def test_deduplication(self):
        from src.ner.entity_extractor import EntityExtractor
        extractor = EntityExtractor()
        grouped = {"MEDICINE_NAME": ["Paracetamol", "Paracetamol", "Amoxicillin"]}
        deduped = extractor._deduplicate(grouped)
        assert deduped["MEDICINE_NAME"].count("Paracetamol") == 1


class TestMedicalDictionary:
    def test_medicine_lookup(self):
        from src.explanation.medical_dictionary import explain_term
        result = explain_term("paracetamol", "medicine")
        assert result is not None
        assert "fever" in result.lower()

    def test_diagnosis_lookup(self):
        from src.explanation.medical_dictionary import explain_term
        result = explain_term("hypertension", "diagnosis")
        assert result is not None
        assert "blood pressure" in result.lower()

    def test_lab_lookup(self):
        from src.explanation.medical_dictionary import explain_term
        result = explain_term("cbc", "lab")
        assert result is not None

    def test_abbreviation_lookup(self):
        from src.explanation.medical_dictionary import explain_term
        result = explain_term("bd", "abbreviation")
        assert result is not None
        assert "twice" in result.lower()

    def test_unknown_returns_none(self):
        from src.explanation.medical_dictionary import explain_term
        result = explain_term("xyztermdoesnotexist999")
        assert result is None

    def test_auto_category(self):
        from src.explanation.medical_dictionary import explain_term
        assert explain_term("metformin") is not None
        assert explain_term("od") is not None
