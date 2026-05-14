"""
test_correction.py — Unit tests for OCR error correction and abbreviation normalization
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parents[1]))


class TestMedicalAbbreviations:
    def test_frequency_normalization(self):
        from src.correction.medical_abbreviations import normalize_abbreviation
        result = normalize_abbreviation("Take this OD")
        assert "Once Daily" in result

    def test_bd_normalization(self):
        from src.correction.medical_abbreviations import normalize_abbreviation
        result = normalize_abbreviation("BD dosing")
        assert "Twice Daily" in result

    def test_ocr_corrections(self):
        from src.correction.medical_abbreviations import apply_ocr_corrections
        result = apply_ocr_corrections("Tab. PCM 500mg")
        assert "Paracetamol" in result

    def test_hba1c_correction(self):
        from src.correction.medical_abbreviations import apply_ocr_corrections
        result = apply_ocr_corrections("HBAIc: 7.2%")
        assert "HbA1c" in result


class TestSpellCorrector:
    def test_corrector_returns_string(self):
        from src.correction.spell_corrector import MedicalSpellCorrector
        corrector = MedicalSpellCorrector(use_biobert=False)
        result = corrector.correct("Paracetamol 500mg BD")
        assert isinstance(result, str)
        assert len(result) > 0

    def test_empty_text(self):
        from src.correction.spell_corrector import MedicalSpellCorrector
        corrector = MedicalSpellCorrector()
        result = corrector.correct("")
        assert result == ""

    def test_batch_correct(self):
        from src.correction.spell_corrector import MedicalSpellCorrector
        corrector = MedicalSpellCorrector()
        texts = ["Tab. PCM BD", "Amoxycillin 500mg TDS"]
        results = corrector.batch_correct(texts)
        assert len(results) == 2
        assert all(isinstance(r, str) for r in results)
