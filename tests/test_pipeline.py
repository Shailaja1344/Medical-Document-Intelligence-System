"""
test_pipeline.py — Integration tests for the full pipeline orchestrator
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parents[1]))

import pytest


PRESCRIPTION_TEXT = """
Dr. Rajesh Sharma — City General Hospital
Patient: Ramesh Patil
Date: 15/03/2024
Diagnosis: Viral Fever

Rx:
  Tab. Paracetamol 500mg BD x 5 days
  Tab. Amoxicillin 500mg TDS x 7 days
  Syp. Cetirizine 5ml OD x 5 days

Advice: Rest, take plenty of fluids, avoid cold food.
Follow up: 1 week
"""

LAB_TEXT = """
Apollo Diagnostics — Lab Report
Patient: Sunita Mehta    Date: 10/04/2024

Complete Blood Count (CBC):
  Hemoglobin (Hb): 11.2 g/dL   [Normal: 12–16]  LOW
  WBC Count: 9800 cells/uL      [Normal]
  Platelet Count: 2.1 Lakh/uL   [Normal]

HbA1c: 8.2%  [Normal: <5.7%]  HIGH — indicates poor diabetes control

Impression: Mild anemia. Poor glycemic control.
"""


class TestPipelineText:
    """Integration tests using text input (no OCR needed)."""

    @pytest.fixture(scope="class")
    def pipeline(self):
        from src.pipeline.orchestrator import MedicalDocumentPipeline
        return MedicalDocumentPipeline()

    def test_prescription_pipeline(self, pipeline):
        result = pipeline.process_text(PRESCRIPTION_TEXT)
        assert result["document_type"] is not None
        assert len(result["ocr_text"]) > 0 or len(result["corrected_text"]) > 0
        assert "entities" in result
        assert "processing_time_seconds" in result

    def test_lab_report_pipeline(self, pipeline):
        result = pipeline.process_text(LAB_TEXT)
        assert result["document_type"] is not None
        assert isinstance(result["validation_warnings"], list)

    def test_output_schema_keys(self, pipeline):
        result = pipeline.process_text(PRESCRIPTION_TEXT)
        required_keys = [
            "schema_version", "timestamp", "document_type",
            "ocr_text", "corrected_text", "entities",
            "translation_hindi", "translation_marathi",
            "patient_explanation", "processing_time_seconds",
            "validation_warnings",
        ]
        for key in required_keys:
            assert key in result, f"Missing key: {key}"

    def test_entities_structure(self, pipeline):
        result = pipeline.process_text(PRESCRIPTION_TEXT)
        entities = result["entities"]
        assert "medicines" in entities
        assert "diagnosis" in entities
        assert "lab_tests" in entities
        assert "symptoms" in entities
        assert isinstance(entities["medicines"], list)
        assert isinstance(entities["diagnosis"], list)

    def test_explanation_generated(self, pipeline):
        result = pipeline.process_text(PRESCRIPTION_TEXT)
        assert isinstance(result.get("patient_explanation"), str)

    def test_processing_time_reasonable(self, pipeline):
        result = pipeline.process_text("Tab. Paracetamol 500mg BD x 5 days")
        assert result["processing_time_seconds"] < 60.0


class TestJSONStructurer:
    def test_build_empty(self):
        from src.utils.json_structurer import build_structured_json
        result = build_structured_json(
            document_type="Prescription",
            ocr_text="test",
            corrected_text="test",
            entities={},
        )
        assert result["document_type"] == "Prescription"
        assert isinstance(result["validation_warnings"], list)

    def test_medicine_extraction(self):
        from src.utils.json_structurer import build_structured_json
        entities = {
            "MEDICINE_NAME": ["Paracetamol", "Amoxicillin"],
            "DOSAGE": ["500mg", "500mg"],
            "FREQUENCY": ["BD", "TDS"],
        }
        result = build_structured_json("Prescription", "", "", entities)
        meds = result["entities"]["medicines"]
        assert len(meds) == 2
        assert meds[0]["name"] == "Paracetamol"
        assert meds[0]["dosage"] == "500mg"


class TestExplainer:
    def test_medicine_explanation(self):
        from src.explanation.explainer import PatientExplainer
        explainer = PatientExplainer()
        result = explainer.explain_single_term("paracetamol")
        assert "fever" in result.lower() or "pain" in result.lower()

    def test_unknown_term(self):
        from src.explanation.explainer import PatientExplainer
        explainer = PatientExplainer()
        result = explainer.explain_single_term("xyzmedicalterm999")
        assert "doctor" in result.lower()
