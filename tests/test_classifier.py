"""
test_classifier.py — Unit tests for document type classifier
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parents[1]))

import pytest
from src.utils.config import DOCUMENT_CLASSES


SAMPLE_TEXTS = {
    "Prescription": (
        "Dr. Sharma — Apollo Hospital\nPatient: Ramesh Patil\n"
        "Dx: Viral Fever\nRx:\n  Tab. Paracetamol 500mg BD x 5 days\n"
        "  Tab. Amoxicillin 500mg TDS x 7 days"
    ),
    "Lab Report": (
        "City Lab — Pathology Report\nPatient: Sunita Mehta\n"
        "CBC: Hb 11.2 g/dL [Low], WBC 9800 cells/uL [Normal], "
        "Platelet Count 2.1 Lakh [Normal]\nHbA1c: 8.2% [High]"
    ),
    "Discharge Summary": (
        "DISCHARGE SUMMARY — Lilavati Hospital\nPatient: Arun Kumar\n"
        "Date of Admission: 10/03/2024\nDate of Discharge: 15/03/2024\n"
        "Primary Diagnosis: Dengue Fever\nCondition at Discharge: Stable"
    ),
}


class TestTFIDFSVMClassifier:
    def test_train_and_predict(self):
        from src.classifier.tfidf_svm_classifier import TFIDFSVMClassifier
        clf = TFIDFSVMClassifier()
        texts, labels = [], []
        for label, text in SAMPLE_TEXTS.items():
            for _ in range(10):
                texts.append(text)
                labels.append(label)
        # add remaining classes with dummy text
        for cls in DOCUMENT_CLASSES:
            if cls not in SAMPLE_TEXTS:
                texts.append(f"Invoice Bill payment charges {cls}")
                labels.append(cls)
        clf.train(texts, labels)
        result = clf.predict(SAMPLE_TEXTS["Prescription"])
        assert "document_type" in result
        assert result["document_type"] in DOCUMENT_CLASSES
        assert 0.0 <= result["confidence"] <= 1.0

    def test_predict_returns_all_scores(self):
        from src.classifier.tfidf_svm_classifier import TFIDFSVMClassifier
        clf = TFIDFSVMClassifier()
        texts = [t for t in SAMPLE_TEXTS.values()] * 5
        labels = [l for l in SAMPLE_TEXTS.keys()] * 5
        for cls in DOCUMENT_CLASSES:
            if cls not in SAMPLE_TEXTS:
                texts.append(f"scan mri result {cls}")
                labels.append(cls)
        clf.train(texts, labels)
        result = clf.predict("CBC blood test report hemoglobin")
        assert "all_scores" in result
        assert len(result["all_scores"]) == len(DOCUMENT_CLASSES)


class TestKeywordClassify:
    def test_prescription_keyword(self):
        from src.pipeline.orchestrator import MedicalDocumentPipeline
        doc_type = MedicalDocumentPipeline._keyword_classify(
            "Rx Tab. Paracetamol BD dosage prescription"
        )
        assert doc_type == "Prescription"

    def test_lab_report_keyword(self):
        from src.pipeline.orchestrator import MedicalDocumentPipeline
        doc_type = MedicalDocumentPipeline._keyword_classify(
            "blood test result CBC lab report hemoglobin"
        )
        assert doc_type == "Lab Report"

    def test_discharge_keyword(self):
        from src.pipeline.orchestrator import MedicalDocumentPipeline
        doc_type = MedicalDocumentPipeline._keyword_classify(
            "discharge summary admitted ward ICU surgery"
        )
        assert doc_type == "Discharge Summary"
