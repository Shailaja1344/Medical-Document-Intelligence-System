"""
explainer.py — Patient-friendly medical explanation generator
"""

from typing import Dict, List, Any, Optional

from src.explanation.medical_dictionary import (
    explain_term, MEDICINE_EXPLANATIONS, DIAGNOSIS_EXPLANATIONS,
    LAB_TEST_EXPLANATIONS, ABBREVIATION_EXPLANATIONS
)
from src.utils.logger import log


class PatientExplainer:
    """
    Generates simple, patient-friendly explanations of:
    - Medicines (what they do)
    - Lab test results (what values mean)
    - Diagnoses (plain-language description)
    - Dosage abbreviations (how to take medicines)
    """

    def explain_document(self, structured_json: Dict[str, Any]) -> str:
        """
        Generate a complete patient-friendly explanation from structured JSON.

        Args:
            structured_json: output from json_structurer.build_structured_json()

        Returns:
            A human-readable explanation paragraph.
        """
        sections = []
        entities = structured_json.get("entities", {})
        doc_type = structured_json.get("document_type", "Unknown")

        # Document type intro
        sections.append(
            f"This is a {doc_type}. "
            "Below is a simple explanation to help you understand it."
        )

        # Diagnoses
        diagnoses = entities.get("diagnosis", [])
        if diagnoses:
            diag_text = self._explain_diagnoses(diagnoses)
            if diag_text:
                sections.append("📋 DIAGNOSIS:\n" + diag_text)

        # Medicines
        medicines = entities.get("medicines", [])
        if medicines:
            med_text = self._explain_medicines(medicines)
            if med_text:
                sections.append("💊 YOUR MEDICINES:\n" + med_text)

        # Lab tests
        lab_tests = entities.get("lab_tests", [])
        if lab_tests:
            lab_text = self._explain_lab_tests(lab_tests)
            if lab_text:
                sections.append("🔬 YOUR LAB RESULTS:\n" + lab_text)

        # Symptoms
        symptoms = entities.get("symptoms", [])
        if symptoms:
            sections.append(
                "🤒 SYMPTOMS MENTIONED: " + ", ".join(symptoms)
            )

        if len(sections) == 1:
            sections.append(
                "No specific medical details could be automatically identified. "
                "Please consult your doctor for clarification."
            )

        return "\n\n".join(sections)

    def _explain_diagnoses(self, diagnoses: List[str]) -> str:
        lines = []
        for diag in diagnoses:
            explanation = explain_term(diag, "diagnosis")
            if explanation:
                lines.append(f"• {diag.title()}: {explanation}.")
            else:
                lines.append(f"• {diag.title()}: please ask your doctor for details.")
        return "\n".join(lines)

    def _explain_medicines(self, medicines: List[Dict[str, Any]]) -> str:
        lines = []
        for med in medicines:
            name = med.get("name") or "Unknown medicine"
            dosage = med.get("dosage") or ""
            freq = med.get("frequency") or ""
            duration = med.get("duration") or ""
            route = med.get("route") or "orally"

            explanation = explain_term(name, "medicine")
            desc = explanation or "a medicine prescribed by your doctor"

            instruction_parts = []
            if dosage:
                instruction_parts.append(dosage)
            if freq:
                freq_exp = explain_term(freq, "abbreviation") or freq
                instruction_parts.append(freq_exp)
            if duration:
                instruction_parts.append(f"for {duration}")

            instruction = ", ".join(instruction_parts) if instruction_parts else ""

            line = f"• {name.title()} ({route}): {desc}."
            if instruction:
                line += f" Take {instruction}."
            lines.append(line)
        return "\n".join(lines)

    def _explain_lab_tests(self, lab_tests: List[Dict[str, Any]]) -> str:
        lines = []
        for test in lab_tests:
            name = test.get("test_name") or "Unknown test"
            value = test.get("value") or ""
            unit = test.get("unit") or ""
            normal_range = test.get("normal_range") or ""
            status = test.get("status") or "Normal"

            explanation = explain_term(name, "lab")
            desc = explanation or "a test done to check your health"

            value_str = f"{value} {unit}".strip() if value else ""
            range_str = f"(normal range: {normal_range})" if normal_range else ""
            status_icon = "✅" if status == "Normal" else "⚠️"

            line = f"• {name.upper()}: {desc}."
            if value_str:
                line += f" Your result: {value_str} {range_str}. {status_icon} {status}."
            lines.append(line)
        return "\n".join(lines)

    def explain_single_term(self, term: str) -> str:
        """Explain a single medical term in plain language."""
        exp = explain_term(term)
        if exp:
            return f"{term.title()} means: {exp}."
        return f"'{term}' — no explanation found. Please consult your doctor."

    def explain_abbreviations_in_text(self, text: str) -> Dict[str, str]:
        """Find and explain all known abbreviations appearing in a text."""
        found = {}
        words = text.upper().split()
        for word in words:
            clean = word.strip(".,;:()")
            exp = explain_term(clean, "abbreviation")
            if exp:
                found[clean] = exp
        return found
