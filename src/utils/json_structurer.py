"""
json_structurer.py — Converts raw NER entity lists to validated structured JSON
"""

from typing import Any, Dict, List, Optional
from datetime import datetime


# ── Schema Defaults ───────────────────────────────────────────────────────────

_EMPTY_PRESCRIPTION = {
    "name": None,
    "dosage": None,
    "frequency": None,
    "duration": None,
    "route": "Oral",
}

_EMPTY_LAB_TEST = {
    "test_name": None,
    "value": None,
    "unit": None,
    "normal_range": None,
    "status": "Normal",
}


def build_structured_json(
    document_type: str,
    ocr_text: str,
    corrected_text: str,
    entities: Dict[str, List[str]],
    translation_hindi: str = "",
    translation_marathi: str = "",
    patient_explanation: str = "",
    processing_time: float = 0.0,
) -> Dict[str, Any]:
    """
    Build the final unified output JSON from pipeline components.

    Args:
        document_type:       classified document type string
        ocr_text:            raw OCR output
        corrected_text:      post-corrected OCR text
        entities:            dict mapping entity types → list of values
        translation_hindi:   Hindi translation string
        translation_marathi: Marathi translation string
        patient_explanation: simplified explanation text
        processing_time:     total pipeline duration in seconds

    Returns:
        Validated output dictionary conforming to the system schema.
    """
    medicines = _extract_medicines(entities)
    lab_tests = _extract_lab_tests(entities)

    structured = {
        "schema_version": "1.0.0",
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "document_type": document_type or "Unknown",
        "ocr_text": ocr_text,
        "corrected_text": corrected_text,
        "entities": {
            "patient_name": _first(entities.get("PATIENT_NAME", [])),
            "doctor_name": _first(entities.get("DOCTOR_NAME", [])),
            "hospital_name": _first(entities.get("HOSPITAL_NAME", [])),
            "date": _first(entities.get("DATE", [])),
            "diagnosis": entities.get("DIAGNOSIS", []),
            "symptoms": entities.get("SYMPTOM", []),
            "procedures": entities.get("PROCEDURE", []),
            "medicines": medicines,
            "lab_tests": lab_tests,
        },
        "translation_hindi": translation_hindi,
        "translation_marathi": translation_marathi,
        "patient_explanation": patient_explanation,
        "processing_time_seconds": round(processing_time, 3),
    }
    return _apply_validation(structured)


def _first(lst: List[str]) -> Optional[str]:
    """Return first element or None."""
    return lst[0] if lst else None


def _extract_medicines(entities: Dict[str, List[str]]) -> List[Dict[str, Any]]:
    """Zip medicine-related entity lists into per-medicine dicts."""
    names = entities.get("MEDICINE_NAME", [])
    dosages = entities.get("DOSAGE", [])
    freqs = entities.get("FREQUENCY", [])
    durations = entities.get("DURATION", [])
    routes = entities.get("ROUTE", [])

    if not names:
        return []

    medicines = []
    for i, name in enumerate(names):
        entry = dict(_EMPTY_PRESCRIPTION)
        entry["name"] = name
        entry["dosage"] = dosages[i] if i < len(dosages) else None
        entry["frequency"] = freqs[i] if i < len(freqs) else None
        entry["duration"] = durations[i] if i < len(durations) else None
        entry["route"] = routes[i] if i < len(routes) else "Oral"
        medicines.append(entry)
    return medicines


def _extract_lab_tests(entities: Dict[str, List[str]]) -> List[Dict[str, Any]]:
    """Zip lab-related entity lists into per-test dicts."""
    names = entities.get("TEST_NAME", [])
    values = entities.get("TEST_VALUE", [])
    units = entities.get("UNIT", [])
    ranges = entities.get("NORMAL_RANGE", [])
    flags = entities.get("ABNORMAL_FLAG", [])

    if not names:
        return []

    tests = []
    for i, name in enumerate(names):
        entry = dict(_EMPTY_LAB_TEST)
        entry["test_name"] = name
        entry["value"] = values[i] if i < len(values) else None
        entry["unit"] = units[i] if i < len(units) else None
        entry["normal_range"] = ranges[i] if i < len(ranges) else None
        entry["status"] = flags[i] if i < len(flags) else "Normal"
        tests.append(entry)
    return tests


def _apply_validation(doc: Dict[str, Any]) -> Dict[str, Any]:
    """Apply basic validation rules and add warning flags."""
    warnings = []
    ents = doc.get("entities", {})

    if not ents.get("diagnosis") and doc["document_type"] in ("Prescription", "Discharge Summary"):
        warnings.append("No diagnosis entity found.")

    if not ents.get("medicines") and doc["document_type"] == "Prescription":
        warnings.append("No medicine entities found in prescription.")

    if not ents.get("lab_tests") and doc["document_type"] == "Lab Report":
        warnings.append("No lab test entities found in report.")

    if doc["document_type"] == "Unknown":
        warnings.append("Document type could not be classified.")

    doc["validation_warnings"] = warnings
    return doc
