"""
medical_abbreviations.py — Medical abbreviation normalization dictionary
"""

from typing import Dict
import re

FREQUENCY_MAP: Dict[str, str] = {
    "OD": "Once Daily", "BD": "Twice Daily", "BID": "Twice Daily",
    "TDS": "Thrice Daily", "TID": "Thrice Daily", "QID": "Four Times Daily",
    "SOS": "When Needed", "PRN": "When Needed", "HS": "At Bedtime",
    "AC": "Before Meals", "PC": "After Meals", "STAT": "Immediately",
    "Q4H": "Every 4 Hours", "Q6H": "Every 6 Hours", "Q8H": "Every 8 Hours",
    "Q12H": "Every 12 Hours", "MANE": "In the Morning", "NOCTE": "At Night",
}

ROUTE_MAP: Dict[str, str] = {
    "PO": "Oral", "IV": "Intravenous", "IM": "Intramuscular",
    "SC": "Subcutaneous", "SL": "Sublingual", "PR": "Per Rectum",
    "TOP": "Topical", "INH": "Inhalation", "TD": "Transdermal",
}

DOSAGE_FORM_MAP: Dict[str, str] = {
    "Tab": "Tablet", "TAB": "Tablet", "Cap": "Capsule", "CAP": "Capsule",
    "Syp": "Syrup", "SYP": "Syrup", "Inj": "Injection", "INJ": "Injection",
    "SUSP": "Suspension", "OINT": "Ointment", "CR": "Cream",
}

CLINICAL_MAP: Dict[str, str] = {
    "BP": "Blood Pressure", "HR": "Heart Rate", "SpO2": "Oxygen Saturation",
    "CBC": "Complete Blood Count", "LFT": "Liver Function Test",
    "KFT": "Kidney Function Test", "TFT": "Thyroid Function Test",
    "ECG": "Electrocardiogram", "MRI": "Magnetic Resonance Imaging",
    "CT": "Computed Tomography", "USG": "Ultrasonography",
    "HbA1c": "Glycated Hemoglobin", "FBS": "Fasting Blood Sugar",
    "WBC": "White Blood Cell Count", "RBC": "Red Blood Cell Count",
    "Hb": "Hemoglobin", "ESR": "Erythrocyte Sedimentation Rate",
    "CRP": "C-Reactive Protein", "DM": "Diabetes Mellitus",
    "HTN": "Hypertension", "COPD": "Chronic Obstructive Pulmonary Disease",
    "UTI": "Urinary Tract Infection", "ICU": "Intensive Care Unit",
    "OPD": "Outpatient Department", "IPD": "Inpatient Department",
    "SOB": "Shortness of Breath", "Rx": "Prescription", "Dx": "Diagnosis",
    "Hx": "History", "Sx": "Symptoms",
}

OCR_CORRECTION_MAP: Dict[str, str] = {
    "Tab. PCM": "Tab. Paracetamol", "Tab. PGM": "Tab. Paracetamol",
    "HBAIc": "HbA1c", "HBA1c": "HbA1c", "l-o-l": "1-0-1",
    "Amoxycillin": "Amoxicillin", "Metform1n": "Metformin",
    "Azithromyc1n": "Azithromycin", "0D": "OD", "8D": "BD", "TD5": "TDS",
}

ALL_ABBREVIATIONS: Dict[str, str] = {
    **FREQUENCY_MAP, **ROUTE_MAP, **DOSAGE_FORM_MAP, **CLINICAL_MAP,
}


def normalize_abbreviation(text: str) -> str:
    for abbr, expansion in ALL_ABBREVIATIONS.items():
        text = re.sub(r'\b' + re.escape(abbr) + r'\b', expansion, text)
    return text


def apply_ocr_corrections(text: str) -> str:
    for wrong, right in OCR_CORRECTION_MAP.items():
        text = text.replace(wrong, right)
    return text
