"""
generate_synthetic_data.py — Creates synthetic annotated medical datasets
for NER training (CoNLL format) and document classifier training (CSV format)
"""

import csv
import random
import re
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parents[1]))

from src.utils.config import SYNTHETIC_DIR
from src.utils.logger import log

random.seed(42)

# ── Sample pools ──────────────────────────────────────────────────────────────
MEDICINES = [
    "Paracetamol", "Amoxicillin", "Azithromycin", "Metformin", "Atorvastatin",
    "Omeprazole", "Cetirizine", "Salbutamol", "Ibuprofen", "Diclofenac",
    "Ciprofloxacin", "Prednisolone", "Amlodipine", "Ramipril", "Losartan",
    "Glimepiride", "Doxycycline", "Cefixime", "Ondansetron", "Pantoprazole",
]
DOSAGES = ["500mg", "250mg", "1g", "10mg", "20mg", "40mg", "5mg", "100mg", "200mg"]
FREQUENCIES = ["OD", "BD", "TDS", "SOS", "QID", "HS"]
FREQ_EXPANDED = {
    "OD": "Once Daily", "BD": "Twice Daily", "TDS": "Thrice Daily",
    "SOS": "When Needed", "QID": "Four Times Daily", "HS": "At Bedtime",
}
DURATIONS = ["3 days", "5 days", "7 days", "10 days", "14 days", "1 month"]
DIAGNOSES = [
    "Viral Fever", "Hypertension", "Type 2 Diabetes", "Urinary Tract Infection",
    "Acute Gastritis", "Bronchitis", "Dengue Fever", "Malaria", "Typhoid",
    "Anemia", "Hypothyroidism", "Pneumonia", "Tonsillitis", "Sinusitis",
]
SYMPTOMS = [
    "fever", "cough", "headache", "body ache", "nausea", "vomiting",
    "fatigue", "dizziness", "chest pain", "shortness of breath",
]
LAB_TESTS = [
    "CBC", "HbA1c", "FBS", "Creatinine", "Bilirubin", "ESR", "CRP",
    "TSH", "Cholesterol", "Triglycerides", "Platelet Count", "WBC", "Hb",
]
DOCTORS = [
    "Dr. Rajesh Sharma", "Dr. Priya Mehta", "Dr. Anand Kulkarni",
    "Dr. Sunita Patil", "Dr. Vikram Joshi", "Dr. Meera Iyer",
]
HOSPITALS = [
    "City General Hospital", "Apollo Medical Centre", "Lilavati Hospital",
    "Fortis Healthcare", "AIIMS Mumbai", "Ruby Hall Clinic",
]
PATIENTS = [
    "Ramesh Patil", "Sunita Sharma", "Arun Kumar", "Priya Singh",
    "Deepak Joshi", "Meena Nair", "Vijay Rao", "Kavita Desai",
]

# ── Template generators ───────────────────────────────────────────────────────

def make_prescription() -> str:
    doc = random.choice(DOCTORS)
    hosp = random.choice(HOSPITALS)
    patient = random.choice(PATIENTS)
    diag = random.choice(DIAGNOSES)
    n_meds = random.randint(2, 4)
    lines = [
        f"{hosp}",
        f"Dr: {doc}",
        f"Patient: {patient}",
        f"Date: {random.randint(1,28):02d}/0{random.randint(1,9)}/2024",
        f"Dx: {diag}",
        "",
        "Rx:",
    ]
    for _ in range(n_meds):
        med = random.choice(MEDICINES)
        dos = random.choice(DOSAGES)
        frq = random.choice(FREQUENCIES)
        dur = random.choice(DURATIONS)
        lines.append(f"  Tab. {med} {dos} - {frq} x {dur}")
    lines += ["", "Advice: Rest, plenty of fluids", "Follow up: 1 week"]
    return "\n".join(lines)


def make_lab_report() -> str:
    patient = random.choice(PATIENTS)
    hosp = random.choice(HOSPITALS)
    n_tests = random.randint(3, 7)
    lines = [
        f"{hosp} — Pathology Department",
        f"Patient: {patient}",
        f"Sample Date: {random.randint(1,28):02d}/0{random.randint(1,9)}/2024",
        "TEST RESULTS:",
    ]
    for _ in range(n_tests):
        test = random.choice(LAB_TESTS)
        val = round(random.uniform(50, 200), 1)
        unit = random.choice(["mg/dL", "g/dL", "%", "mm/hr", "cells/uL"])
        flag = random.choice(["Normal", "Normal", "Normal", "High", "Low"])
        lines.append(f"  {test}: {val} {unit}  [{flag}]")
    return "\n".join(lines)


def make_discharge_summary() -> str:
    patient = random.choice(PATIENTS)
    doc = random.choice(DOCTORS)
    hosp = random.choice(HOSPITALS)
    diag = random.choice(DIAGNOSES)
    symp = random.sample(SYMPTOMS, k=random.randint(2, 4))
    proc = random.choice(["IV fluids", "Nebulisation", "Blood transfusion", "Catheterisation"])
    lines = [
        f"DISCHARGE SUMMARY — {hosp}",
        f"Patient: {patient}",
        f"Treating Physician: {doc}",
        f"Date of Admission: {random.randint(1,20):02d}/0{random.randint(1,9)}/2024",
        f"Date of Discharge: {random.randint(21,28):02d}/0{random.randint(1,9)}/2024",
        f"Primary Diagnosis: {diag}",
        f"Symptoms at Admission: {', '.join(symp)}",
        f"Procedure Done: {proc}",
        "Condition at Discharge: Stable",
        "Follow-up: 2 weeks with treating physician",
    ]
    return "\n".join(lines)


def make_medical_bill() -> str:
    patient = random.choice(PATIENTS)
    hosp = random.choice(HOSPITALS)
    items = [
        ("Consultation", random.randint(300, 1200)),
        ("Lab Tests", random.randint(500, 3000)),
        ("Medicines", random.randint(200, 1500)),
        ("Room Charges", random.randint(1000, 5000)),
    ]
    total = sum(v for _, v in items)
    lines = [
        f"INVOICE — {hosp}",
        f"Patient: {patient}",
        f"Bill Date: {random.randint(1,28):02d}/0{random.randint(1,9)}/2024",
        "CHARGES:",
    ] + [f"  {k}: Rs. {v}" for k, v in items] + [
        f"TOTAL AMOUNT: Rs. {total}",
        "Payment Mode: Cash / UPI",
    ]
    return "\n".join(lines)


def make_scan_report() -> str:
    patient = random.choice(PATIENTS)
    doc = random.choice(DOCTORS)
    scan_type = random.choice(["MRI Brain", "CT Abdomen", "USG Abdomen", "Chest X-Ray", "ECG"])
    finding = random.choice([
        "No significant abnormality detected.",
        "Mild hepatomegaly noted. No focal lesion.",
        "Small pleural effusion on the left side.",
        "Normal cardiac silhouette. Lung fields clear.",
        "Mild degenerative changes in lumbar spine.",
    ])
    return "\n".join([
        f"DIAGNOSTIC REPORT — {random.choice(HOSPITALS)}",
        f"Patient: {patient}",
        f"Referral: {doc}",
        f"Scan Type: {scan_type}",
        f"Date: {random.randint(1,28):02d}/0{random.randint(1,9)}/2024",
        "FINDINGS:",
        f"  {finding}",
        "IMPRESSION: As above. Clinical correlation advised.",
    ])


GENERATORS = {
    "Prescription": make_prescription,
    "Lab Report": make_lab_report,
    "Discharge Summary": make_discharge_summary,
    "Medical Bill": make_medical_bill,
    "Diagnostic Scan Report": make_scan_report,
}

# ── CoNLL NER label helpers ───────────────────────────────────────────────────

def tag_tokens(sentence: str, spans: list) -> list:
    """Tag tokens in a sentence given a list of (start_char, end_char, label)."""
    tokens = sentence.split()
    result = []
    char_pos = 0
    for token in tokens:
        start = sentence.find(token, char_pos)
        end = start + len(token)
        label = "O"
        for (s, e, lbl) in spans:
            if start >= s and end <= e:
                prefix = "B" if start == s else "I"
                label = f"{prefix}-{lbl}"
                break
        result.append((token, label))
        char_pos = end
    return result


def make_ner_sample():
    """Generate one sentence with entity annotations."""
    med = random.choice(MEDICINES)
    dos = random.choice(DOSAGES)
    frq = random.choice(FREQUENCIES)
    dur = random.choice(DURATIONS)
    sentence = f"Tab. {med} {dos} {frq} for {dur}"
    tokens_labels = [
        ("Tab.", "O"),
        (med, "B-MEDICINE_NAME"),
        (dos, "B-DOSAGE"),
        (frq, "B-FREQUENCY"),
        ("for", "O"),
    ] + [(w, "B-DURATION" if i == 0 else "I-DURATION")
         for i, w in enumerate(dur.split())]
    return tokens_labels


def make_diagnosis_ner():
    diag = random.choice(DIAGNOSES)
    patient = random.choice(PATIENTS)
    tokens = []
    for w in f"Patient {patient} diagnosed with".split():
        tokens.append((w, "O"))
    tokens[1] = (patient.split()[0], "B-PATIENT_NAME")
    if len(patient.split()) > 1:
        tokens.insert(2, (patient.split()[1], "I-PATIENT_NAME"))
    for i, w in enumerate(diag.split()):
        tokens.append((w, "B-DIAGNOSIS" if i == 0 else "I-DIAGNOSIS"))
    return tokens


# ── Main generation ───────────────────────────────────────────────────────────

def generate_classifier_dataset(n_per_class: int = 200) -> Path:
    SYNTHETIC_DIR.mkdir(parents=True, exist_ok=True)
    out_path = SYNTHETIC_DIR / "classifier_dataset.csv"
    rows = []
    for label, gen_fn in GENERATORS.items():
        for _ in range(n_per_class):
            rows.append({"text": gen_fn(), "label": label})
    random.shuffle(rows)
    with open(str(out_path), "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["text", "label"])
        writer.writeheader()
        writer.writerows(rows)
    log.info(f"Classifier dataset: {len(rows)} samples → {out_path}")
    return out_path


def generate_ner_dataset(n_samples: int = 500) -> Path:
    SYNTHETIC_DIR.mkdir(parents=True, exist_ok=True)
    out_path = SYNTHETIC_DIR / "ner_dataset.conll"
    makers = [make_ner_sample, make_diagnosis_ner]
    with open(str(out_path), "w", encoding="utf-8") as f:
        for i in range(n_samples):
            fn = random.choice(makers)
            tokens_labels = fn()
            for token, label in tokens_labels:
                f.write(f"{token} {label}\n")
            f.write("\n")
    log.info(f"NER dataset: {n_samples} sentences → {out_path}")
    return out_path


if __name__ == "__main__":
    log.info("Generating synthetic datasets...")
    clf_path = generate_classifier_dataset(n_per_class=200)
    ner_path = generate_ner_dataset(n_samples=600)
    log.info(f"Done. Files written:\n  {clf_path}\n  {ner_path}")
