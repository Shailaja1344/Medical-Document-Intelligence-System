"""
medical_dictionary.py — Curated medical term lookup for patient-friendly explanations
"""

from typing import Optional, Dict

# ── Medicine Explanations ─────────────────────────────────────────────────────
MEDICINE_EXPLANATIONS: Dict[str, str] = {
    "paracetamol": "a medicine used to reduce fever and relieve mild to moderate pain",
    "acetaminophen": "a medicine used to reduce fever and relieve mild to moderate pain",
    "ibuprofen": "a pain reliever and fever reducer; also reduces inflammation",
    "diclofenac": "a medicine to reduce pain, swelling, and inflammation",
    "aspirin": "a medicine that reduces pain, fever, and inflammation; also prevents blood clots",
    "amoxicillin": "an antibiotic used to treat bacterial infections like throat, ear, and lung infections",
    "azithromycin": "an antibiotic used to treat respiratory and skin bacterial infections",
    "ciprofloxacin": "an antibiotic used for urinary tract and other bacterial infections",
    "metronidazole": "an antibiotic used for stomach and intestinal infections",
    "cefixime": "an antibiotic used to treat various bacterial infections",
    "doxycycline": "an antibiotic effective against many types of bacterial infections",
    "omeprazole": "a medicine that reduces stomach acid production; used for acidity and ulcers",
    "pantoprazole": "a medicine that reduces stomach acid; used for acid reflux and ulcers",
    "ranitidine": "a medicine that reduces stomach acid production",
    "ondansetron": "a medicine that prevents nausea and vomiting",
    "domperidone": "a medicine that relieves nausea and helps food move through the stomach",
    "metformin": "a medicine used to control blood sugar levels in type 2 diabetes",
    "glimepiride": "a medicine that lowers blood sugar levels in type 2 diabetes",
    "insulin": "a hormone used as medicine to control blood sugar levels in diabetes",
    "atorvastatin": "a medicine that lowers cholesterol levels to protect the heart",
    "amlodipine": "a medicine that relaxes blood vessels to lower blood pressure",
    "enalapril": "a medicine used to treat high blood pressure and heart failure",
    "ramipril": "a medicine used to lower blood pressure and protect the heart",
    "losartan": "a medicine used to treat high blood pressure",
    "telmisartan": "a medicine used to lower blood pressure",
    "cetirizine": "an antihistamine medicine used to treat allergies and itching",
    "montelukast": "a medicine used to prevent asthma and treat allergies",
    "salbutamol": "a medicine that opens up the airways; used for asthma and breathing difficulty",
    "prednisolone": "a steroid medicine that reduces swelling and immune reactions",
    "dexamethasone": "a strong steroid medicine that reduces severe inflammation",
    "levothyroxine": "a thyroid hormone replacement medicine for hypothyroidism",
    "furosemide": "a water tablet that removes excess fluid from the body",
    "clopidogrel": "a medicine that prevents blood clots; used after heart attack or stroke",
    "warfarin": "a blood-thinning medicine that prevents dangerous blood clots",
    "vitamin d": "a vitamin supplement important for bone health and immunity",
    "calcium": "a mineral supplement important for strong bones and teeth",
    "iron": "a supplement used to treat iron deficiency and anemia",
    "folic acid": "a vitamin supplement important during pregnancy and for blood health",
    "zinc": "a mineral supplement that supports immunity and wound healing",
    "multivitamin": "a supplement containing multiple vitamins and minerals for overall health",
}

# ── Diagnosis / Condition Explanations ───────────────────────────────────────
DIAGNOSIS_EXPLANATIONS: Dict[str, str] = {
    "hypertension": "high blood pressure — the pressure of blood in arteries is too high",
    "diabetes mellitus": "a condition where blood sugar levels are too high due to lack of or resistance to insulin",
    "type 2 diabetes": "a condition where the body doesn't use insulin properly, causing high blood sugar",
    "viral fever": "a fever caused by a virus; usually resolves on its own with rest and fluids",
    "bacterial infection": "an illness caused by bacteria; usually treated with antibiotics",
    "urinary tract infection": "an infection in the urinary system (bladder/kidneys); causes burning urination",
    "anemia": "a condition where blood lacks enough healthy red blood cells to carry oxygen",
    "hypothyroidism": "a condition where the thyroid gland doesn't produce enough thyroid hormone",
    "hyperthyroidism": "a condition where the thyroid gland produces too much thyroid hormone",
    "asthma": "a breathing condition where airways narrow and become inflamed",
    "copd": "a lung disease that causes breathing difficulty; usually from smoking",
    "pneumonia": "an infection that inflames the air sacs in the lungs",
    "bronchitis": "inflammation of the bronchial tubes that carry air to the lungs",
    "dengue": "a viral illness spread by mosquitoes causing fever, rash, and body aches",
    "malaria": "a mosquito-borne illness causing fever, chills, and flu-like symptoms",
    "typhoid": "a bacterial infection causing high fever, stomach pain, and weakness",
    "gastritis": "inflammation of the stomach lining causing pain and indigestion",
    "peptic ulcer": "a sore in the lining of the stomach or small intestine",
    "chronic kidney disease": "gradual loss of kidney function over time",
    "coronary artery disease": "narrowing of heart arteries reducing blood flow to the heart",
    "heart failure": "the heart cannot pump enough blood to meet the body's needs",
    "stroke": "brain damage caused by a blocked or burst blood vessel in the brain",
    "arthritis": "painful inflammation and stiffness of joints",
    "osteoporosis": "bones become weak and brittle, increasing fracture risk",
}

# ── Lab Test Explanations ─────────────────────────────────────────────────────
LAB_TEST_EXPLANATIONS: Dict[str, str] = {
    "cbc": "Complete Blood Count — checks the health of red cells, white cells, and platelets",
    "complete blood count": "checks overall blood health including red cells, white cells, and platelets",
    "hemoglobin": "the protein in red blood cells that carries oxygen; low levels indicate anemia",
    "hba1c": "Glycated Hemoglobin — shows average blood sugar levels over the past 3 months",
    "fasting blood sugar": "blood sugar level measured after 8+ hours without eating",
    "random blood sugar": "blood sugar level measured at any time of day",
    "creatinine": "a waste product filtered by kidneys; high levels may indicate kidney problems",
    "urea": "a waste product from protein metabolism filtered by kidneys",
    "cholesterol": "a fatty substance in blood; high levels increase heart disease risk",
    "triglycerides": "a type of fat in blood; high levels increase heart disease risk",
    "tsh": "Thyroid Stimulating Hormone — checks thyroid gland function",
    "esr": "Erythrocyte Sedimentation Rate — measures inflammation in the body",
    "crp": "C-Reactive Protein — a marker that rises when there is inflammation or infection",
    "liver function test": "a group of tests checking how well the liver is working",
    "kidney function test": "a group of tests checking how well the kidneys are filtering waste",
    "urine routine": "basic urine test checking for infection, kidney problems, or diabetes",
    "ecg": "Electrocardiogram — records the electrical activity of the heart",
    "chest x-ray": "an image of the chest showing the heart, lungs, and bones",
    "usg abdomen": "Ultrasound of the abdomen — painless imaging of abdominal organs",
    "bilirubin": "a yellow substance from broken-down red blood cells; high levels cause jaundice",
    "platelet count": "the number of platelets that help blood clot; low counts increase bleeding risk",
    "wbc": "White Blood Cell Count — measures immune system cells; high levels may indicate infection",
    "rbc": "Red Blood Cell Count — measures oxygen-carrying cells; low levels indicate anemia",
}

# ── Abbreviation Explanations ─────────────────────────────────────────────────
ABBREVIATION_EXPLANATIONS: Dict[str, str] = {
    "od": "take once daily (once a day)",
    "bd": "take twice daily (two times a day)",
    "tds": "take thrice daily (three times a day)",
    "qid": "take four times daily",
    "sos": "take only when needed",
    "hs": "take at bedtime",
    "ac": "take before meals",
    "pc": "take after meals",
    "tab": "tablet",
    "cap": "capsule",
    "syp": "syrup",
    "inj": "injection",
    "po": "by mouth (to be swallowed)",
    "iv": "through a vein (intravenous)",
    "im": "injected into a muscle (intramuscular)",
    "bp": "blood pressure",
    "hr": "heart rate (pulse)",
    "spo2": "oxygen saturation level in blood",
    "bmi": "body mass index — a measure of body fat based on height and weight",
    "opd": "outpatient department — clinic visit without hospital admission",
    "ipd": "inpatient department — admitted to the hospital",
    "icu": "intensive care unit — for critically ill patients",
    "rx": "prescription",
    "dx": "diagnosis",
}


def explain_term(term: str, category: str = "auto") -> Optional[str]:
    """
    Look up patient-friendly explanation for a medical term.

    Args:
        term: medical term, abbreviation, or drug name
        category: 'medicine', 'diagnosis', 'lab', 'abbreviation', or 'auto'

    Returns:
        Explanation string, or None if not found.
    """
    key = term.lower().strip()

    if category == "medicine" or category == "auto":
        if key in MEDICINE_EXPLANATIONS:
            return MEDICINE_EXPLANATIONS[key]

    if category == "diagnosis" or category == "auto":
        if key in DIAGNOSIS_EXPLANATIONS:
            return DIAGNOSIS_EXPLANATIONS[key]

    if category == "lab" or category == "auto":
        if key in LAB_TEST_EXPLANATIONS:
            return LAB_TEST_EXPLANATIONS[key]

    if category == "abbreviation" or category == "auto":
        if key in ABBREVIATION_EXPLANATIONS:
            return ABBREVIATION_EXPLANATIONS[key]

    return None


def get_all_known_terms() -> Dict[str, int]:
    """Return count of known terms per category."""
    return {
        "medicines": len(MEDICINE_EXPLANATIONS),
        "diagnoses": len(DIAGNOSIS_EXPLANATIONS),
        "lab_tests": len(LAB_TEST_EXPLANATIONS),
        "abbreviations": len(ABBREVIATION_EXPLANATIONS),
        "total": (
            len(MEDICINE_EXPLANATIONS) + len(DIAGNOSIS_EXPLANATIONS) +
            len(LAB_TEST_EXPLANATIONS) + len(ABBREVIATION_EXPLANATIONS)
        ),
    }
