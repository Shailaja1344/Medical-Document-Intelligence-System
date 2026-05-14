"""
streamlit_app.py — Interactive Streamlit UI for Medical Document Intelligence
"""

import json
import time
import io
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import streamlit as st
import numpy as np
from PIL import Image

# ── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Medical Document Intelligence",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
  html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
  .main { background: #0f1117; }
  .stApp { background: linear-gradient(135deg, #0f1117 0%, #1a1f2e 100%); }

  .metric-card {
    background: linear-gradient(145deg, #1e2130, #252b40);
    border: 1px solid #2d3561;
    border-radius: 16px;
    padding: 20px;
    margin: 8px 0;
    box-shadow: 0 4px 20px rgba(0,0,0,0.3);
  }
  .entity-badge {
    display: inline-block;
    padding: 4px 12px;
    border-radius: 20px;
    font-size: 13px;
    font-weight: 500;
    margin: 3px;
  }
  .medicine-badge { background: #1a3a2a; color: #4ade80; border: 1px solid #166534; }
  .diagnosis-badge { background: #2a1a1a; color: #f87171; border: 1px solid #7f1d1d; }
  .test-badge { background: #1a2a3a; color: #60a5fa; border: 1px solid #1e3a5f; }
  .warning-badge { background: #2a2a1a; color: #fbbf24; border: 1px solid #78350f; }

  .result-section {
    background: #1e2130;
    border-radius: 12px;
    padding: 16px 20px;
    margin: 10px 0;
    border-left: 4px solid #4f46e5;
  }
  .explanation-card {
    background: linear-gradient(145deg, #1a2a1a, #1e2820);
    border: 1px solid #166534;
    border-radius: 12px;
    padding: 20px;
    white-space: pre-line;
    line-height: 1.8;
  }
  h1, h2, h3 { color: #e2e8f0 !important; }
  .stButton>button {
    background: linear-gradient(135deg, #4f46e5, #7c3aed);
    color: white;
    border: none;
    border-radius: 10px;
    padding: 10px 24px;
    font-weight: 600;
    transition: all 0.3s;
  }
  .stButton>button:hover { transform: translateY(-2px); box-shadow: 0 6px 20px rgba(79,70,229,0.4); }
</style>
""", unsafe_allow_html=True)


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🏥 Medical Document AI")
    st.markdown("---")
    mode = st.radio(
        "Input Mode",
        ["📄 Upload Document", "📝 Enter Text Directly"],
        index=0,
    )
    st.markdown("---")
    st.markdown("### ⚙️ Settings")
    translate_hi = st.checkbox("Translate to Hindi", value=True)
    translate_mr = st.checkbox("Translate to Marathi", value=True)
    explain_mode = st.checkbox("Generate Patient Explanation", value=True)
    st.markdown("---")
    st.markdown("### 📊 About")
    st.info(
        "Processes medical prescriptions, lab reports, "
        "discharge summaries, and more using AI."
    )
    st.markdown("**Modules:**")
    st.markdown("- 🔍 OCR: EasyOCR + Tesseract")
    st.markdown("- 🏷️ NER: BioBERT")
    st.markdown("- 🌐 Translation: MarianMT")
    st.markdown("- 💊 Explanation: Medical Dictionary")


# ── Lazy pipeline loader ──────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading AI pipeline...")
def load_pipeline():
    from src.pipeline.orchestrator import MedicalDocumentPipeline
    return MedicalDocumentPipeline()


# ── Main UI ───────────────────────────────────────────────────────────────────
st.markdown("# 🏥 Medical Document Intelligence System")
st.markdown("*AI-powered healthcare document analysis — OCR, NER, Translation & Patient Explanation*")
st.markdown("---")

input_text = None
input_image = None

if mode == "📄 Upload Document":
    col1, col2 = st.columns([1, 1])
    with col1:
        uploaded = st.file_uploader(
            "Upload medical document",
            type=["jpg", "jpeg", "png", "pdf", "tiff", "bmp"],
            help="Supports scanned prescriptions, lab reports, handwritten notes, PDFs",
        )
        if uploaded:
            if uploaded.type != "application/pdf":
                pil_img = Image.open(uploaded)
                st.image(pil_img, caption="Uploaded Document", use_column_width=True)
                input_image = np.array(pil_img)
            else:
                st.success(f"PDF uploaded: {uploaded.name}")
                # Save temp PDF
                tmp_path = Path("data/processed/temp_upload.pdf")
                tmp_path.parent.mkdir(parents=True, exist_ok=True)
                tmp_path.write_bytes(uploaded.read())
                from src.preprocessing.image_processor import ImageProcessor
                import cv2
                input_image = ImageProcessor.load_from_pdf(tmp_path)
    with col2:
        if input_image is not None:
            st.markdown("### 🔧 Preprocessing Preview")
            from src.preprocessing.image_processor import ImageProcessor
            proc = ImageProcessor()
            preprocessed = proc.preprocess(input_image)
            st.image(preprocessed, caption="Preprocessed Image", use_column_width=True, clamp=True)

else:
    input_text = st.text_area(
        "Paste medical text",
        height=200,
        placeholder="e.g.\nDr. Sharma — Apollo Hospital\nPatient: Ramesh Patil\nDx: Viral Fever\nTab. Paracetamol 500mg BD x 5 days\nTab. Amoxicillin 500mg TDS x 7 days",
    )

# ── Run Pipeline ──────────────────────────────────────────────────────────────
run_col, _ = st.columns([1, 3])
with run_col:
    run_btn = st.button("🚀 Analyze Document", use_container_width=True)

if run_btn and (input_image is not None or (input_text and input_text.strip())):
    pipeline = load_pipeline()

    with st.spinner("Running AI pipeline..."):
        start = time.perf_counter()
        if input_image is not None:
            from src.preprocessing.image_processor import ImageProcessor
            proc = ImageProcessor()
            preprocessed = proc.preprocess(input_image)
            result = pipeline.process_image(preprocessed)
        else:
            result = pipeline.process_text(input_text)
        elapsed = time.perf_counter() - start

    st.success(f"✅ Analysis complete in {elapsed:.2f}s")
    st.markdown("---")

    # ── Top Metrics ───────────────────────────────────────────────────────────
    c1, c2, c3, c4 = st.columns(4)
    meta = result.get("pipeline_metadata", {})
    c1.metric("📋 Document Type", result.get("document_type", "—"))
    c2.metric("🔍 OCR Confidence", f"{meta.get('ocr_confidence', 0):.0%}")
    c3.metric("🏷️ Entities Found",
              sum(len(v) for v in [
                  result.get("entities", {}).get("medicines", []),
                  result.get("entities", {}).get("diagnosis", []),
                  result.get("entities", {}).get("lab_tests", []),
              ]))
    c4.metric("⚡ Total Time", f"{result.get('processing_time_seconds', 0):.2f}s")

    st.markdown("---")
    tab1, tab2, tab3, tab4, tab5 = st.tabs(
        ["📝 OCR Text", "🏷️ Entities", "🌐 Translation", "💊 Explanation", "📊 Raw JSON"]
    )

    # ── Tab 1: OCR Text ───────────────────────────────────────────────────────
    with tab1:
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("#### Raw OCR Output")
            st.text_area("", result.get("ocr_text", ""), height=250, key="raw_ocr")
        with col2:
            st.markdown("#### Corrected Text")
            st.text_area("", result.get("corrected_text", ""), height=250, key="corr_ocr")

    # ── Tab 2: Entities ───────────────────────────────────────────────────────
    with tab2:
        entities = result.get("entities", {})

        if entities.get("diagnosis"):
            st.markdown("**🔴 Diagnoses**")
            html = " ".join(
                f'<span class="entity-badge diagnosis-badge">{d}</span>'
                for d in entities["diagnosis"]
            )
            st.markdown(html, unsafe_allow_html=True)

        if entities.get("medicines"):
            st.markdown("**💊 Medicines**")
            for med in entities["medicines"]:
                name = med.get("name") or "Unknown"
                info = " | ".join(filter(None, [
                    med.get("dosage"), med.get("frequency"), med.get("duration")
                ]))
                st.markdown(
                    f'<span class="entity-badge medicine-badge">{name}</span> '
                    f'<small style="color:#94a3b8">{info}</small>',
                    unsafe_allow_html=True
                )

        if entities.get("lab_tests"):
            st.markdown("**🔬 Lab Tests**")
            for test in entities["lab_tests"]:
                name = test.get("test_name") or "Unknown"
                val = f'{test.get("value", "")} {test.get("unit", "")}'.strip()
                status = test.get("status", "Normal")
                icon = "✅" if status == "Normal" else "⚠️"
                st.markdown(
                    f'<span class="entity-badge test-badge">{name}</span> '
                    f'<small style="color:#94a3b8">{val} {icon} {status}</small>',
                    unsafe_allow_html=True
                )

        if entities.get("symptoms"):
            st.markdown("**🤒 Symptoms**")
            html = " ".join(
                f'<span class="entity-badge warning-badge">{s}</span>'
                for s in entities["symptoms"]
            )
            st.markdown(html, unsafe_allow_html=True)

        if result.get("validation_warnings"):
            st.warning("⚠️ " + " | ".join(result["validation_warnings"]))

    # ── Tab 3: Translation ────────────────────────────────────────────────────
    with tab3:
        if result.get("translation_hindi"):
            st.markdown("#### 🇮🇳 Hindi Translation")
            st.markdown(
                f'<div class="result-section">{result["translation_hindi"]}</div>',
                unsafe_allow_html=True
            )
        if result.get("translation_marathi"):
            st.markdown("#### 🌸 Marathi Translation")
            st.markdown(
                f'<div class="result-section">{result["translation_marathi"]}</div>',
                unsafe_allow_html=True
            )
        if not result.get("translation_hindi") and not result.get("translation_marathi"):
            st.info("Enable translations in the sidebar settings.")

    # ── Tab 4: Patient Explanation ────────────────────────────────────────────
    with tab4:
        explanation = result.get("patient_explanation", "")
        if explanation:
            st.markdown(
                f'<div class="explanation-card">{explanation}</div>',
                unsafe_allow_html=True
            )
        else:
            st.info("No explanation generated.")

    # ── Tab 5: Raw JSON ───────────────────────────────────────────────────────
    with tab5:
        st.json(result)
        json_str = json.dumps(result, indent=2, ensure_ascii=False)
        st.download_button(
            "⬇️ Download JSON",
            data=json_str,
            file_name="medical_analysis.json",
            mime="application/json",
        )

elif run_btn:
    st.warning("Please upload a document or enter text before clicking Analyze.")
