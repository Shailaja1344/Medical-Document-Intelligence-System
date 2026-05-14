"""
export_onnx.py — ONNX export and dynamic quantization for NER model
Includes latency benchmarking: original vs quantized BioBERT
"""

import sys
import time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parents[1]))

import torch
import numpy as np
from src.utils.config import NER_MODEL_PATH, NER_MODEL_NAME, MODELS_DIR
from src.utils.logger import log

ONNX_PATH = MODELS_DIR / "ner_model.onnx"
QUANTIZED_PATH = MODELS_DIR / "ner_model_quantized.onnx"
SAMPLE_TEXT = (
    "Patient Ramesh Patil prescribed Tab. Paracetamol 500mg BD for 5 days "
    "for Viral Fever diagnosed by Dr. Rajesh Sharma at City General Hospital."
)


def export_to_onnx():
    """Export the NER model to ONNX format."""
    from transformers import AutoTokenizer, AutoModelForTokenClassification

    model_path = str(NER_MODEL_PATH) if NER_MODEL_PATH.exists() else NER_MODEL_NAME
    log.info(f"Loading model from: {model_path}")

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForTokenClassification.from_pretrained(model_path, ignore_mismatched_sizes=True)
    model.eval()

    inputs = tokenizer(
        SAMPLE_TEXT, return_tensors="pt", truncation=True, max_length=128, padding="max_length"
    )

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    log.info(f"Exporting to ONNX: {ONNX_PATH}")

    torch.onnx.export(
        model,
        (inputs["input_ids"], inputs["attention_mask"]),
        str(ONNX_PATH),
        input_names=["input_ids", "attention_mask"],
        output_names=["logits"],
        dynamic_axes={
            "input_ids": {0: "batch_size", 1: "sequence_length"},
            "attention_mask": {0: "batch_size", 1: "sequence_length"},
            "logits": {0: "batch_size"},
        },
        opset_version=14,
    )
    log.info("ONNX export complete")
    return model, tokenizer, inputs


def apply_dynamic_quantization():
    """Apply dynamic quantization to the exported ONNX model."""
    try:
        from onnxruntime.quantization import quantize_dynamic, QuantType
        log.info("Applying dynamic quantization...")
        quantize_dynamic(
            str(ONNX_PATH),
            str(QUANTIZED_PATH),
            weight_type=QuantType.QInt8,
        )
        log.info(f"Quantized model saved: {QUANTIZED_PATH}")
    except ImportError:
        log.warning("onnxruntime.quantization not available — skipping quantization")


def benchmark(model, tokenizer, inputs, n_runs=20):
    """Benchmark PyTorch vs ONNX vs Quantized inference latency."""
    import onnxruntime as ort

    input_ids = inputs["input_ids"].numpy()
    attn_mask = inputs["attention_mask"].numpy()

    results = {}

    # PyTorch latency
    times = []
    with torch.no_grad():
        for _ in range(n_runs):
            t = time.perf_counter()
            model(**{k: v for k, v in inputs.items()})
            times.append((time.perf_counter() - t) * 1000)
    results["pytorch_ms"] = {
        "mean": round(np.mean(times), 2),
        "std": round(np.std(times), 2),
    }

    # ONNX Runtime latency
    if ONNX_PATH.exists():
        sess = ort.InferenceSession(str(ONNX_PATH))
        times = []
        for _ in range(n_runs):
            t = time.perf_counter()
            sess.run(None, {"input_ids": input_ids, "attention_mask": attn_mask})
            times.append((time.perf_counter() - t) * 1000)
        results["onnx_ms"] = {
            "mean": round(np.mean(times), 2),
            "std": round(np.std(times), 2),
        }

    # Quantized ONNX latency
    if QUANTIZED_PATH.exists():
        sess_q = ort.InferenceSession(str(QUANTIZED_PATH))
        times = []
        for _ in range(n_runs):
            t = time.perf_counter()
            sess_q.run(None, {"input_ids": input_ids, "attention_mask": attn_mask})
            times.append((time.perf_counter() - t) * 1000)
        results["quantized_onnx_ms"] = {
            "mean": round(np.mean(times), 2),
            "std": round(np.std(times), 2),
        }

    # Model size comparison
    size_info = {}
    orig_model_size = sum(
        p.numel() * p.element_size() for p in model.parameters()
    ) / (1024 ** 2)
    size_info["pytorch_mb"] = round(orig_model_size, 2)
    if ONNX_PATH.exists():
        size_info["onnx_mb"] = round(ONNX_PATH.stat().st_size / (1024 ** 2), 2)
    if QUANTIZED_PATH.exists():
        size_info["quantized_mb"] = round(QUANTIZED_PATH.stat().st_size / (1024 ** 2), 2)

    return {"latency": results, "model_sizes_mb": size_info}


if __name__ == "__main__":
    import json
    log.info("Starting ONNX export and quantization benchmark...")
    model, tokenizer, inputs = export_to_onnx()
    apply_dynamic_quantization()
    report = benchmark(model, tokenizer, inputs)
    log.info("Benchmark Results:")
    log.info(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
