"""
metrics.py — Evaluation metrics: CER, WER, classification, NER
"""

from typing import List, Dict, Any
import numpy as np


# ── OCR Metrics ───────────────────────────────────────────────────────────────

def compute_cer(reference: str, hypothesis: str) -> float:
    """
    Character Error Rate (CER) using dynamic programming edit distance.
    CER = (S + D + I) / N  where N = number of chars in reference.
    """
    ref = list(reference)
    hyp = list(hypothesis)
    return _edit_distance(ref, hyp) / max(len(ref), 1)


def compute_wer(reference: str, hypothesis: str) -> float:
    """
    Word Error Rate (WER).
    WER = (S + D + I) / N  where N = number of words in reference.
    """
    ref = reference.split()
    hyp = hypothesis.split()
    return _edit_distance(ref, hyp) / max(len(ref), 1)


def _edit_distance(ref: List, hyp: List) -> int:
    """Levenshtein edit distance between two sequences."""
    n, m = len(ref), len(hyp)
    dp = np.zeros((n + 1, m + 1), dtype=int)
    dp[:, 0] = np.arange(n + 1)
    dp[0, :] = np.arange(m + 1)
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if ref[i - 1] == hyp[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1])
    return int(dp[n][m])


def ocr_metrics_report(reference: str, hypothesis: str) -> Dict[str, float]:
    """Return CER and WER as a dictionary."""
    return {
        "cer": round(compute_cer(reference, hypothesis), 4),
        "wer": round(compute_wer(reference, hypothesis), 4),
    }


# ── Classification Metrics ────────────────────────────────────────────────────

def classification_report_dict(
    y_true: List[str], y_pred: List[str], labels: List[str]
) -> Dict[str, Any]:
    """Compute precision, recall, F1 per class and macro average."""
    from sklearn.metrics import classification_report
    report = classification_report(
        y_true, y_pred, labels=labels, output_dict=True, zero_division=0
    )
    return report


# ── NER Metrics (seqeval-compatible) ─────────────────────────────────────────

def ner_metrics_report(
    true_labels: List[List[str]], pred_labels: List[List[str]]
) -> Dict[str, float]:
    """
    Compute entity-level precision, recall, F1 using seqeval.
    Args:
        true_labels: list of label sequences (one per sentence)
        pred_labels: list of predicted label sequences
    Returns:
        dict with precision, recall, f1, accuracy
    """
    try:
        from seqeval.metrics import (
            precision_score, recall_score, f1_score, classification_report
        )
        return {
            "precision": round(precision_score(true_labels, pred_labels), 4),
            "recall": round(recall_score(true_labels, pred_labels), 4),
            "f1": round(f1_score(true_labels, pred_labels), 4),
            "report": classification_report(true_labels, pred_labels),
        }
    except ImportError:
        return {"error": "seqeval not installed. Run: pip install seqeval"}


# ── Latency Benchmarking ──────────────────────────────────────────────────────

def latency_benchmark(func, *args, n_runs: int = 10, **kwargs) -> Dict[str, float]:
    """Measure mean/std inference latency of a callable over n_runs."""
    import time
    times = []
    for _ in range(n_runs):
        start = time.perf_counter()
        func(*args, **kwargs)
        times.append(time.perf_counter() - start)
    arr = np.array(times)
    return {
        "mean_ms": round(float(arr.mean()) * 1000, 2),
        "std_ms": round(float(arr.std()) * 1000, 2),
        "min_ms": round(float(arr.min()) * 1000, 2),
        "max_ms": round(float(arr.max()) * 1000, 2),
        "n_runs": n_runs,
    }
