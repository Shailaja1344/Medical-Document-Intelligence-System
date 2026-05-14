"""
bert_classifier.py — DistilBERT-based document type classifier
"""

from typing import Dict, Any, List, Optional
from pathlib import Path
import torch
import torch.nn.functional as F

from src.utils.config import (
    DOCUMENT_CLASSES, CLASSIFIER_MODEL_NAME,
    CLASSIFIER_MODEL_PATH, CLASSIFIER_MAX_LENGTH
)
from src.utils.logger import log

LABEL2ID = {lbl: i for i, lbl in enumerate(DOCUMENT_CLASSES)}
ID2LABEL = {i: lbl for lbl, i in LABEL2ID.items()}


class BERTDocumentClassifier:
    """
    DistilBERT sequence classification for medical document type detection.
    Runs inference from a fine-tuned checkpoint.
    Falls back to HuggingFace hub model if local checkpoint absent.
    """

    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.is_loaded = False

    def load(self) -> None:
        from transformers import AutoTokenizer, AutoModelForSequenceClassification

        model_path = (
            str(CLASSIFIER_MODEL_PATH)
            if CLASSIFIER_MODEL_PATH.exists()
            else CLASSIFIER_MODEL_NAME
        )
        log.info(f"Loading BERT classifier from: {model_path}")

        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_path,
            num_labels=len(DOCUMENT_CLASSES),
            id2label=ID2LABEL,
            label2id=LABEL2ID,
            ignore_mismatched_sizes=True,
        )
        self.model.to(self.device)
        self.model.eval()
        self.is_loaded = True
        log.info("BERT classifier loaded successfully")

    def predict(self, text: str) -> Dict[str, Any]:
        if not self.is_loaded:
            self.load()

        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=CLASSIFIER_MAX_LENGTH,
            padding="max_length",
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            logits = self.model(**inputs).logits
            probs = F.softmax(logits, dim=-1).squeeze().cpu().numpy()

        pred_id = int(probs.argmax())
        return {
            "document_type": ID2LABEL[pred_id],
            "confidence": round(float(probs.max()), 4),
            "all_scores": {
                ID2LABEL[i]: round(float(p), 4) for i, p in enumerate(probs)
            },
            "model": "bert_classifier",
        }

    def predict_batch(self, texts: List[str]) -> List[Dict[str, Any]]:
        return [self.predict(t) for t in texts]

    def save(self, output_dir: Optional[Path] = None) -> None:
        path = output_dir or CLASSIFIER_MODEL_PATH
        path.mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(str(path))
        self.tokenizer.save_pretrained(str(path))
        log.info(f"BERT classifier saved to {path}")
