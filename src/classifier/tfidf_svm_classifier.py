"""
tfidf_svm_classifier.py — TF-IDF + SVM document type baseline classifier
"""

import joblib
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import cross_val_score

from src.utils.config import (
    DOCUMENT_CLASSES, TFIDF_SVM_MODEL_PATH, TFIDF_VECTORIZER_PATH
)
from src.utils.logger import log


class TFIDFSVMClassifier:
    """
    TF-IDF + LinearSVC baseline classifier for medical document types.
    5 classes: Prescription, Lab Report, Discharge Summary, Medical Bill, Diagnostic Scan Report
    """

    def __init__(self):
        self.pipeline: Optional[Pipeline] = None
        self.label_encoder = LabelEncoder()
        self.label_encoder.classes_ = np.array(DOCUMENT_CLASSES)
        self.is_fitted = False

    def build(self) -> Pipeline:
        """Build sklearn pipeline."""
        self.pipeline = Pipeline([
            ("tfidf", TfidfVectorizer(
                ngram_range=(1, 2),
                max_features=20000,
                sublinear_tf=True,
                strip_accents="unicode",
                analyzer="word",
                token_pattern=r"(?u)\b[a-zA-Z]\w+\b",
                min_df=2,
            )),
            ("clf", LinearSVC(
                C=1.0,
                class_weight="balanced",
                max_iter=2000,
            )),
        ])
        return self.pipeline

    def train(self, texts: List[str], labels: List[str]) -> Dict[str, Any]:
        """Train pipeline and return cross-validation metrics."""
        if self.pipeline is None:
            self.build()

        encoded_labels = self.label_encoder.transform(labels)
        log.info(f"Training TF-IDF SVM on {len(texts)} samples...")

        cv_scores = cross_val_score(
            self.pipeline, texts, encoded_labels, cv=5, scoring="f1_macro"
        )
        self.pipeline.fit(texts, encoded_labels)
        self.is_fitted = True

        metrics = {
            "cv_f1_mean": round(float(cv_scores.mean()), 4),
            "cv_f1_std": round(float(cv_scores.std()), 4),
            "n_samples": len(texts),
            "classes": DOCUMENT_CLASSES,
        }
        log.info(f"TF-IDF SVM CV F1: {metrics['cv_f1_mean']:.4f} ± {metrics['cv_f1_std']:.4f}")
        return metrics

    def predict(self, text: str) -> Dict[str, Any]:
        """Predict document type for a single text."""
        if not self.is_fitted:
            self.load()

        label_id = self.pipeline.predict([text])[0]
        label = self.label_encoder.classes_[label_id]

        # LinearSVC confidence via decision function
        decision = self.pipeline.decision_function([text])[0]
        proba = self._softmax(decision)
        confidence = float(proba.max())

        return {
            "document_type": str(label),
            "confidence": round(confidence, 4),
            "all_scores": {
                cls: round(float(p), 4)
                for cls, p in zip(DOCUMENT_CLASSES, proba)
            },
            "model": "tfidf_svm",
        }

    def save(self) -> None:
        """Persist trained pipeline to disk."""
        TFIDF_SVM_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.pipeline, str(TFIDF_SVM_MODEL_PATH))
        log.info(f"Saved TF-IDF SVM to {TFIDF_SVM_MODEL_PATH}")

    def load(self) -> None:
        """Load persisted pipeline."""
        if not TFIDF_SVM_MODEL_PATH.exists():
            raise FileNotFoundError(
                f"No trained model at {TFIDF_SVM_MODEL_PATH}. "
                "Run: python scripts/train_all.py"
            )
        self.pipeline = joblib.load(str(TFIDF_SVM_MODEL_PATH))
        self.is_fitted = True
        log.info("Loaded TF-IDF SVM classifier")

    @staticmethod
    def _softmax(x: np.ndarray) -> np.ndarray:
        e = np.exp(x - x.max())
        return e / e.sum()
