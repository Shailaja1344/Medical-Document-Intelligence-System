"""
train_classifier.py — Training script for document type classifiers
"""

import json
from pathlib import Path
from typing import List, Tuple
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import pandas as pd

from src.classifier.tfidf_svm_classifier import TFIDFSVMClassifier
from src.classifier.bert_classifier import BERTDocumentClassifier
from src.utils.config import SYNTHETIC_DIR, DOCUMENT_CLASSES
from src.utils.logger import log


def load_dataset(csv_path: Path) -> Tuple[List[str], List[str]]:
    """Load synthetic dataset CSV with 'text' and 'label' columns."""
    df = pd.read_csv(str(csv_path))
    df = df.dropna(subset=["text", "label"])
    df = df[df["label"].isin(DOCUMENT_CLASSES)]
    log.info(f"Loaded {len(df)} samples from {csv_path}")
    return df["text"].tolist(), df["label"].tolist()


def train_tfidf_svm(texts: List[str], labels: List[str]) -> dict:
    """Train and evaluate TF-IDF SVM classifier."""
    clf = TFIDFSVMClassifier()
    X_train, X_test, y_train, y_test = train_test_split(
        texts, labels, test_size=0.2, random_state=42, stratify=labels
    )
    metrics = clf.train(X_train, y_train)

    # Evaluate on test set
    preds = [clf.predict(t)["document_type"] for t in X_test]
    report = classification_report(
        y_test, preds, labels=DOCUMENT_CLASSES, output_dict=True, zero_division=0
    )
    metrics["test_report"] = report
    metrics["test_accuracy"] = report["accuracy"]

    clf.save()
    log.info(f"TF-IDF SVM test accuracy: {report['accuracy']:.4f}")
    return metrics


def train_bert_classifier(texts: List[str], labels: List[str]) -> dict:
    """Fine-tune DistilBERT classifier on synthetic dataset using HuggingFace Trainer."""
    from transformers import (
        AutoTokenizer, AutoModelForSequenceClassification,
        TrainingArguments, Trainer, DataCollatorWithPadding
    )
    from datasets import Dataset
    import numpy as np
    from sklearn.metrics import accuracy_score, f1_score

    from src.utils.config import (
        CLASSIFIER_MODEL_NAME, CLASSIFIER_MODEL_PATH,
        CLASSIFIER_TRAIN_EPOCHS, CLASSIFIER_BATCH_SIZE,
        CLASSIFIER_LR, CLASSIFIER_MAX_LENGTH
    )
    from src.classifier.bert_classifier import LABEL2ID, ID2LABEL

    X_train, X_val, y_train, y_val = train_test_split(
        texts, labels, test_size=0.15, random_state=42, stratify=labels
    )

    tokenizer = AutoTokenizer.from_pretrained(CLASSIFIER_MODEL_NAME)

    def tokenize(batch):
        return tokenizer(
            batch["text"], truncation=True, padding="max_length",
            max_length=CLASSIFIER_MAX_LENGTH
        )

    def make_dataset(txts, lbls):
        return Dataset.from_dict({
            "text": txts,
            "label": [LABEL2ID[l] for l in lbls]
        }).map(tokenize, batched=True)

    train_ds = make_dataset(X_train, y_train)
    val_ds = make_dataset(X_val, y_val)

    model = AutoModelForSequenceClassification.from_pretrained(
        CLASSIFIER_MODEL_NAME,
        num_labels=len(DOCUMENT_CLASSES),
        id2label=ID2LABEL,
        label2id=LABEL2ID,
        ignore_mismatched_sizes=True,
    )

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        return {
            "accuracy": accuracy_score(labels, preds),
            "f1": f1_score(labels, preds, average="macro"),
        }

    args = TrainingArguments(
        output_dir=str(CLASSIFIER_MODEL_PATH),
        num_train_epochs=CLASSIFIER_TRAIN_EPOCHS,
        per_device_train_batch_size=CLASSIFIER_BATCH_SIZE,
        per_device_eval_batch_size=CLASSIFIER_BATCH_SIZE,
        learning_rate=CLASSIFIER_LR,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        logging_dir=str(CLASSIFIER_MODEL_PATH / "logs"),
        report_to="none",
        seed=42,
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        tokenizer=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer),
        compute_metrics=compute_metrics,
    )

    log.info("Starting BERT classifier fine-tuning...")
    trainer.train()
    metrics = trainer.evaluate()
    trainer.save_model(str(CLASSIFIER_MODEL_PATH))
    tokenizer.save_pretrained(str(CLASSIFIER_MODEL_PATH))
    log.info(f"BERT classifier eval metrics: {metrics}")
    return metrics


if __name__ == "__main__":
    csv_path = SYNTHETIC_DIR / "classifier_dataset.csv"
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {csv_path}. "
            "Run: python scripts/generate_synthetic_data.py"
        )
    texts, labels = load_dataset(csv_path)
    log.info("=== Training TF-IDF SVM ===")
    svm_metrics = train_tfidf_svm(texts, labels)
    log.info(f"SVM Results: {json.dumps(svm_metrics, indent=2)}")
    log.info("=== Training BERT Classifier ===")
    bert_metrics = train_bert_classifier(texts, labels)
    log.info(f"BERT Results: {bert_metrics}")
