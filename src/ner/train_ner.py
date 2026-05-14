"""
train_ner.py — HuggingFace Trainer-based NER fine-tuning script
"""

import json
import numpy as np
from pathlib import Path
from typing import List, Dict

from src.utils.config import (
    NER_MODEL_NAME, NER_MODEL_PATH, NER_LABELS,
    NER_LABEL2ID, NER_ID2LABEL,
    NER_TRAIN_EPOCHS, NER_BATCH_SIZE, NER_LR, NER_MAX_LENGTH,
    SYNTHETIC_DIR
)
from src.utils.logger import log


def load_conll_dataset(path: Path):
    """Load CoNLL-format NER dataset from file."""
    sentences, current = [], []
    with open(str(path), encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line == "":
                if current:
                    sentences.append(current)
                    current = []
            else:
                parts = line.split()
                if len(parts) >= 2:
                    current.append({"token": parts[0], "label": parts[-1]})
    if current:
        sentences.append(current)
    log.info(f"Loaded {len(sentences)} sentences from {path}")
    return sentences


def tokenize_and_align_labels(examples, tokenizer):
    """Align word-level labels to subword tokens."""
    tokenized = tokenizer(
        examples["tokens"],
        truncation=True,
        is_split_into_words=True,
        max_length=NER_MAX_LENGTH,
        padding="max_length",
    )
    all_labels = []
    for i, labels in enumerate(examples["ner_tags"]):
        word_ids = tokenized.word_ids(batch_index=i)
        prev_word_id, label_ids = None, []
        for word_id in word_ids:
            if word_id is None:
                label_ids.append(-100)
            elif word_id != prev_word_id:
                label_ids.append(labels[word_id])
            else:
                label_ids.append(-100)
            prev_word_id = word_id
        all_labels.append(label_ids)
    tokenized["labels"] = all_labels
    return tokenized


def train_ner():
    from transformers import (
        AutoTokenizer, AutoModelForTokenClassification,
        TrainingArguments, Trainer, DataCollatorForTokenClassification
    )
    from datasets import Dataset
    from seqeval.metrics import f1_score, precision_score, recall_score

    conll_path = SYNTHETIC_DIR / "ner_dataset.conll"
    if not conll_path.exists():
        raise FileNotFoundError(
            f"NER dataset not found at {conll_path}. "
            "Run: python scripts/generate_synthetic_data.py"
        )

    sentences = load_conll_dataset(conll_path)
    split = int(0.85 * len(sentences))
    train_sents, val_sents = sentences[:split], sentences[split:]

    def sents_to_dict(sents):
        return {
            "tokens": [[tok["token"] for tok in s] for s in sents],
            "ner_tags": [
                [NER_LABEL2ID.get(tok["label"], 0) for tok in s]
                for s in sents
            ],
        }

    tokenizer = AutoTokenizer.from_pretrained(NER_MODEL_NAME)
    train_ds = Dataset.from_dict(sents_to_dict(train_sents)).map(
        lambda x: tokenize_and_align_labels(x, tokenizer), batched=True
    )
    val_ds = Dataset.from_dict(sents_to_dict(val_sents)).map(
        lambda x: tokenize_and_align_labels(x, tokenizer), batched=True
    )

    model = AutoModelForTokenClassification.from_pretrained(
        NER_MODEL_NAME,
        num_labels=len(NER_LABELS),
        id2label=NER_ID2LABEL,
        label2id=NER_LABEL2ID,
        ignore_mismatched_sizes=True,
    )

    def compute_metrics(p):
        preds = np.argmax(p.predictions, axis=2)
        labels = p.label_ids
        true_preds, true_labels = [], []
        for pred_row, label_row in zip(preds, labels):
            tp, tl = [], []
            for p_id, l_id in zip(pred_row, label_row):
                if l_id != -100:
                    tp.append(NER_ID2LABEL[p_id])
                    tl.append(NER_ID2LABEL[l_id])
            true_preds.append(tp)
            true_labels.append(tl)
        return {
            "precision": precision_score(true_labels, true_preds),
            "recall": recall_score(true_labels, true_preds),
            "f1": f1_score(true_labels, true_preds),
        }

    args = TrainingArguments(
        output_dir=str(NER_MODEL_PATH),
        num_train_epochs=NER_TRAIN_EPOCHS,
        per_device_train_batch_size=NER_BATCH_SIZE,
        per_device_eval_batch_size=NER_BATCH_SIZE,
        learning_rate=NER_LR,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        report_to="none",
        seed=42,
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        tokenizer=tokenizer,
        data_collator=DataCollatorForTokenClassification(tokenizer),
        compute_metrics=compute_metrics,
    )

    log.info("Starting NER fine-tuning...")
    trainer.train()
    metrics = trainer.evaluate()
    trainer.save_model(str(NER_MODEL_PATH))
    tokenizer.save_pretrained(str(NER_MODEL_PATH))
    log.info(f"NER eval metrics: {json.dumps(metrics, indent=2)}")
    return metrics


if __name__ == "__main__":
    train_ner()
