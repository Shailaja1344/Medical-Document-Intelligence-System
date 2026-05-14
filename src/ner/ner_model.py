"""
ner_model.py — BioBERT-based clinical Named Entity Recognition model
"""

from typing import Dict, Any, List, Optional
from pathlib import Path
import torch

from src.utils.config import (
    NER_MODEL_NAME, NER_MODEL_PATH, NER_LABELS,
    NER_LABEL2ID, NER_ID2LABEL, NER_MAX_LENGTH
)
from src.utils.logger import log


class ClinicalNERModel:
    """
    Token classification model for clinical entity extraction.
    Wraps dslim/bert-base-NER (BioBERT-compatible) with custom medical labels.
    Uses local fine-tuned checkpoint if available, else HuggingFace hub.
    """

    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.is_loaded = False

    def load(self) -> None:
        from transformers import AutoTokenizer, AutoModelForTokenClassification

        model_path = (
            str(NER_MODEL_PATH)
            if NER_MODEL_PATH.exists()
            else NER_MODEL_NAME
        )
        log.info(f"Loading NER model from: {model_path}")

        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForTokenClassification.from_pretrained(
            model_path,
            num_labels=len(NER_LABELS),
            id2label=NER_ID2LABEL,
            label2id=NER_LABEL2ID,
            ignore_mismatched_sizes=True,
        )
        self.model.to(self.device)
        self.model.eval()
        self.is_loaded = True
        log.info(f"NER model loaded on {self.device}")

    def predict_tokens(self, text: str) -> List[Dict[str, Any]]:
        """
        Run token classification on input text.
        Returns list of {word, entity, score, start, end}.
        """
        if not self.is_loaded:
            self.load()

        from transformers import pipeline as hf_pipeline

        pipe = hf_pipeline(
            "token-classification",
            model=self.model,
            tokenizer=self.tokenizer,
            aggregation_strategy="simple",
            device=0 if self.device == "cuda" else -1,
        )

        # Sliding window for long texts
        results = []
        max_len = NER_MAX_LENGTH - 10
        words = text.split()
        chunks = [
            " ".join(words[i: i + max_len])
            for i in range(0, len(words), max_len)
        ]

        for chunk in chunks:
            chunk_results = pipe(chunk)
            results.extend(chunk_results)

        return results

    def save(self, output_dir: Optional[Path] = None) -> None:
        path = output_dir or NER_MODEL_PATH
        path.mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(str(path))
        self.tokenizer.save_pretrained(str(path))
        log.info(f"NER model saved to {path}")
