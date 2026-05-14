"""
train_all.py — One-shot training runner for classifier and NER
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parents[1]))

from src.utils.logger import log
from src.utils.config import SYNTHETIC_DIR


def main():
    # Step 1: Generate data if missing
    clf_csv = SYNTHETIC_DIR / "classifier_dataset.csv"
    ner_conll = SYNTHETIC_DIR / "ner_dataset.conll"

    if not clf_csv.exists() or not ner_conll.exists():
        log.info("Synthetic datasets not found — generating now...")
        from scripts.generate_synthetic_data import (
            generate_classifier_dataset, generate_ner_dataset
        )
        generate_classifier_dataset(n_per_class=200)
        generate_ner_dataset(n_samples=600)

    # Step 2: Train TF-IDF SVM classifier
    log.info("=" * 60)
    log.info("STEP 1: Training TF-IDF SVM Document Classifier")
    log.info("=" * 60)
    from src.classifier.train_classifier import load_dataset, train_tfidf_svm
    texts, labels = load_dataset(clf_csv)
    svm_metrics = train_tfidf_svm(texts, labels)
    log.info(f"SVM Test Accuracy: {svm_metrics.get('test_accuracy', 'N/A'):.4f}")
    log.info(f"SVM CV F1: {svm_metrics.get('cv_f1_mean', 'N/A'):.4f}")

    # Step 3: Skip NER training to keep it lightweight (Use Pre-trained directly)
    log.info("=" * 60)
    log.info("STEP 2: Skipping NER Training — Using pre-trained BioBERT for inference")
    log.info("=" * 60)

    log.info("=" * 60)
    log.info("Training complete. Run the API with:")
    log.info("  python app.py")
    log.info("  or: uvicorn app:app --reload")
    log.info("=" * 60)


if __name__ == "__main__":
    main()
