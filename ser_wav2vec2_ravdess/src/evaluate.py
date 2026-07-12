"""
Evaluation script: runs the trained model on the held-out test set,
prints/saves a classification report, and plots + saves a confusion matrix.

Usage:
    python -m src.evaluate
"""

import os

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix

from . import config


def run_evaluation(trainer, test_ds, output_dir: str = config.OUTPUT_DIR):
    os.makedirs(output_dir, exist_ok=True)

    print("Running inference on the test set...")
    predictions = trainer.predict(test_ds)
    logits = predictions.predictions
    labels = predictions.label_ids
    preds = np.argmax(logits, axis=1)

    label_names = config.LABELS

    report = classification_report(labels, preds, target_names=label_names, digits=4)
    print("\nClassification report (test set):\n")
    print(report)

    report_path = os.path.join(output_dir, "classification_report.txt")
    with open(report_path, "w") as f:
        f.write(report)
    print(f"Saved classification report to {report_path}")

    cm = confusion_matrix(labels, preds)
    cm_path = plot_confusion_matrix(cm, label_names, output_dir)
    print(f"Saved confusion matrix to {cm_path}")

    overall_accuracy = float((preds == labels).mean())
    print(f"Overall test accuracy: {overall_accuracy:.4f}")

    return {
        "report": report,
        "confusion_matrix": cm,
        "accuracy": overall_accuracy,
        "predictions": preds,
        "labels": labels,
    }


def plot_confusion_matrix(cm, label_names, output_dir: str) -> str:
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=label_names,
        yticklabels=label_names,
        cbar=True,
    )
    plt.xlabel("Predicted label")
    plt.ylabel("True label")
    plt.title("Confusion Matrix — Speech Emotion Recognition (wav2vec2 + RAVDESS)")
    plt.tight_layout()

    save_path = os.path.join(output_dir, "confusion_matrix.png")
    plt.savefig(save_path, dpi=150)
    plt.close()
    return save_path


if __name__ == "__main__":
    # Running this module standalone will train the model first (if not
    # already trained) and then evaluate it on the test split.
    from .train import main as train_main

    trainer, test_ds = train_main()
    run_evaluation(trainer, test_ds)
