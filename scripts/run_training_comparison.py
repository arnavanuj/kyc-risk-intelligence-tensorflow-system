"""Run the updated training pipeline and produce a before/after comparison summary."""

from __future__ import annotations

import json
from pathlib import Path
from statistics import mean

from app.pipeline.training import TrainingConfig, train_hybrid_model

MODELS_DIR = Path("models")
TRAINING_LOG_BEFORE = MODELS_DIR / "training_log_before.jsonl"
INFERENCE_LOG_BEFORE = MODELS_DIR / "inference_log_before.jsonl"
COMPARISON_PATH = MODELS_DIR / "before_after_comparison.json"
AFTER_SUMMARY_PATH = MODELS_DIR / "after_training_summary.json"


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def baseline_summary() -> dict:
    training_rows = read_jsonl(TRAINING_LOG_BEFORE)
    inference_rows = read_jsonl(INFERENCE_LOG_BEFORE)

    last_training = training_rows[-1] if training_rows else {}
    recent_inference = inference_rows[-5:] if len(inference_rows) >= 5 else inference_rows

    if recent_inference:
        lstm_contribution = mean(float(row.get("lstm_contribution", 0.0)) for row in recent_inference)
        cnn_contribution = mean(float(row.get("cnn_contribution", 0.0)) for row in recent_inference)
    else:
        lstm_contribution = 0.0
        cnn_contribution = 0.0

    return {
        "accuracy": float(last_training.get("val_accuracy", 0.0)),
        "precision": float(last_training.get("val_precision", 0.0)),
        "recall": float(last_training.get("val_recall", 0.0)),
        "f1_score": float(last_training.get("val_f1_score", 0.0)),
        "lstm_contribution": lstm_contribution,
        "cnn_contribution": cnn_contribution,
        "source": "Existing validation/inference logs captured before the architecture refresh.",
    }


def main() -> None:
    before = baseline_summary()

    config = TrainingConfig(
        embedding_dim=64,
        lstm_units=128,
        cnn_filters=64,
        kernel_size=3,
        dropout_rate=0.2,
        epochs=8,
        batch_size=16,
        learning_rate=1e-3,
        optimizer_name="Adam",
        loss_name="BinaryCrossentropy",
        fusion_method="weighted_average",
        lstm_weight=0.5,
        cnn_weight=0.5,
        dataset_type="AllAgree",
        include_synthetic=True,
        synthetic_ratio=0.3,
        random_seed=42,
    )
    after = train_hybrid_model(config)

    after_metrics = after["test_metrics"]
    after_balance = after["branch_balance"]
    comparison = {
        "before": before,
        "after": {
            "accuracy": float(after_metrics["accuracy"]),
            "precision": float(after_metrics["precision"]),
            "recall": float(after_metrics["recall"]),
            "f1_score": float(after_metrics["f1_score"]),
            "lstm_contribution": float(after_balance["lstm_contribution"]),
            "cnn_contribution": float(after_balance["cnn_contribution"]),
            "alpha": float(after_balance["alpha"]),
            "imbalance_warning": str(after_balance["imbalance_warning"]),
        },
        "dataset_info": after["dataset_info"],
        "dataset_sizes": after["dataset_sizes"],
        "history_rows": len(after["history"]),
    }

    AFTER_SUMMARY_PATH.write_text(json.dumps(after, indent=2), encoding="utf-8")
    COMPARISON_PATH.write_text(json.dumps(comparison, indent=2), encoding="utf-8")
    print(json.dumps(comparison, indent=2))


if __name__ == "__main__":
    main()