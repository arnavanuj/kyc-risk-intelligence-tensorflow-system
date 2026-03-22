"""Training orchestration for the hybrid KYC model."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List

import numpy as np
import tensorflow as tf

from app.models.hybrid_model import build_hybrid_model
from app.observability.logging import EpochLog, append_jsonl
from app.observability.metrics import F1Score, calculate_binary_metrics
from app.pipeline.preprocessing import SYNTHETIC_DATASET_PATH, prepare_datasets


MODEL_PATH = Path("models/hybrid_model.h5")
CONFIG_PATH = Path("models/model_config.json")
TRAINING_LOG_PATH = Path("models/training_log.jsonl")
IMBALANCE_THRESHOLD = 0.90


@dataclass
class TrainingConfig:
    embedding_dim: int
    lstm_units: int
    cnn_filters: int
    kernel_size: int
    dropout_rate: float
    epochs: int
    batch_size: int
    learning_rate: float
    optimizer_name: str
    loss_name: str
    fusion_method: str
    lstm_weight: float
    cnn_weight: float
    dataset_type: str = "AllAgree"
    vocab_size: int = 10000
    include_synthetic: bool = False
    synthetic_ratio: float = 0.3
    synthetic_path: str = str(SYNTHETIC_DATASET_PATH)
    random_seed: int = 42


class EpochMetricsLogger(tf.keras.callbacks.Callback):
    """Captures structured per-epoch metrics for UI and file logging."""

    def __init__(
        self,
        log_path: Path,
        explain_model: tf.keras.Model,
        validation_features: np.ndarray,
        epoch_update_fn: Callable[[List[Dict[str, Any]]], None] | None = None,
    ):
        super().__init__()
        self.log_path = log_path
        self.explain_model = explain_model
        self.validation_features = validation_features
        self.history: List[Dict[str, Any]] = []
        self.epoch_update_fn = epoch_update_fn

    @staticmethod
    def _mean_std(values: np.ndarray) -> tuple[float, float]:
        values = np.asarray(values, dtype=np.float32)
        return float(np.mean(values)), float(np.std(values))

    def _get_alpha(self) -> float:
        fusion_layer = self.model.get_layer("ensemble_output")
        if hasattr(fusion_layer, "get_alpha"):
            return float(tf.keras.backend.get_value(fusion_layer.get_alpha()))
        return 0.5

    def _get_branch_diagnostics(self) -> Dict[str, float | str]:
        outputs = self.explain_model.predict(self.validation_features, verbose=0)
        lstm_mean, lstm_std = self._mean_std(outputs["lstm_score"])
        cnn_mean, cnn_std = self._mean_std(outputs["cnn_score"])
        attention_mean, attention_std = self._mean_std(outputs["lstm_attention"])

        if self.model.get_layer("ensemble_output").__class__.__name__ == "TrainableScalarFusion":
            alpha = self._get_alpha()
            lstm_contribution = alpha
            cnn_contribution = 1.0 - alpha
        else:
            total_mean = max(lstm_mean + cnn_mean, 1e-8)
            lstm_contribution = lstm_mean / total_mean
            cnn_contribution = cnn_mean / total_mean
            alpha = 0.5

        imbalance_warning = ""
        if cnn_contribution > IMBALANCE_THRESHOLD:
            imbalance_warning = "Model imbalance detected: CNN dominance. LSTM under-utilized."

        return {
            "lstm_score_mean": lstm_mean,
            "lstm_score_std": lstm_std,
            "cnn_score_mean": cnn_mean,
            "cnn_score_std": cnn_std,
            "attention_mean": attention_mean,
            "attention_std": attention_std,
            "alpha": float(alpha),
            "lstm_contribution": float(lstm_contribution),
            "cnn_contribution": float(cnn_contribution),
            "imbalance_warning": imbalance_warning,
        }

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        optimizer = self.model.optimizer
        learning_rate = tf.keras.backend.get_value(optimizer.learning_rate)
        diagnostics = self._get_branch_diagnostics()

        payload = EpochLog(
            epoch=int(epoch + 1),
            loss=float(logs.get("loss", 0.0)),
            accuracy=float(logs.get("accuracy", 0.0)),
            precision=float(logs.get("precision", 0.0)),
            recall=float(logs.get("recall", 0.0)),
            f1_score=float(logs.get("f1_score", 0.0)),
            val_loss=float(logs.get("val_loss", 0.0)),
            val_accuracy=float(logs.get("val_accuracy", 0.0)),
            val_precision=float(logs.get("val_precision", 0.0)),
            val_recall=float(logs.get("val_recall", 0.0)),
            val_f1_score=float(logs.get("val_f1_score", 0.0)),
            learning_rate=float(learning_rate),
            lstm_score_mean=float(diagnostics["lstm_score_mean"]),
            lstm_score_std=float(diagnostics["lstm_score_std"]),
            cnn_score_mean=float(diagnostics["cnn_score_mean"]),
            cnn_score_std=float(diagnostics["cnn_score_std"]),
            attention_mean=float(diagnostics["attention_mean"]),
            attention_std=float(diagnostics["attention_std"]),
            alpha=float(diagnostics["alpha"]),
            lstm_contribution=float(diagnostics["lstm_contribution"]),
            cnn_contribution=float(diagnostics["cnn_contribution"]),
            imbalance_warning=str(diagnostics["imbalance_warning"]),
        ).to_dict()

        self.history.append(payload)
        append_jsonl(self.log_path, payload)
        if self.epoch_update_fn is not None:
            self.epoch_update_fn(self.history)


def _build_optimizer(name: str, learning_rate: float) -> tf.keras.optimizers.Optimizer:
    if name.lower() == "adam":
        return tf.keras.optimizers.Adam(learning_rate=learning_rate)
    if name.lower() == "sgd":
        return tf.keras.optimizers.SGD(learning_rate=learning_rate, momentum=0.9)
    raise ValueError(f"Unsupported optimizer: {name}")


def _compile_model(model: tf.keras.Model, optimizer_name: str, learning_rate: float) -> None:
    model.compile(
        optimizer=_build_optimizer(optimizer_name, learning_rate),
        loss=tf.keras.losses.BinaryCrossentropy(),
        metrics=[
            tf.keras.metrics.BinaryAccuracy(name="accuracy"),
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
            F1Score(name="f1_score"),
        ],
    )


def _build_callbacks(
    log_path: Path,
    explain_model: tf.keras.Model,
    validation_features: np.ndarray,
    epoch_update_fn: Callable[[List[Dict[str, Any]]], None] | None,
) -> List[tf.keras.callbacks.Callback]:
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    return [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=3,
            restore_best_weights=True,
            verbose=0,
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(MODEL_PATH),
            monitor="val_loss",
            save_best_only=True,
            verbose=0,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=2,
            min_lr=1e-5,
            verbose=0,
        ),
        EpochMetricsLogger(
            log_path=log_path,
            explain_model=explain_model,
            validation_features=validation_features,
            epoch_update_fn=epoch_update_fn,
        ),
    ]


def _save_training_config(config: TrainingConfig, max_sequence_length: int) -> None:
    CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    payload = asdict(config)
    payload["max_sequence_length"] = max_sequence_length
    payload["imbalance_threshold"] = IMBALANCE_THRESHOLD
    with CONFIG_PATH.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2)


def _compute_class_weight(y_train: np.ndarray) -> Dict[int, float]:
    labels = np.asarray(y_train, dtype=np.int32)
    counts = np.bincount(labels, minlength=2)
    total = max(int(counts.sum()), 1)
    weights: Dict[int, float] = {}
    for label, count in enumerate(counts):
        weights[label] = float(total / max(2 * int(count), 1))
    return weights


def _collect_branch_balance(
    explain_model: tf.keras.Model,
    features: np.ndarray,
    fusion_method: str,
    model: tf.keras.Model,
) -> Dict[str, float | str]:
    outputs = explain_model.predict(features, verbose=0)
    lstm_scores = np.asarray(outputs["lstm_score"], dtype=np.float32)
    cnn_scores = np.asarray(outputs["cnn_score"], dtype=np.float32)
    attention = np.asarray(outputs["lstm_attention"], dtype=np.float32)

    lstm_score_mean = float(np.mean(lstm_scores))
    lstm_score_std = float(np.std(lstm_scores))
    cnn_score_mean = float(np.mean(cnn_scores))
    cnn_score_std = float(np.std(cnn_scores))
    attention_mean = float(np.mean(attention))
    attention_std = float(np.std(attention))

    alpha = 0.5
    if fusion_method == "weighted_average" and hasattr(model.get_layer("ensemble_output"), "get_alpha"):
        alpha = float(tf.keras.backend.get_value(model.get_layer("ensemble_output").get_alpha()))
        lstm_contribution = alpha
        cnn_contribution = 1.0 - alpha
    else:
        total_score = max(lstm_score_mean + cnn_score_mean, 1e-8)
        lstm_contribution = lstm_score_mean / total_score
        cnn_contribution = cnn_score_mean / total_score

    imbalance_warning = ""
    if cnn_contribution > IMBALANCE_THRESHOLD:
        imbalance_warning = "Model imbalance detected: CNN dominance. LSTM under-utilized."

    return {
        "lstm_score_mean": lstm_score_mean,
        "lstm_score_std": lstm_score_std,
        "cnn_score_mean": cnn_score_mean,
        "cnn_score_std": cnn_score_std,
        "attention_mean": attention_mean,
        "attention_std": attention_std,
        "alpha": alpha,
        "lstm_contribution": float(lstm_contribution),
        "cnn_contribution": float(cnn_contribution),
        "imbalance_warning": imbalance_warning,
    }


def train_hybrid_model(
    config: TrainingConfig,
    epoch_update_fn: Callable[[List[Dict[str, Any]]], None] | None = None,
) -> Dict[str, Any]:
    tf.keras.utils.set_random_seed(config.random_seed)
    prepared = prepare_datasets(
        batch_size=config.batch_size,
        dataset_type=config.dataset_type,
        vocab_size=config.vocab_size,
        include_synthetic=config.include_synthetic,
        synthetic_ratio=config.synthetic_ratio,
        synthetic_path=config.synthetic_path,
        random_state=config.random_seed,
    )
    artifacts = build_hybrid_model(
        vocab_size=min(config.vocab_size, len(prepared.word_index) + 2),
        sequence_length=prepared.max_sequence_length,
        embedding_dim=config.embedding_dim,
        lstm_units=config.lstm_units,
        cnn_filters=config.cnn_filters,
        kernel_size=config.kernel_size,
        dropout_rate=config.dropout_rate,
        fusion_method=config.fusion_method,
        lstm_weight=config.lstm_weight,
        cnn_weight=config.cnn_weight,
    )
    _compile_model(
        artifacts.model,
        optimizer_name=config.optimizer_name,
        learning_rate=config.learning_rate,
    )

    TRAINING_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    if TRAINING_LOG_PATH.exists():
        TRAINING_LOG_PATH.unlink()

    callbacks = _build_callbacks(
        TRAINING_LOG_PATH,
        explain_model=artifacts.explain_model,
        validation_features=prepared.x_val,
        epoch_update_fn=epoch_update_fn,
    )
    epoch_logger = next(
        callback for callback in callbacks if isinstance(callback, EpochMetricsLogger)
    )

    class_weight = _compute_class_weight(prepared.y_train)
    history = artifacts.model.fit(
        prepared.train_dataset,
        validation_data=prepared.val_dataset,
        epochs=config.epochs,
        callbacks=callbacks,
        class_weight=class_weight,
        verbose=0,
    )

    test_predictions = artifacts.model.predict(prepared.test_dataset, verbose=0).ravel()
    final_metrics = calculate_binary_metrics(prepared.y_test, test_predictions)
    branch_balance = _collect_branch_balance(
        explain_model=artifacts.explain_model,
        features=prepared.x_test,
        fusion_method=config.fusion_method,
        model=artifacts.model,
    )

    print(
        "Branch balance after training:",
        {
            "lstm_contribution": round(float(branch_balance["lstm_contribution"]), 4),
            "cnn_contribution": round(float(branch_balance["cnn_contribution"]), 4),
            "alpha": round(float(branch_balance["alpha"]), 4),
        },
    )
    if branch_balance["imbalance_warning"]:
        print(branch_balance["imbalance_warning"])

    _save_training_config(config, prepared.max_sequence_length)

    return {
        "model_path": str(MODEL_PATH),
        "history": epoch_logger.history,
        "keras_history": history.history,
        "test_metrics": final_metrics.as_dict(),
        "max_sequence_length": prepared.max_sequence_length,
        "vocab_size": min(config.vocab_size, len(prepared.word_index) + 2),
        "dataset_sizes": {
            "train": int(len(prepared.y_train)),
            "validation": int(len(prepared.y_val)),
            "test": int(len(prepared.y_test)),
        },
        "dataset_info": {
            "dataset_type": prepared.dataset_type,
            "total_samples": prepared.total_samples,
            "class_distribution": prepared.class_distribution,
            "dataset_composition": prepared.dataset_composition,
            "class_weight": class_weight,
        },
        "branch_balance": branch_balance,
    }