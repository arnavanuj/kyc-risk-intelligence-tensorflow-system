"""Model loading and explainable inference utilities."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import numpy as np
import tensorflow as tf

from app.models.hybrid_model import (
    TrainableScalarFusion,
    WeightedAverageFusion,
    build_explain_model_from_trained_model,
)
from app.observability.logging import InferenceLog, append_jsonl
from app.observability.metrics import F1Score
from app.pipeline.preprocessing import load_tokenizer, tokenize_for_inference


MODEL_PATH = Path("models/hybrid_model.h5")
CONFIG_PATH = Path("models/model_config.json")
INFERENCE_LOG_PATH = Path("models/inference_log.jsonl")


def load_model_artifacts() -> Dict[str, Any]:
    if not MODEL_PATH.exists():
        raise FileNotFoundError("No trained model found. Train the model before running inference.")
    if not CONFIG_PATH.exists():
        raise FileNotFoundError("Model configuration not found. Retrain the model to regenerate artifacts.")

    with CONFIG_PATH.open("r", encoding="utf-8") as file:
        config = json.load(file)

    model = tf.keras.models.load_model(
        MODEL_PATH,
        custom_objects={
            "F1Score": F1Score,
            "WeightedAverageFusion": WeightedAverageFusion,
            "TrainableScalarFusion": TrainableScalarFusion,
        },
        compile=False,
    )
    explain_model = build_explain_model_from_trained_model(model)
    tokenizer = load_tokenizer()
    return {"model": model, "explain_model": explain_model, "tokenizer": tokenizer, "config": config}


def _mean_std(values: np.ndarray) -> tuple[float, float]:
    values = np.asarray(values, dtype=np.float32)
    return float(np.mean(values)), float(np.std(values))


def _estimate_contributions(
    fusion_method: str,
    lstm_score: float,
    cnn_score: float,
    model: tf.keras.Model,
) -> Dict[str, float]:
    fusion_layer = model.get_layer("ensemble_output")
    if fusion_method == "weighted_average" and hasattr(fusion_layer, "get_alpha"):
        alpha = float(tf.keras.backend.get_value(fusion_layer.get_alpha()))
        return {
            "lstm_contribution": alpha,
            "cnn_contribution": 1.0 - alpha,
            "alpha": alpha,
        }

    total_score = max(lstm_score + cnn_score, 1e-8)
    return {
        "lstm_contribution": lstm_score / total_score,
        "cnn_contribution": cnn_score / total_score,
        "alpha": 0.5,
    }


def predict_text(text: str) -> Dict[str, Any]:
    artifacts = load_model_artifacts()
    tokenizer = artifacts["tokenizer"]
    config = artifacts["config"]
    explain_model = artifacts["explain_model"]
    model = artifacts["model"]

    tokens, token_ids, padded = tokenize_for_inference(
        text=text,
        tokenizer=tokenizer,
        max_sequence_length=int(config["max_sequence_length"]),
    )
    outputs = explain_model.predict(padded, verbose=0)

    lstm_score = float(outputs["lstm_score"].ravel()[0])
    cnn_score = float(outputs["cnn_score"].ravel()[0])
    ensemble_score = float(outputs["ensemble_output"].ravel()[0])
    embedding_shape = list(outputs["embedding"].shape)
    contribution = _estimate_contributions(
        fusion_method=str(config["fusion_method"]),
        lstm_score=lstm_score,
        cnn_score=cnn_score,
        model=model,
    )
    lstm_feature_mean, lstm_feature_std = _mean_std(outputs["lstm_features"])
    cnn_feature_mean, cnn_feature_std = _mean_std(outputs["cnn_features"])
    attention_mean, attention_std = _mean_std(outputs["lstm_attention"])
    imbalance_warning = ""
    if contribution["cnn_contribution"] > float(config.get("imbalance_threshold", 0.90)):
        imbalance_warning = "Model imbalance detected: CNN dominance. LSTM under-utilized."
    final_label = "Material Risk" if ensemble_score >= 0.5 else "Non-Material Risk"

    log = InferenceLog(
        input_text=text,
        tokenized_words=tokens,
        token_ids=token_ids,
        sequence_length=len(token_ids),
        model_confidence=ensemble_score,
        final_classification=final_label,
        lstm_score=lstm_score,
        cnn_score=cnn_score,
        ensemble_score=ensemble_score,
        lstm_contribution=contribution["lstm_contribution"],
        cnn_contribution=contribution["cnn_contribution"],
        alpha=contribution["alpha"],
        lstm_feature_mean=lstm_feature_mean,
        lstm_feature_std=lstm_feature_std,
        cnn_feature_mean=cnn_feature_mean,
        cnn_feature_std=cnn_feature_std,
        attention_mean=attention_mean,
        attention_std=attention_std,
        imbalance_warning=imbalance_warning,
    )
    append_jsonl(INFERENCE_LOG_PATH, log.to_dict())

    return {
        "input_text": text,
        "tokens": tokens,
        "token_ids": token_ids,
        "sequence_length": len(token_ids),
        "embedding_shape": embedding_shape,
        "lstm_score": lstm_score,
        "cnn_score": cnn_score,
        "ensemble_score": ensemble_score,
        "final_label": final_label,
        "lstm_contribution": contribution["lstm_contribution"],
        "cnn_contribution": contribution["cnn_contribution"],
        "alpha": contribution["alpha"],
        "lstm_feature_mean": lstm_feature_mean,
        "lstm_feature_std": lstm_feature_std,
        "cnn_feature_mean": cnn_feature_mean,
        "cnn_feature_std": cnn_feature_std,
        "attention_mean": attention_mean,
        "attention_std": attention_std,
        "imbalance_warning": imbalance_warning,
    }