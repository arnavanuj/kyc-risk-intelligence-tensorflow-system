"""Structured logging helpers for training and inference observability."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def append_jsonl(path: str | Path, payload: Dict[str, Any]) -> None:
    target = Path(path)
    ensure_parent(target)
    record = {"timestamp": datetime.now(timezone.utc).isoformat(), **payload}
    with target.open("a", encoding="utf-8") as file:
        file.write(json.dumps(record, ensure_ascii=True) + "\n")


@dataclass
class EpochLog:
    epoch: int
    loss: float
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    val_loss: float
    val_accuracy: float
    val_precision: float
    val_recall: float
    val_f1_score: float
    learning_rate: float
    lstm_score_mean: float = 0.0
    lstm_score_std: float = 0.0
    cnn_score_mean: float = 0.0
    cnn_score_std: float = 0.0
    attention_mean: float = 0.0
    attention_std: float = 0.0
    alpha: float = 0.5
    lstm_contribution: float = 0.5
    cnn_contribution: float = 0.5
    imbalance_warning: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class InferenceLog:
    input_text: str
    tokenized_words: List[str]
    token_ids: List[int]
    sequence_length: int
    model_confidence: float
    final_classification: str
    lstm_score: float
    cnn_score: float
    ensemble_score: float
    lstm_contribution: float
    cnn_contribution: float
    alpha: float = 0.5
    lstm_feature_mean: float = 0.0
    lstm_feature_std: float = 0.0
    cnn_feature_mean: float = 0.0
    cnn_feature_std: float = 0.0
    attention_mean: float = 0.0
    attention_std: float = 0.0
    imbalance_warning: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)