"""Dataset loading, cleansing, tokenization, and tf.data preparation."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple
import re

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer, tokenizer_from_json


PHRASEBANK_DIR = Path("data/FinancialPhraseBank-v1.0")
TOKENIZER_PATH = Path("models/tokenizer.json")
SYNTHETIC_DATASET_PATH = Path("data/synthetic_adverse_media.csv")
DATASET_FILE_MAP = {
    "50Agree": PHRASEBANK_DIR / "Sentences_50Agree.txt",
    "66Agree": PHRASEBANK_DIR / "Sentences_66Agree.txt",
    "75Agree": PHRASEBANK_DIR / "Sentences_75Agree.txt",
    "AllAgree": PHRASEBANK_DIR / "Sentences_AllAgree.txt",
}
LABEL_MAP = {
    "negative": 1,
    "positive": 0,
    "neutral": 0,
    0: 0,
    1: 0,
    2: 1,
}


@dataclass
class PreparedData:
    train_dataset: tf.data.Dataset
    val_dataset: tf.data.Dataset
    test_dataset: tf.data.Dataset
    tokenizer: Tokenizer
    x_train: np.ndarray
    x_val: np.ndarray
    x_test: np.ndarray
    y_train: np.ndarray
    y_val: np.ndarray
    y_test: np.ndarray
    word_index: Dict[str, int]
    max_sequence_length: int
    dataset_type: str
    total_samples: int
    class_distribution: Dict[str, int]
    dataset_composition: Dict[str, int | float | bool | str]


def normalize_text(text: str) -> str:
    text = str(text).strip().strip('"').strip("'")
    text = re.sub(r"\s+", " ", text)
    return text.lower()


def _safe_stratify_values(labels: pd.Series | np.ndarray, context: str) -> pd.Series | np.ndarray | None:
    label_series = pd.Series(labels)
    class_counts = Counter(label_series.tolist())
    if len(class_counts) > 1 and min(class_counts.values()) > 1:
        print(f"Stratify active for {context}: {dict(class_counts)}")
        return labels

    print(
        f"Warning: Stratified split disabled for {context}. "
        f"Least populated class has <= 1 member or only one class present: {dict(class_counts)}"
    )
    return None


def resolve_dataset_path(dataset_type: str) -> Path:
    if dataset_type not in DATASET_FILE_MAP:
        valid = ", ".join(DATASET_FILE_MAP.keys())
        raise ValueError(f"Unsupported dataset_type '{dataset_type}'. Expected one of: {valid}.")

    file_path = DATASET_FILE_MAP[dataset_type]
    if not file_path.exists():
        raise FileNotFoundError(f"Dataset file not found: {file_path}")
    return file_path


def load_phrasebank_from_txt(file_path: str | Path) -> pd.DataFrame:
    rows: List[Dict[str, int | str]] = []
    file_path = Path(file_path)

    with file_path.open("r", encoding="latin-1") as file:
        for line_number, raw_line in enumerate(file, start=1):
            line = raw_line.strip()
            if not line:
                continue
            if "@" not in line:
                raise ValueError(f"Invalid Financial PhraseBank row at line {line_number}: {line}")

            text, sentiment = line.rsplit("@", 1)
            cleaned_text = normalize_text(text)
            cleaned_sentiment = sentiment.strip().lower()
            label = LABEL_MAP.get(cleaned_sentiment)
            if label is None:
                raise ValueError(
                    f"Unsupported sentiment '{cleaned_sentiment}' at line {line_number} in {file_path}."
                )
            rows.append({"text": cleaned_text, "label": int(label), "source": "original"})

    dataset = pd.DataFrame(rows, columns=["text", "label", "source"])
    if dataset.empty:
        raise ValueError(f"No valid samples were loaded from {file_path}.")
    return dataset


def load_financial_phrasebank(dataset_type: str = "AllAgree") -> pd.DataFrame:
    data_path = resolve_dataset_path(dataset_type)
    dataset = load_phrasebank_from_txt(data_path)
    dataset = dataset.sample(frac=1.0, random_state=42).reset_index(drop=True)
    return dataset


def load_synthetic_dataset(path: str | Path = SYNTHETIC_DATASET_PATH) -> pd.DataFrame:
    data_path = Path(path)
    if not data_path.exists():
        raise FileNotFoundError(f"Synthetic dataset file not found: {data_path}")

    frame = pd.read_csv(data_path)
    required_columns = {"text", "label"}
    if not required_columns.issubset(frame.columns):
        raise ValueError(f"Synthetic dataset must include columns: {sorted(required_columns)}")

    dataset = frame.loc[:, ["text", "label"]].copy()
    dataset["text"] = dataset["text"].map(normalize_text)
    dataset["label"] = dataset["label"].map(lambda value: int(value) if str(value).strip().isdigit() else LABEL_MAP.get(str(value).strip().lower()))
    dataset["source"] = "synthetic"
    if dataset.empty:
        raise ValueError(f"No valid samples were loaded from {data_path}.")
    return dataset


def mix_datasets(
    original_frame: pd.DataFrame,
    synthetic_frame: pd.DataFrame,
    synthetic_ratio: float,
    random_state: int,
) -> Tuple[pd.DataFrame, Dict[str, int | float | bool | str]]:
    ratio = min(max(float(synthetic_ratio), 0.0), 0.95)
    if ratio <= 0.0:
        shuffled = original_frame.sample(frac=1.0, random_state=random_state).reset_index(drop=True)
        return shuffled, {
            "synthetic_enabled": False,
            "synthetic_ratio_requested": 0.0,
            "synthetic_ratio_actual": 0.0,
            "original_samples": int(len(original_frame)),
            "synthetic_samples": 0,
            "total_samples": int(len(original_frame)),
            "synthetic_path": str(SYNTHETIC_DATASET_PATH),
        }

    synthetic_target = int(round(len(original_frame) * ratio / max(1.0 - ratio, 1e-8)))
    synthetic_target = max(1, min(synthetic_target, len(synthetic_frame)))
    synthetic_stratify = _safe_stratify_values(
        synthetic_frame["label"],
        context="synthetic dataset sampling",
    )
    sampled_synthetic, _ = train_test_split(
        synthetic_frame,
        train_size=synthetic_target,
        random_state=random_state,
        stratify=synthetic_stratify,
    )

    combined = pd.concat([original_frame, sampled_synthetic], ignore_index=True)
    combined = combined.sample(frac=1.0, random_state=random_state).reset_index(drop=True)
    actual_ratio = len(sampled_synthetic) / max(len(combined), 1)
    return combined, {
        "synthetic_enabled": True,
        "synthetic_ratio_requested": ratio,
        "synthetic_ratio_actual": float(actual_ratio),
        "original_samples": int(len(original_frame)),
        "synthetic_samples": int(len(sampled_synthetic)),
        "total_samples": int(len(combined)),
        "synthetic_path": str(SYNTHETIC_DATASET_PATH),
    }


def stratified_split(
    frame: pd.DataFrame,
    test_size: float = 0.15,
    val_size: float = 0.15,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    labels = frame["label"]
    print(f"Total samples before split: {len(labels)}")
    print(f"Class distribution before split: {dict(Counter(labels.tolist()))}")

    first_stratify = _safe_stratify_values(labels, context="train/temp split")
    train_frame, temp_frame = train_test_split(
        frame,
        test_size=test_size + val_size,
        random_state=random_state,
        stratify=first_stratify,
    )

    adjusted_val_size = val_size / (test_size + val_size)
    second_stratify = _safe_stratify_values(temp_frame["label"], context="validation/test split")
    val_frame, test_frame = train_test_split(
        temp_frame,
        test_size=1.0 - adjusted_val_size,
        random_state=random_state,
        stratify=second_stratify,
    )

    print(f"Train distribution: {dict(Counter(train_frame['label'].tolist()))}")
    print(f"Validation distribution: {dict(Counter(val_frame['label'].tolist()))}")
    print(f"Test distribution: {dict(Counter(test_frame['label'].tolist()))}")
    return (
        train_frame.reset_index(drop=True),
        val_frame.reset_index(drop=True),
        test_frame.reset_index(drop=True),
    )


def fit_tokenizer(texts: List[str], vocab_size: int = 10000) -> Tokenizer:
    tokenizer = Tokenizer(num_words=vocab_size, oov_token="<OOV>")
    tokenizer.fit_on_texts(texts)
    return tokenizer


def save_tokenizer(tokenizer: Tokenizer, path: str | Path = TOKENIZER_PATH) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        file.write(tokenizer.to_json())


def load_tokenizer(path: str | Path = TOKENIZER_PATH) -> Tokenizer:
    with Path(path).open("r", encoding="utf-8") as file:
        return tokenizer_from_json(file.read())


def tokenize_and_pad(
    tokenizer: Tokenizer,
    texts: List[str],
    max_sequence_length: int | None = None,
) -> Tuple[np.ndarray, int]:
    sequences = tokenizer.texts_to_sequences(texts)
    if max_sequence_length is None:
        lengths = [len(sequence) for sequence in sequences]
        percentile_length = int(np.percentile(lengths, 95)) if lengths else 16
        max_sequence_length = max(16, min(percentile_length, 160))
    padded = pad_sequences(
        sequences,
        maxlen=max_sequence_length,
        padding="post",
        truncating="post",
    )
    return padded.astype(np.int32), int(max_sequence_length)


def make_tf_dataset(
    features: np.ndarray,
    labels: np.ndarray,
    batch_size: int,
    shuffle: bool,
    random_state: int,
) -> tf.data.Dataset:
    dataset = tf.data.Dataset.from_tensor_slices((features, labels))
    if shuffle:
        dataset = dataset.shuffle(
            buffer_size=len(features),
            seed=random_state,
            reshuffle_each_iteration=True,
        )
    return dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)


def prepare_datasets(
    batch_size: int,
    dataset_type: str = "AllAgree",
    vocab_size: int = 10000,
    include_synthetic: bool = False,
    synthetic_ratio: float = 0.3,
    synthetic_path: str | Path = SYNTHETIC_DATASET_PATH,
    random_state: int = 42,
) -> PreparedData:
    tf.keras.utils.set_random_seed(random_state)

    original_frame = load_financial_phrasebank(dataset_type=dataset_type)
    if include_synthetic:
        synthetic_frame = load_synthetic_dataset(path=synthetic_path)
        frame, dataset_composition = mix_datasets(
            original_frame=original_frame,
            synthetic_frame=synthetic_frame,
            synthetic_ratio=synthetic_ratio,
            random_state=random_state,
        )
    else:
        frame, dataset_composition = mix_datasets(
            original_frame=original_frame,
            synthetic_frame=original_frame.iloc[0:0].copy(),
            synthetic_ratio=0.0,
            random_state=random_state,
        )
        dataset_composition["synthetic_path"] = str(Path(synthetic_path))

    class_distribution = {
        "non_material_risk": int((frame["label"] == 0).sum()),
        "material_risk": int((frame["label"] == 1).sum()),
    }
    print(f"Dataset selected: {dataset_type}")
    print(f"Total samples: {len(frame)}")
    print(f"Class distribution: {class_distribution}")
    print(f"Dataset composition: {dataset_composition}")

    train_frame, val_frame, test_frame = stratified_split(frame, random_state=random_state)

    tokenizer = fit_tokenizer(train_frame["text"].tolist(), vocab_size=vocab_size)
    save_tokenizer(tokenizer)

    x_train, max_sequence_length = tokenize_and_pad(tokenizer, train_frame["text"].tolist())
    x_val, _ = tokenize_and_pad(
        tokenizer, val_frame["text"].tolist(), max_sequence_length=max_sequence_length
    )
    x_test, _ = tokenize_and_pad(
        tokenizer, test_frame["text"].tolist(), max_sequence_length=max_sequence_length
    )

    y_train = train_frame["label"].to_numpy(dtype=np.float32)
    y_val = val_frame["label"].to_numpy(dtype=np.float32)
    y_test = test_frame["label"].to_numpy(dtype=np.float32)

    return PreparedData(
        train_dataset=make_tf_dataset(
            x_train, y_train, batch_size=batch_size, shuffle=True, random_state=random_state
        ),
        val_dataset=make_tf_dataset(
            x_val, y_val, batch_size=batch_size, shuffle=False, random_state=random_state
        ),
        test_dataset=make_tf_dataset(
            x_test, y_test, batch_size=batch_size, shuffle=False, random_state=random_state
        ),
        tokenizer=tokenizer,
        x_train=x_train,
        x_val=x_val,
        x_test=x_test,
        y_train=y_train,
        y_val=y_val,
        y_test=y_test,
        word_index=tokenizer.word_index,
        max_sequence_length=max_sequence_length,
        dataset_type=dataset_type,
        total_samples=int(len(frame)),
        class_distribution=class_distribution,
        dataset_composition=dataset_composition,
    )


def tokenize_for_inference(
    text: str,
    tokenizer: Tokenizer,
    max_sequence_length: int,
) -> Tuple[List[str], List[int], np.ndarray]:
    cleaned_text = normalize_text(text)
    tokens = cleaned_text.split()
    sequence = tokenizer.texts_to_sequences([cleaned_text])[0]
    padded = pad_sequences(
        [sequence],
        maxlen=max_sequence_length,
        padding="post",
        truncating="post",
    ).astype(np.int32)
    return tokens, sequence, padded