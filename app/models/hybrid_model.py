"""Hybrid LSTM + CNN model definition for adverse media classification."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import tensorflow as tf
from tensorflow.keras import layers


@tf.keras.utils.register_keras_serializable(package="kyc_model")
class WeightedAverageFusion(layers.Layer):
    """Combines branch scores with user-defined weights."""

    def __init__(self, lstm_weight: float = 0.5, cnn_weight: float = 0.5, **kwargs):
        super().__init__(**kwargs)
        self.lstm_weight = float(lstm_weight)
        self.cnn_weight = float(cnn_weight)

    def call(self, inputs, **kwargs):
        lstm_score, cnn_score = inputs
        total = self.lstm_weight + self.cnn_weight
        total = total if total != 0 else 1.0
        return ((self.lstm_weight * lstm_score) + (self.cnn_weight * cnn_score)) / total

    def get_config(self):
        config = super().get_config()
        config.update(
            {"lstm_weight": self.lstm_weight, "cnn_weight": self.cnn_weight}
        )
        return config


@tf.keras.utils.register_keras_serializable(package="kyc_model")
class TrainableScalarFusion(layers.Layer):
    """Learns a constrained interpolation weight between LSTM and CNN scores."""

    def __init__(self, initial_alpha: float = 0.5, **kwargs):
        super().__init__(**kwargs)
        self.initial_alpha = min(max(float(initial_alpha), 1e-4), 1.0 - 1e-4)

    def build(self, input_shape):
        initial_logit = tf.math.log(self.initial_alpha / (1.0 - self.initial_alpha))
        self.alpha_logit = self.add_weight(
            name="alpha_logit",
            shape=(),
            initializer=tf.keras.initializers.Constant(initial_logit),
            trainable=True,
        )
        super().build(input_shape)

    def call(self, inputs, **kwargs):
        lstm_score, cnn_score = inputs
        alpha = tf.nn.sigmoid(self.alpha_logit)
        return (alpha * lstm_score) + ((1.0 - alpha) * cnn_score)

    def get_alpha(self) -> tf.Tensor:
        return tf.nn.sigmoid(self.alpha_logit)

    def get_config(self):
        config = super().get_config()
        config.update({"initial_alpha": self.initial_alpha})
        return config


@dataclass
class HybridModelArtifacts:
    model: tf.keras.Model
    explain_model: tf.keras.Model
    config: Dict[str, int | float | str]


def build_hybrid_model(
    vocab_size: int,
    sequence_length: int,
    embedding_dim: int,
    lstm_units: int,
    cnn_filters: int,
    kernel_size: int,
    dropout_rate: float,
    fusion_method: str,
    lstm_weight: float,
    cnn_weight: float,
) -> HybridModelArtifacts:
    """Builds the trainable hybrid model and a companion explainability model."""

    inputs = layers.Input(shape=(sequence_length,), dtype="int32", name="token_ids")
    embedding = layers.Embedding(
        input_dim=vocab_size,
        output_dim=embedding_dim,
        name="shared_embedding",
    )(inputs)

    effective_lstm_dropout = min(dropout_rate, 0.3)
    lstm_sequence = layers.Bidirectional(
        layers.LSTM(
            lstm_units,
            return_sequences=True,
            dropout=effective_lstm_dropout,
            recurrent_dropout=0.0,
        ),
        name="lstm_branch",
    )(embedding)
    attention_context = layers.Attention(name="lstm_attention")(
        [lstm_sequence, lstm_sequence]
    )
    lstm_features = layers.GlobalAveragePooling1D(name="lstm_pool")(attention_context)
    lstm_features = layers.Dropout(
        effective_lstm_dropout, name="lstm_dropout"
    )(lstm_features)
    lstm_features = layers.LayerNormalization(name="lstm_norm")(lstm_features)
    lstm_score = layers.Dense(1, activation="sigmoid", name="lstm_score")(lstm_features)

    cnn_features = layers.Conv1D(
        filters=cnn_filters,
        kernel_size=kernel_size,
        activation="relu",
        padding="same",
        name="cnn_conv",
    )(embedding)
    cnn_features = layers.GlobalMaxPooling1D(name="cnn_pool")(cnn_features)
    cnn_features = layers.Dropout(dropout_rate, name="cnn_dropout")(cnn_features)
    cnn_features = layers.LayerNormalization(name="cnn_norm")(cnn_features)
    cnn_score = layers.Dense(1, activation="sigmoid", name="cnn_score")(cnn_features)

    if fusion_method == "simple_average":
        final_output = layers.Average(name="ensemble_output")([lstm_score, cnn_score])
    elif fusion_method == "weighted_average":
        initial_alpha = lstm_weight / max(lstm_weight + cnn_weight, 1e-8)
        final_output = TrainableScalarFusion(
            initial_alpha=initial_alpha,
            name="ensemble_output",
        )([lstm_score, cnn_score])
    elif fusion_method == "concat_dense":
        fusion_features = layers.Concatenate(name="fusion_concat")(
            [lstm_features, cnn_features]
        )
        fusion_features = layers.Dense(
            max(lstm_units * 2, cnn_filters),
            activation="relu",
            name="fusion_dense",
        )(fusion_features)
        fusion_features = layers.Dropout(dropout_rate, name="fusion_dropout")(
            fusion_features
        )
        final_output = layers.Dense(
            1, activation="sigmoid", name="ensemble_output"
        )(fusion_features)
    else:
        raise ValueError(f"Unsupported fusion method: {fusion_method}")

    model = tf.keras.Model(inputs=inputs, outputs=final_output, name="hybrid_kyc_model")
    explain_model = tf.keras.Model(
        inputs=inputs,
        outputs={
            "embedding": embedding,
            "lstm_attention": attention_context,
            "lstm_features": lstm_features,
            "cnn_features": cnn_features,
            "lstm_score": lstm_score,
            "cnn_score": cnn_score,
            "ensemble_output": final_output,
        },
        name="hybrid_kyc_explain_model",
    )

    config = {
        "vocab_size": vocab_size,
        "sequence_length": sequence_length,
        "embedding_dim": embedding_dim,
        "lstm_units": lstm_units,
        "cnn_filters": cnn_filters,
        "kernel_size": kernel_size,
        "dropout_rate": dropout_rate,
        "effective_lstm_dropout": effective_lstm_dropout,
        "fusion_method": fusion_method,
        "lstm_weight": lstm_weight,
        "cnn_weight": cnn_weight,
    }
    return HybridModelArtifacts(model=model, explain_model=explain_model, config=config)


def build_explain_model_from_trained_model(model: tf.keras.Model) -> tf.keras.Model:
    """Recreates the explainability view from a persisted trained model."""

    return tf.keras.Model(
        inputs=model.input,
        outputs={
            "embedding": model.get_layer("shared_embedding").output,
            "lstm_attention": model.get_layer("lstm_attention").output,
            "lstm_features": model.get_layer("lstm_norm").output,
            "cnn_features": model.get_layer("cnn_norm").output,
            "lstm_score": model.get_layer("lstm_score").output,
            "cnn_score": model.get_layer("cnn_score").output,
            "ensemble_output": model.get_layer("ensemble_output").output,
        },
        name="loaded_hybrid_kyc_explain_model",
    )