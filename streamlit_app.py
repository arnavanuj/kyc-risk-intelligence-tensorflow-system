from __future__ import annotations

from typing import Dict, List

import pandas as pd
import plotly.express as px
import streamlit as st

from app.pipeline.inference import predict_text
from app.pipeline.training import TrainingConfig, train_hybrid_model


st.set_page_config(
    page_title="KYC Risk Intelligence TensorFlow System",
    page_icon="KYC",
    layout="wide",
)


FUSION_OPTIONS = {
    "Simple Average": "simple_average",
    "Weighted Average": "weighted_average",
    "Concatenation + Dense": "concat_dense",
}
DATASET_OPTIONS = {
    "All Agree": "AllAgree",
    "75% Agree": "75Agree",
    "66% Agree": "66Agree",
    "50% Agree": "50Agree",
}


if "training_result" not in st.session_state:
    st.session_state.training_result = None
if "latest_epoch_logs" not in st.session_state:
    st.session_state.latest_epoch_logs = []
if "latest_prediction" not in st.session_state:
    st.session_state.latest_prediction = None
if "training_run_id" not in st.session_state:
    st.session_state.training_run_id = 0
if "inference_run_id" not in st.session_state:
    st.session_state.inference_run_id = 0


def build_chart_key(prefix: str, key_suffix: str = "default") -> str:
    return f"{prefix}_{key_suffix}"


def render_plotly_chart(figure, prefix: str, key_suffix: str = "default") -> None:
    key = build_chart_key(prefix, key_suffix)
    print(f"Rendering chart with key: {key}")
    st.plotly_chart(figure, use_container_width=True, key=key)


def history_frame(history: List[Dict]) -> pd.DataFrame:
    if not history:
        return pd.DataFrame()
    return pd.DataFrame(history)


def render_metric_chart(frame: pd.DataFrame, metric: str, title: str, key_suffix: str = "default"):
    if frame.empty:
        st.info("Training history will appear here after training.")
        return
    chart_frame = frame[["epoch", metric, f"val_{metric}"]].melt(
        id_vars="epoch",
        var_name="split",
        value_name="value",
    )
    figure = px.line(
        chart_frame,
        x="epoch",
        y="value",
        color="split",
        markers=True,
        title=title,
    )
    figure.update_layout(legend_title_text="Metric")
    render_plotly_chart(figure, prefix=f"metric_{metric}", key_suffix=key_suffix)


def render_train_validation_gap(frame: pd.DataFrame, key_suffix: str = "default"):
    if frame.empty:
        st.info("Gap analysis will appear here after training.")
        return
    gap_frame = pd.DataFrame(
        {
            "epoch": frame["epoch"],
            "accuracy_gap": frame["accuracy"] - frame["val_accuracy"],
            "loss_gap": frame["val_loss"] - frame["loss"],
        }
    ).melt(id_vars="epoch", var_name="gap_type", value_name="gap_value")
    figure = px.bar(
        gap_frame,
        x="epoch",
        y="gap_value",
        color="gap_type",
        barmode="group",
        title="Train vs Validation Gap",
    )
    render_plotly_chart(figure, prefix="train_validation_gap", key_suffix=key_suffix)


def render_branch_balance_chart(balance: Dict[str, float], key_suffix: str = "default"):
    contribution_frame = pd.DataFrame(
        {
            "branch": ["LSTM", "CNN"],
            "contribution": [
                balance.get("lstm_contribution", 0.5),
                balance.get("cnn_contribution", 0.5),
            ],
        }
    )
    figure = px.pie(
        contribution_frame,
        names="branch",
        values="contribution",
        title="Branch Contribution Share",
    )
    render_plotly_chart(figure, prefix="branch_balance", key_suffix=key_suffix)


def overfitting_status(frame: pd.DataFrame) -> tuple[str, str]:
    if frame.empty:
        return "Waiting for training", "Train the model to start overfitting monitoring."
    last_row = frame.iloc[-1]
    gap = float(last_row["accuracy"] - last_row["val_accuracy"])
    if gap > 0.10:
        return "Overfitting detected", f"Training accuracy exceeds validation accuracy by {gap:.2%}."
    return "Healthy training", f"Accuracy gap is controlled at {gap:.2%}."


st.title("KYC Risk Intelligence TensorFlow System")
st.caption(
    "Enterprise-style adverse media classification using a balanced hybrid LSTM + CNN ensemble with observability and explainability."
)

with st.sidebar:
    st.header("Training Control Panel")
    dataset_label = st.selectbox("Dataset Selection", options=list(DATASET_OPTIONS.keys()), index=0)
    dataset_type = DATASET_OPTIONS[dataset_label]
    embedding_dim = st.select_slider("Embedding dimension", options=[32, 64, 128], value=64)
    lstm_units = st.select_slider("LSTM units", options=[64, 128, 256], value=128)
    cnn_filters = st.select_slider("CNN filters", options=[32, 64, 128], value=64)
    kernel_size = st.select_slider("Kernel size", options=[3, 5], value=3)
    dropout_rate = st.slider("Dropout", min_value=0.1, max_value=0.5, value=0.2, step=0.05)

    st.subheader("Dataset Mixing")
    include_synthetic = st.checkbox("Include synthetic dataset", value=True)
    synthetic_ratio = st.slider(
        "Synthetic mix ratio",
        min_value=0.0,
        max_value=0.7,
        value=0.3,
        step=0.05,
        disabled=not include_synthetic,
    )

    st.subheader("Training")
    epochs = st.slider("Epochs", min_value=1, max_value=20, value=8, step=1)
    batch_size = st.select_slider("Batch size", options=[8, 16, 32, 64], value=16)
    learning_rate = st.selectbox("Learning rate", options=[1e-2, 1e-3, 1e-4], index=1, format_func=lambda value: f"{value:.0e}")
    optimizer_name = st.selectbox("Optimizer", options=["Adam", "SGD"], index=0)
    loss_name = st.selectbox("Loss", options=["BinaryCrossentropy"], index=0)

    st.subheader("Ensemble")
    fusion_label = st.selectbox("Fusion method", options=list(FUSION_OPTIONS.keys()), index=1)
    fusion_method = FUSION_OPTIONS[fusion_label]
    lstm_weight = st.slider("Initial LSTM weight", min_value=0.0, max_value=1.0, value=0.5, step=0.05)
    cnn_weight = 1.0 - lstm_weight
    st.slider(
        "Initial CNN weight",
        min_value=0.0,
        max_value=1.0,
        value=float(cnn_weight),
        step=0.05,
        disabled=True,
    )
    train_button = st.button("Train Model", type="primary", use_container_width=True)

training_tab, summary_tab, inference_tab, explain_tab = st.tabs(
    [
        "Training Visualization",
        "Model Performance Summary",
        "Inference Section",
        "Explainability Panel",
    ]
)

with training_tab:
    live_status = st.empty()
    live_logs_placeholder = st.empty()
    chart_col_1, chart_col_2 = st.columns(2)
    chart_col_3, chart_col_4 = st.columns(2)
    chart_col_5, chart_col_6 = st.columns(2)

    if train_button:
        st.session_state.training_run_id += 1
        config = TrainingConfig(
            embedding_dim=embedding_dim,
            lstm_units=lstm_units,
            cnn_filters=cnn_filters,
            kernel_size=kernel_size,
            dropout_rate=dropout_rate,
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=float(learning_rate),
            optimizer_name=optimizer_name,
            loss_name=loss_name,
            fusion_method=fusion_method,
            lstm_weight=float(lstm_weight),
            cnn_weight=float(cnn_weight),
            dataset_type=dataset_type,
            include_synthetic=include_synthetic,
            synthetic_ratio=float(synthetic_ratio),
        )

        def update_live_view(epoch_history: List[Dict]):
            st.session_state.latest_epoch_logs = epoch_history
            frame = history_frame(epoch_history)
            latest = frame.iloc[-1]
            live_status.info(
                f"Epoch {len(epoch_history)} completed on {dataset_label}. "
                f"Latest validation F1: {latest['val_f1_score']:.4f}. "
                f"LSTM contribution: {latest['lstm_contribution']:.2%}."
            )
            live_logs_placeholder.dataframe(frame, use_container_width=True)

        with st.spinner("Training hybrid model and capturing observability logs..."):
            st.session_state.training_result = train_hybrid_model(
                config=config,
                epoch_update_fn=update_live_view,
            )
        live_status.success("Training run complete. Best model persisted to models/hybrid_model.h5")

    result = st.session_state.training_result
    frame = history_frame((result or {}).get("history", st.session_state.latest_epoch_logs))
    if not frame.empty:
        live_logs_placeholder.dataframe(frame, use_container_width=True)

    training_key_suffix = f"training_{st.session_state.training_run_id}"
    with chart_col_1:
        render_metric_chart(frame, "loss", "Loss vs Epoch", key_suffix=training_key_suffix)
    with chart_col_2:
        render_metric_chart(frame, "accuracy", "Accuracy vs Epoch", key_suffix=training_key_suffix)
    with chart_col_3:
        render_metric_chart(frame, "precision", "Precision vs Epoch", key_suffix=training_key_suffix)
    with chart_col_4:
        render_metric_chart(frame, "recall", "Recall vs Epoch", key_suffix=training_key_suffix)
    with chart_col_5:
        render_metric_chart(frame, "f1_score", "F1 Score vs Epoch", key_suffix=training_key_suffix)
    with chart_col_6:
        render_train_validation_gap(frame, key_suffix=training_key_suffix)

with summary_tab:
    result = st.session_state.training_result
    if result is None:
        st.info("Run training to populate the performance summary.")
    else:
        metrics = result["test_metrics"]
        metric_cols = st.columns(4)
        metric_cols[0].metric("Final Accuracy", f"{metrics['accuracy']:.3f}")
        metric_cols[1].metric("Precision", f"{metrics['precision']:.3f}")
        metric_cols[2].metric("Recall", f"{metrics['recall']:.3f}")
        metric_cols[3].metric("F1 Score", f"{metrics['f1_score']:.3f}")
        st.write("Dataset split sizes")
        st.json(result["dataset_sizes"])
        st.write("Dataset information")
        st.json(result["dataset_info"])

        balance = result.get("branch_balance", {})
        balance_cols = st.columns(4)
        balance_cols[0].metric("LSTM contribution", f"{balance.get('lstm_contribution', 0.0):.2%}")
        balance_cols[1].metric("CNN contribution", f"{balance.get('cnn_contribution', 0.0):.2%}")
        balance_cols[2].metric("Alpha", f"{balance.get('alpha', 0.5):.3f}")
        balance_cols[3].metric("Attention std", f"{balance.get('attention_std', 0.0):.4f}")
        render_branch_balance_chart(balance, key_suffix=f"summary_{st.session_state.training_run_id}")
        if balance.get("imbalance_warning"):
            st.warning(balance["imbalance_warning"])
        else:
            st.success("Branch balance monitor: no severe CNN dominance detected.")

        frame = history_frame(result["history"])
        status, detail = overfitting_status(frame)
        if status == "Overfitting detected":
            st.error(f"{status}: {detail}")
        else:
            st.success(f"{status}: {detail}")

with inference_tab:
    inference_text = st.text_area(
        "Input financial news text",
        value="Bank faces regulatory scrutiny after money laundering compliance failures surfaced.",
        height=120,
    )
    if st.button("Run Inference", use_container_width=False):
        try:
            st.session_state.inference_run_id += 1
            prediction = predict_text(inference_text)
            score_cols = st.columns(5)
            score_cols[0].metric("LSTM score", f"{prediction['lstm_score']:.3f}")
            score_cols[1].metric("CNN score", f"{prediction['cnn_score']:.3f}")
            score_cols[2].metric("Ensemble score", f"{prediction['ensemble_score']:.3f}")
            score_cols[3].metric("Alpha", f"{prediction['alpha']:.3f}")
            score_cols[4].metric("Final label", prediction["final_label"])
            if prediction.get("imbalance_warning"):
                st.warning(prediction["imbalance_warning"])
            st.session_state.latest_prediction = prediction
        except Exception as exc:
            st.error(str(exc))

with explain_tab:
    prediction = st.session_state.get("latest_prediction")
    if not prediction:
        st.info("Run inference to inspect tokens, sequence encoding, branch contributions, and embedding details.")
    else:
        explain_col_1, explain_col_2 = st.columns(2)
        with explain_col_1:
            st.write("Tokenized words")
            st.code(str(prediction["tokens"]))
            st.write("Encoded sequence")
            st.code(str(prediction["token_ids"]))
            st.write(f"Sequence length: {prediction['sequence_length']}")
            st.write(f"Embedding shape: {prediction['embedding_shape']}")
        with explain_col_2:
            st.metric("LSTM contribution %", f"{prediction['lstm_contribution'] * 100:.2f}%")
            st.metric("CNN contribution %", f"{prediction['cnn_contribution'] * 100:.2f}%")
            st.metric("Attention mean", f"{prediction['attention_mean']:.4f}")
            st.metric("Attention std", f"{prediction['attention_std']:.4f}")
            render_branch_balance_chart(prediction, key_suffix=f"explain_{st.session_state.inference_run_id}")
            stats_frame = pd.DataFrame(
                {
                    "branch": ["LSTM", "CNN"],
                    "mean": [prediction["lstm_feature_mean"], prediction["cnn_feature_mean"]],
                    "std": [prediction["lstm_feature_std"], prediction["cnn_feature_std"]],
                }
            )
            st.dataframe(stats_frame, use_container_width=True)
            if prediction.get("imbalance_warning"):
                st.warning(prediction["imbalance_warning"])

frame = history_frame((st.session_state.training_result or {}).get("history", []))
status, detail = overfitting_status(frame)
st.subheader("Overfitting Monitor")
if status == "Overfitting detected":
    st.error(f"{status}: {detail}")
elif status == "Healthy training":
    st.success(f"{status}: {detail}")
else:
    st.info(detail)