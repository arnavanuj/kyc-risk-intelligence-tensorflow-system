# KYC Risk Intelligence TensorFlow System

## Project Overview
This project analyzes financial and adverse media text to classify whether a case should be treated as **Material Risk** or **Non-Material Risk**. It combines a hybrid deep learning architecture with a Streamlit interface so teams can train, inspect, and review model behavior in one place.

The current system uses a shared text embedding layer, a sequence-oriented LSTM branch, a pattern-oriented CNN branch, attention for richer long-text understanding, and ensemble logic for final scoring. It also includes branch-level explainability, structured observability logs, synthetic data generation, and optional dataset mixing.

## Tech Stack
- Python
- TensorFlow / Keras
- Streamlit
- Scikit-learn
- Pandas / NumPy
- Plotly

## Architecture
The model is designed to combine local phrase detection with sequence-level context.

### CNN branch
The CNN branch focuses on short local patterns such as adverse terms, suspicious phrase fragments, and compact lexical cues that often appear in financial media.

### LSTM branch
The LSTM branch focuses on sequence and progression. It is better suited for understanding how allegations evolve across a sentence or multi-sentence narrative.

### Attention layer
Attention is applied on top of the LSTM sequence output to help the model retain the most relevant parts of longer texts instead of relying only on the final recurrent state.

### Ensemble layer
The final prediction combines branch outputs through a learnable fusion mechanism. This helps keep both branches on the same scale and reduces the chance of one branch overwhelming the other.

## Features
- Hybrid TensorFlow model with CNN + BiLSTM + attention
- Explainability panel with branch contributions and confidence outputs
- Structured observability logs for training and inference
- Synthetic adverse-media dataset generation
- Optional original + synthetic dataset mixing
- Stratified preprocessing and training pipeline
- Streamlit UI for controlled experimentation
- Model balance monitoring for CNN/LSTM dominance
- Reproducible training with configurable random seed

## Repository Structure
```text
kyc-risk-intelligence-tensorflow-system/
|-- app/
|   |-- models/
|   |   |-- hybrid_model.py
|   |-- observability/
|   |   |-- logging.py
|   |   |-- metrics.py
|   |-- pipeline/
|       |-- inference.py
|       |-- preprocessing.py
|       |-- training.py
|-- data/
|   |-- FinancialPhraseBank-v1.0/
|   |-- synthetic_adverse_media.csv
|-- scripts/
|   |-- generate_synthetic_dataset.py
|   |-- run_training_comparison.py
|-- .github/workflows/ci.yml
|-- FAQ.md
|-- README.md
|-- requirements.txt
|-- streamlit_app.py
```

## Setup Instructions
### Prerequisites
- Python 3.10+
- pip

### Installation
```bash
pip install -r requirements.txt
```

### Run Application
```bash
streamlit run streamlit_app.py
```

## Data Sources
### Original dataset
The system supports multiple Financial PhraseBank agreement sets located under `data/FinancialPhraseBank-v1.0/`.

### Synthetic dataset
`data/synthetic_adverse_media.csv` contains multi-sentence contextual samples designed to improve sequence learning, label balance, and long-form adverse-media coverage.

### Dataset mixing
The training pipeline can mix original and synthetic data using a configurable ratio. Stratification is applied on the final combined label distribution to preserve class balance through training, validation, and test splits.

## Training and Evaluation
The training pipeline includes:
- Early stopping
- ReduceLROnPlateau
- Class weighting for imbalance control
- Structured per-epoch logging
- Branch balance diagnostics
- Persisted configuration artifacts

The model can log:
- Loss, accuracy, precision, recall, and F1
- Validation metrics
- LSTM score mean/std
- CNN score mean/std
- Attention statistics
- Learnable ensemble alpha
- LSTM and CNN contribution shares
- Imbalance warnings when one branch dominates

## Explainability and Observability
The Streamlit interface exposes:
- Tokenized text and token IDs
- LSTM score
- CNN score
- Ensemble score
- LSTM vs CNN contribution percentage
- Attention statistics
- Overfitting monitor
- Branch balance visualization

The system writes structured artifacts into the `models/` directory during training and inference. Runtime logs are intentionally excluded from version control.

## CI/CD
GitHub Actions is configured in `.github/workflows/ci.yml` to run on every push and pull request.

The pipeline:
- Checks out the repository
- Sets up Python 3.11
- Installs dependencies
- Runs syntax validation with `py_compile`
- Runs a preprocessing smoke test on the synthetic dataset

## Challenges Faced
### CNN branch dominating early predictions
Initial runs showed the CNN branch contributing nearly all of the final score because short adverse keywords were easier to learn than broader sequence context.

### LSTM under-utilization
The original LSTM path produced very small scores relative to the CNN branch, which made the hybrid model behave like a keyword detector rather than a balanced sequence model.

### Normalization before branch fusion
Branch outputs needed normalization before combination so the final ensemble was not biased by raw magnitude differences alone.

### Static vs learnable ensemble weights
A static fusion ratio was too rigid. Replacing it with a learnable alpha allowed the model to adapt branch weighting during training.

### Limited sequence-rich examples in the original dataset
Financial PhraseBank contains many short statements, which is useful for sentiment-style classification but weaker for long-form KYC adverse-media progression.

### Synthetic multi-sentence dataset creation
A larger contextual dataset was introduced to add progression narratives, weak-to-strong signal transitions, and more realistic financial media structure.

### Incorrect synthetic label mapping
Synthetic labels initially required correction so numeric labels remained intact during loading and did not collapse class balance.

### Stratification under class imbalance
The split logic needed validation and safe fallback handling so train, validation, and test sets preserve class balance even under skewed distributions.

### Threshold calibration for production behavior
A strong branch score does not automatically mean the final decision threshold is well calibrated for every long-form case, so threshold selection still matters for operational deployment.

### Streamlit duplicate element IDs
Repeated Plotly charts needed explicit stable keys to prevent duplicate element ID errors during reruns.

### Explainability consistency debugging
Branch contribution logic and chart rendering had to stay aligned with the current fusion method so UI explanations matched actual model behavior.

## FAQ
See [FAQ.md](FAQ.md) for model, data, training, and engineering questions.

## Usage Notes
- Train the model before running inference if no saved model is present.
- Generate or refresh the synthetic dataset with `python scripts/generate_synthetic_dataset.py` when needed.
- Use the comparison script to produce a before/after artifact set after architecture changes.