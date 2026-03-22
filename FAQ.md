# FAQ

## Model & ML Concepts

### What is LSTM and why is it used?
LSTM is a recurrent neural network layer designed to learn order and context in sequences. In this project it helps the model understand how risk signals evolve across a sentence or across multiple sentences.

### What is CNN in NLP?
A CNN in NLP scans text embeddings for short local patterns. It is useful for detecting compact phrases such as adverse keywords, suspicious collocations, and small text fragments that often appear in financial media.

### Why combine LSTM and CNN?
They solve different parts of the problem. CNN is strong at local phrase detection, while LSTM is better at context and progression. Combining them improves coverage across both short and long narratives.

### What is the attention mechanism?
Attention helps the model focus on the most relevant parts of a sequence instead of treating every token equally. It is especially useful when a long article contains only a few critical sentences.

### Why was CNN dominating initially?
The original setup made it easier for the CNN branch to learn fast from short adverse keywords, while the LSTM branch produced much smaller outputs. That imbalance caused the final prediction to behave too much like a keyword classifier.

### What does model contribution mean?
Contribution is the relative share each branch has in the final prediction logic. It is a way to inspect whether the LSTM and CNN are both participating or whether one branch is effectively carrying the whole decision.

### Should LSTM and CNN be 50-50?
Not necessarily. Equal contribution is not a goal by itself. The goal is for both branches to be meaningfully useful and for one branch not to dominate only because of scale or architectural bias.

### What is overfitting and how is it detected?
Overfitting happens when the model learns the training data too specifically and stops generalizing well. It is monitored here by comparing training and validation metrics, especially accuracy, loss, recall, and F1.

### What is dropout and what impact does it have?
Dropout randomly disables part of the network during training. It reduces overfitting by forcing the model to rely on more distributed patterns instead of memorizing narrow shortcuts.

## Data & Training

### Why is stratified split important?
Stratified split preserves class balance across train, validation, and test sets. Without it, a skewed split can make evaluation metrics unreliable, especially recall and F1.

### What happens if the dataset has only one class?
A stratified split cannot be applied safely if only one class is present or if a class has too few examples. In that case the pipeline falls back to a non-stratified split and logs a warning.

### Why was synthetic data created?
The original dataset contains many short statements. Synthetic data was added to introduce longer contextual narratives, stronger sequence patterns, and more balanced supervision for material-risk scenarios.

### How does synthetic data improve the model?
It adds multi-sentence progression, weak-to-strong signal transitions, and more realistic adverse-media wording. This helps the LSTM and attention layers learn sequence behavior more effectively.

### What is class imbalance and how do you fix it?
Class imbalance happens when one class appears much more often than the other. Common fixes include stratified splitting, class weighting, targeted synthetic data, threshold tuning, and better sampling strategy.

### Why verify label mapping so carefully?
If labels are mapped incorrectly during loading, the model can train on the wrong target distribution. That can distort stratification, class balance, and final evaluation results.

## Model Behavior

### Why can a long text still give a low score?
A long text may include mixed signals, hedged language, or insufficient confirmed evidence. The model may also require threshold tuning if the branch scores are informative but the final decision boundary is too conservative.

### What is threshold tuning?
Threshold tuning is the process of adjusting the score cut-off used to convert model probabilities into class labels. A default threshold of `0.5` is common, but operational goals may justify a different threshold.

### Why is explainability needed?
KYC and adverse-media workflows often require users to understand why a case was flagged. Explainability helps reviewers see which branch contributed more and whether the output is driven by pattern signals, sequence context, or both.

### Can a high LSTM score still result in a low final score?
Yes. The ensemble combines both branches and may apply a learned weighting. If one branch is strong and the other is weak, the final score depends on the fusion logic and the current threshold.

### Why monitor branch balance over time?
A hybrid model can drift toward one branch if the data or architecture creates shortcuts. Monitoring balance helps detect when explainability and intended architecture behavior no longer match actual model use.

## System & Engineering

### What caused the Streamlit duplicate element error?
The same Plotly chart type was rendered multiple times without explicit unique keys. Streamlit generated identical internal IDs and raised a duplicate element error during reruns.

### How was the Streamlit issue fixed?
Chart rendering was centralized through a helper that assigns explicit keys. Session-state-backed suffixes are used for training and inference reruns so repeated charts remain unique.

### Why normalize branch outputs before combining models?
Normalization helps keep both branches on a comparable scale. Without it, the larger-magnitude branch can dominate the ensemble even if both branches contain useful information.

### What is learnable alpha in the ensemble?
Learnable alpha is a trainable parameter that controls how much the final prediction relies on the LSTM branch versus the CNN branch. It allows the model to adapt the fusion weight during training instead of using a fixed rule.

### Why use structured logs?
Structured logs make it easier to trace training quality, branch behavior, and inference results over time. They are useful for debugging, monitoring, and reproducibility.

### Why are runtime artifacts excluded from version control?
Model checkpoints and logs change frequently and can become large quickly. Excluding them keeps the repository smaller and makes source control easier to maintain.

### Why use a preprocessing smoke test in CI?
It gives a quick signal that the dataset loader, imports, and basic schema assumptions still work after a code change. This helps catch regressions early without running a full training job.

### Why not rely only on accuracy?
Accuracy can look acceptable even when the model misses most positive cases in an imbalanced dataset. Recall, precision, and F1 are more informative for adverse-media risk classification.

### How does dataset mixing avoid skewing the split?
The system combines original and synthetic rows first, then stratifies on the final combined labels. That ensures the split preserves the actual class balance used for training.

### When should synthetic data be refreshed?
It should be refreshed when the scenario library changes, when label definitions evolve, or when the model needs broader sequence coverage than the current synthetic set provides.