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
## TensorFlow & Deep Learning Concepts

### What is TensorFlow and why is it used in this project?
TensorFlow is a deep learning framework used to build, train, save, and load the models in this repository. It is used here because the project needs neural network layers, training utilities, model serialization, and dataset pipelines in one consistent stack.

### How is TensorFlow different from plain Python?
Plain Python is a general programming language. TensorFlow adds optimized operations for matrix math, gradient-based training, neural network layers, and model execution on larger workloads. Python provides the control flow, while TensorFlow provides the deep learning engine.

### What is Keras and how does it relate to TensorFlow?
Keras is the high-level API used inside TensorFlow for defining models and layers. In this project, layers such as `Embedding`, `LSTM`, `Conv1D`, `Attention`, `Dense`, and callbacks such as `EarlyStopping` are all used through `tf.keras`.

### What is a computational graph in simple terms?
A computational graph is a structured way to represent how data moves through operations. In practice, it means TensorFlow can track how inputs pass through layers, compute predictions, measure loss, and work backward to update weights.

### What is an epoch?
An epoch is one full pass through the training dataset. If the training set has been fully seen once by the model, one epoch is complete.

### What is batch size?
Batch size is how many samples the model processes before updating its weights once. If the batch size is `16`, the model trains on 16 examples at a time.

### How do epochs and batch size impact training?
More epochs usually mean more learning opportunities, but they also increase training time and can increase overfitting if pushed too far. Batch size changes how often the model updates weights and how noisy those updates are.

### Does increasing epochs make training faster or slower?
Slower. More epochs mean the model goes through the dataset more times, so total training time increases.

### What is a loss function and why is BinaryCrossentropy used?
A loss function measures how wrong the model is during training. `BinaryCrossentropy` is used here because the task is binary classification: Material Risk vs Non-Material Risk.

### What is an optimizer such as Adam?
An optimizer controls how the model updates its weights after measuring loss. Adam is commonly used because it usually converges faster and more smoothly than simpler optimizers on many NLP tasks.

### What are metrics like accuracy, precision, recall, and F1?
Accuracy shows how often the prediction is correct overall. Precision shows how often predicted positives are actually positive. Recall shows how many real positives the model finds. F1 balances precision and recall in one score.

### Why is F1 important in this repository?
This system deals with class imbalance and adverse-risk detection, so accuracy alone can be misleading. F1 is useful because it reflects whether the model is actually finding material-risk cases without generating too many false positives.

### What is an LSTM layer and how does it work?
An LSTM is a recurrent layer designed for ordered data such as text. It processes tokens step by step and uses internal gates to decide what information to keep, update, or forget.

### What is a bidirectional LSTM and why use it here?
A bidirectional LSTM reads the sequence in both forward and backward directions. That helps the model understand a token using both earlier and later context in the sentence.

### What is a CNN layer in the NLP context?
A CNN layer in NLP scans across token embeddings to detect short local patterns. It is effective for phrases, fragments, and concentrated lexical signals such as sanctions, bribery, fraud, or money-laundering terms.

### Why combine CNN and LSTM in TensorFlow instead of using only one branch?
Using both branches gives the model two different views of the same text. CNN captures local phrase patterns, while LSTM captures progression and context across the sequence.

### What is dropout and how does it affect training?
Dropout randomly turns off part of the network during training. This makes the model less dependent on any single narrow pattern and helps reduce overfitting.

### What is the attention layer and why was it added to the LSTM branch?
The attention layer helps the model focus on the most relevant sequence regions instead of compressing everything into one final recurrent state. It was added to improve long-text understanding and strengthen the LSTM branch.

### Why was LayerNormalization added before combining branch outputs?
Normalization helps keep the LSTM and CNN feature representations on a more comparable scale. Without that, one branch can dominate the fusion mostly because of output magnitude rather than better reasoning.

### Why was CNN dominating initially?
The original setup made it easier for the CNN branch to react strongly to short adverse keywords, while the LSTM branch had much weaker output values. That caused the ensemble to lean too heavily on pattern matching.

### What is learnable alpha in the ensemble?
Learnable alpha is a trainable fusion weight that controls how much the final output depends on the LSTM branch versus the CNN branch. In this repository it is constrained with a sigmoid so it stays between 0 and 1.

### Why do model outputs need calibration?
A model can rank samples reasonably well but still produce scores that are too conservative or too aggressive for the chosen threshold. Calibration helps align raw output scores with practical decision boundaries.

### Why can long sequences fail to increase LSTM contribution by themselves?
Longer text does not automatically mean better sequence learning. If the extra tokens are noisy, repetitive, or weakly labeled, the LSTM may still contribute less unless the data and architecture support meaningful sequence patterns.

### What is tf.data.Dataset?
`tf.data.Dataset` is TensorFlow's input pipeline abstraction for feeding data into a model efficiently. It helps manage batching, shuffling, and prefetching in a clean training pipeline.

### Why are batching and shuffling important?
Batching keeps training efficient by processing multiple samples together. Shuffling reduces unwanted ordering effects so the model does not learn from accidental sample order patterns.

### What does prefetching do?
Prefetching prepares future batches while the model is still training on the current one. This helps reduce idle time in the input pipeline and keeps training more efficient.

### How does tokenization work in TensorFlow in this project?
The project uses a Keras `Tokenizer` to convert text into integer token IDs based on the training vocabulary. Those sequences are then padded to a consistent length so they can be fed into TensorFlow models.

### Why is padding required?
Neural network batches need consistent tensor shapes. Padding makes shorter sequences the same length as longer ones so they can be processed together.

### Why is truncation used together with padding?
Very long sequences can make training slower and harder to manage. Truncation limits sequence length so memory use and runtime stay controlled.

### How does batch size impact model accuracy?
There is no single rule. Smaller batches can make updates noisier but sometimes help generalization, while larger batches can be more stable and faster per epoch but may need different tuning.

### Why specific layers like `Conv1D`, `GlobalMaxPooling1D`, and `Dense` were chosen?
`Conv1D` captures local patterns, `GlobalMaxPooling1D` keeps the strongest activation from those patterns, and `Dense` layers turn learned features into branch-level risk scores. The combination is compact and effective for text classification.

### Why is `GlobalAveragePooling1D` used after attention?
After attention produces a contextualized sequence, global average pooling summarizes that sequence into a fixed-size vector. This gives a stable feature representation before the final LSTM branch scoring layer.

### Why does the training pipeline use callbacks?
Callbacks help automate training control and monitoring. In this project they are used for early stopping, learning-rate reduction, checkpoint saving, and structured epoch logging.

### How is the training loop structured in this repository?
The system builds the model, compiles it with loss and metrics, prepares datasets through the preprocessing pipeline, and then trains with `model.fit(...)` using callbacks and class weights.

### Why are class weights used during training?
Class weights increase the importance of the minority class during optimization. This helps the model pay more attention to material-risk examples when the dataset is imbalanced.

### Why does TensorFlow need model compilation before training?
Compilation tells TensorFlow which optimizer, loss function, and metrics to use. Without `model.compile(...)`, the framework would not know how to train or evaluate the model.

### What does `return_sequences=True` do in the LSTM branch?
It makes the LSTM return an output for every time step instead of only one final output. That is necessary here because the attention layer needs access to the full sequence representation.

### How should explainability weights such as LSTM vs CNN contributions be interpreted?
They should be read as branch influence indicators, not as direct proof of correctness. A higher contribution means a branch had more impact on the final prediction, but the prediction still needs to be judged in the context of score quality and threshold choice.
