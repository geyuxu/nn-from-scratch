# 02 - Logistic Regression

[中文版](README_CN.md)

## Objective

Implement binary classification: determine if `x > 0`
- Input: random number x
- Output: 0 or 1

## Core Principles

### 1. Difference from Linear Regression

| | Linear Regression | Logistic Regression |
|---|---|---|
| Task | Regression (predict continuous value) | Classification (predict category) |
| Output | Any real number | Probability in (0, 1) |
| Activation | None | Sigmoid |
| Loss Function | MSE | Cross-Entropy |

### 2. Model Structure

```
t = X @ W + b              # linear transformation
y_pred = sigmoid(t)        # map to (0, 1) probability
```

### 3. Sigmoid Function

Maps any real number to (0, 1) interval, representing probability:

```python
sigmoid(t) = 1 / (1 + exp(-t))
```

Properties:
- t → +∞: sigmoid → 1
- t → -∞: sigmoid → 0
- t = 0: sigmoid = 0.5

### 4. Loss Function - Binary Cross-Entropy (BCE)

```python
L = -mean(y_true * log(y_pred) + (1 - y_true) * log(1 - y_pred))
```

Intuition:
- When `y_true = 1`: want `y_pred` close to 1, larger `log(y_pred)` means smaller loss
- When `y_true = 0`: want `y_pred` close to 0, larger `log(1 - y_pred)` means smaller loss

### 5. Classification Decision

```python
predicted_class = 1 if y_pred > 0.5 else 0
```

## Implementations

| File | Method | Features |
|------|--------|----------|
| `logistic_regression_torch.py` | Synthetic data | Simple x > 0 classification |
| `logistic_regression_breast_cancer.py` | Real dataset | 30 features, train/test split, full metrics |

## Training Loop

```python
for epoch in range(epoch_count):
    # 1. Forward pass
    t = X @ W + b
    y_pred = 1 / (1 + torch.exp(-t))  # sigmoid

    # 2. Compute loss (cross-entropy)
    loss = -(y_true * torch.log(y_pred) + (1 - y_true) * torch.log(1 - y_pred)).mean()

    # 3. Backward pass
    loss.backward()

    # 4. Update parameters
    W -= lr * W.grad
    b -= lr * b.grad
```

## Training Results - Synthetic Data

The model learns to classify positive/negative numbers:

![training result](Figure_logistic_regression.png)

- **Left**: Loss curve, decreasing during training
- **Right**: Classification result, blue=class 0, red=class 1, green curve=decision boundary (sigmoid)

## From Linear to Logistic Regression

Only two changes needed:

1. **Add Sigmoid**: `y_pred = sigmoid(X @ W + b)`
2. **Change Loss**: MSE → Cross-Entropy

This is the fundamental pattern of neural networks: linear transformation + nonlinear activation + appropriate loss function

---

## Real Dataset - Breast Cancer Wisconsin

Apply logistic regression to a real medical dataset for tumor classification.

### Dataset Info

- **Samples**: 569
- **Features**: 30 (radius, texture, perimeter, area, smoothness, etc.)
- **Target**: Malignant (0) or Benign (1)
- **Class distribution**: ~63% Benign, ~37% Malignant

### Key Differences from Synthetic Data

| Aspect | Synthetic | Breast Cancer |
|--------|-----------|---------------|
| Features | 1 | 30 |
| Data split | None | 80% train / 20% test |
| Preprocessing | None | StandardScaler |
| Evaluation | Accuracy only | Accuracy, Precision, Recall, F1 |

### Classification Metrics

| Metric | Description |
|--------|-------------|
| **Accuracy** | Overall correct predictions / total |
| **Precision** | TP / (TP + FP) - "Of predicted positives, how many are correct?" |
| **Recall** | TP / (TP + FN) - "Of actual positives, how many were found?" |
| **F1** | Harmonic mean of Precision and Recall |

### Training Results

![Breast Cancer Result](Figure_logistic_regression_breast_cancer.png)

- **Left**: Training loss curve (Binary Cross-Entropy)
- **Middle**: Confusion matrix showing TP, TN, FP, FN
- **Right**: PCA projection of test set (black × = misclassified)

### Result Analysis

**Performance**

| Metric | Train | Test |
|--------|-------|------|
| Accuracy | 0.989 | 0.974 |
| Precision | 0.993 | 0.986 |
| Recall | 0.989 | 0.972 |
| F1 | 0.991 | 0.979 |

**Key Insights**

- Logistic regression achieves **97%+ accuracy** on this dataset
- Very few misclassifications (only 3 out of 114 test samples)
- Linear decision boundary works well because features are well-separated after standardization

### Conclusion

Logistic regression performs excellently on Breast Cancer dataset (**~97% accuracy**), showing that:

1. **Linear models can be powerful** when features are informative
2. **Feature scaling is crucial** - StandardScaler helps a lot
3. **Medical diagnosis** requires high recall (don't miss cancer cases)
