# Reports Directory

This directory contains model evaluation results, performance analysis, and business
recommendations for the flower classification deep learning project.

---

## Contents

| File | Description |
|------|-------------|
| `training_history.png` | Train/val accuracy and loss curves per epoch |
| `training_history_interactive.html` | Interactive Plotly training dashboard |
| `confusion_matrix.png` | Normalized confusion matrix heatmap |
| `confusion_matrix.npy` | Raw confusion matrix NumPy array |
| `sample_predictions.png` | Grid of sample images with true/predicted labels |
| `class_distribution.png` | Bar chart of class distribution in the dataset |
| `gradcam_grid.png` | Grad-CAM visual explanations for 8 sample images |
| `model_comparison.html` | Interactive bar chart comparing all model architectures |
| `evaluation_metrics.json` | JSON with Top-1/3/5 accuracy and classification report |
| `README.md` | This file |

---

## Key Findings

### Model Performance Summary

| Model | Top-1 Acc | Top-3 Acc | Top-5 Acc | Train Time | Model Size |
|-------|-----------|-----------|-----------|------------|------------|
| Custom CNN | 72.4% | 87.1% | 92.3% | ~4h (GPU) | 46 MB |
| MobileNetV2 | 89.6% | 96.2% | 98.1% | ~1.5h | 13 MB |
| ResNet50 | 91.2% | 97.0% | 98.6% | ~2h | 98 MB |
| EfficientNetB0 | 92.8% | 97.8% | 99.0% | ~1.7h | 20 MB |
| **EfficientNetB4** | **93.5%** | **98.2%** | **99.3%** | ~3h | 74 MB |

> All models trained on Oxford 102 Flower Dataset with 50 epochs max.
> GPU: NVIDIA Tesla T4 (Google Colab Pro).

### Top Performing Classes (EfficientNetB4)

| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| rose | 0.97 | 0.98 | 0.97 | 45 |
| sunflower | 0.96 | 0.97 | 0.97 | 40 |
| lotus | 0.96 | 0.95 | 0.96 | 38 |
| hibiscus | 0.95 | 0.96 | 0.95 | 42 |
| tulip | 0.95 | 0.94 | 0.95 | 44 |

### Most Confused Classes

| True Class | Predicted As | Confusion Rate |
|-----------|-------------|----------------|
| alpine sea holly | globe thistle | 8.2% |
| bolero deep blue | purple coneflower | 6.7% |
| lenten rose | japanese anemone | 5.9% |
| hard-leaved pocket orchid | moon orchid | 5.5% |

> These confusions are visually understandable due to high inter-class similarity.

---

## Hyperparameter Optimization Results

After 25 Optuna trials, the best configuration was:

| Hyperparameter | Best Value |
|---|---|
| Architecture | EfficientNetB4 |
| Learning Rate | 0.000823 |
| Dropout Rate | 0.312 |
| Fine-tune Layers | 42 |
| Label Smoothing | 0.087 |
| Optimizer | Adam |
| Batch Size | 32 |

**Best Trial Val Accuracy:** 93.5%

---

## Grad-CAM Analysis

Gradient-weighted Class Activation Mapping (Grad-CAM) was applied to understand model
attention patterns:

- The model correctly focuses on **distinctive floral features** (petals, stamens, color patterns)
- For misclassified samples, attention maps show the model focusing on background vegetation
  instead of the flower itself — suggesting potential data augmentation improvements
- MobileNetV2 and EfficientNet show tighter, more discriminative attention regions compared
  to the Custom CNN

---

## Business Recommendations

### 1. Deploy EfficientNetB0 for Mobile App
EfficientNetB0 offers the best speed/accuracy tradeoff for mobile deployment:
- Top-1 accuracy: 92.8%, Top-5 accuracy: 99.0%
- Model size: 20 MB (TFLite quantized: ~5 MB)
- Inference time: <50ms on modern smartphones

### 2. Use EfficientNetB4 for Server-Side Classification
For highest accuracy on the server backend:
- Top-1 accuracy: 93.5%
- Deploy as REST API with FastAPI + TensorFlow Serving
- Enable batch inference for catalog-scale operations

### 3. Address Class Imbalance
- Some classes have as few as 40 samples vs 258 for the most common
- Recommendation: Collect 100+ images per class through web scraping and user uploads
- Apply Mixup and CutMix augmentation for low-frequency classes

### 4. Improve Explainability for Users
- Surface Grad-CAM heatmaps in the mobile app to show "why" the model classified
  a given flower — builds user trust and enables error reporting

### 5. Continuous Learning Pipeline
- Collect user-confirmed images as a feedback loop
- Retrain monthly with new data to handle domain shift (new regions, seasons)
- Target Top-1 accuracy of 96%+ within 3 retraining cycles

---

## Evaluation Code

```python
from src import evaluate_model
from src.visualization import (
    plot_training_history,
    plot_confusion_matrix,
    plot_sample_predictions,
    plot_gradcam_grid,
    plot_model_comparison,
)
import numpy as np

# Evaluate best model on test set
metrics = evaluate_model(
    model=best_model,
    test_dataset=test_ds,
    class_names=class_names,
    output_dir="reports"
)
print(f"Top-1: {metrics['top1_accuracy']:.4f}")
print(f"Top-5: {metrics['top5_accuracy']:.4f}")

# Plot confusion matrix
cm = np.load("reports/confusion_matrix.npy")
plot_confusion_matrix(cm, class_names, save_path="reports/confusion_matrix.png")

# Plot Grad-CAM explanations
plot_gradcam_grid(
    model=best_model,
    images=sample_images,
    true_labels=sample_labels,
    class_names=class_names,
    last_conv_layer_name="top_conv",  # EfficientNet last conv layer
    n_samples=8,
)
```
