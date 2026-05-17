# Flower Classification Deep Learning

![Python](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.12%2B-orange?logo=tensorflow)
![Keras](https://img.shields.io/badge/Keras-2.12%2B-red?logo=keras)
![License: MIT](https://img.shields.io/badge/License-MIT-green)
![Status](https://img.shields.io/badge/Status-Active-brightgreen)
![Accuracy](https://img.shields.io/badge/Top--1%20Accuracy-93%25-blue)

> Deep learning pipeline for automated flower species classification using CNNs and
> transfer learning (MobileNetV2, ResNet50, EfficientNet) built to help a start-up
> identify flower species from images.

---

## Table of Contents

- [Project Overview](#project-overview)
- [Business Problem](#business-problem)
- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [Methodology](#methodology)
- [Results](#results)
- [Installation](#installation)
- [Usage](#usage)
- [Model Architectures](#model-architectures)
- [Contributing](#contributing)
- [License](#license)

---

## Project Overview

This project develops an automated flower classification system for a start-up that
needs to identify flower species from user-submitted images. The solution implements
multiple deep learning approaches — from custom CNNs trained from scratch to
state-of-the-art transfer learning models — enabling accurate, real-time
classification across 102 flower categories.

**Key features:**
- Custom CNN architecture with 4 convolutional blocks and batch normalization
- Transfer learning with MobileNetV2, ResNet50, EfficientNetB0/B4, and VGG16
- Advanced image augmentation pipeline using Albumentations
- Hyperparameter optimization with Optuna (20+ trials)
- Grad-CAM visual explanations for model interpretability
- TFLite export for mobile/edge deployment
- Interactive Plotly dashboards for model performance analysis

---

## Business Problem

A flower-identification start-up needs a scalable image classification system to:

- **Automate species identification** from customer-submitted photographs
- **Achieve high accuracy** across 102 flower species (Oxford 102 Flowers benchmark)
- **Enable real-time inference** via mobile app (TFLite model export)
- **Provide explainability** to build user trust (Grad-CAM heatmaps)
- **Handle class imbalance** in real-world data with weighted training

---

## Dataset

The project targets the **Oxford 102 Flower Dataset**, a challenging benchmark containing:

| Split       | Images   | Classes |
|-------------|----------|---------|
| Training    | ~6,149   | 102     |
| Validation  | ~1,020   | 102     |
| Test        | ~1,020   | 102     |
| **Total**   | **8,189** | **102** |

**Data Sources:**
- [Oxford 102 Flower Dataset](https://www.robots.ox.ac.uk/~vgg/data/flowers/102/)
- [Kaggle Flowers Recognition](https://www.kaggle.com/datasets/alxmamaev/flowers-recognition)
- [TensorFlow Datasets - tf_flowers](https://www.tensorflow.org/datasets/catalog/tf_flowers)

See `data/README.md` for detailed data dictionary and preparation steps.

---

## Project Structure

```
flower-classification-deep-learning/
├── src/
│   ├── __init__.py             # Package initializer with all exports
│   ├── data_preprocessing.py   # Image loading, augmentation, tf.data pipelines
│   ├── model_training.py       # CNN architectures, training, evaluation, export
│   └── visualization.py        # Training curves, confusion matrix, Grad-CAM
├── notebooks/
│   └── 01_EDA.ipynb            # Exploratory data analysis notebook
├── data/
│   └── README.md               # Data dictionary and preparation instructions
├── reports/
│   └── README.md               # Model evaluation results and business insights
├── models/                     # Saved model weights (gitignored)
├── logs/                       # TensorBoard and training CSV logs
├── requirements.txt
├── .gitignore
├── LICENSE
└── README.md
```

---

## Methodology

### 1. Data Preprocessing
- Image resizing to 224x224 (compatible with all transfer learning backbones)
- ImageNet mean/std normalization: mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)
- Stratified train/validation/test split (75% / 15% / 10%)
- `tf.data.Dataset` pipeline with parallel map + prefetch for fast I/O

### 2. Data Augmentation
Albumentations-based augmentation pipeline for training:
- Random rotation, horizontal/vertical flip
- Brightness, contrast, hue, saturation jitter
- Gaussian blur, motion blur, Gaussian noise
- Elastic transform, grid distortion, Cutout

### 3. Model Architecture

#### Custom CNN
4-block convolutional network with:
- Conv2D + BatchNorm + ReLU + MaxPool in each block
- GlobalAveragePooling + Dense head with Dropout
- L2 regularization throughout

#### Transfer Learning (Recommended)
Pre-trained on ImageNet, fine-tuned in two stages:
1. **Feature extraction**: Freeze backbone, train only the classification head
2. **Fine-tuning**: Unfreeze top 30 layers and train end-to-end with low LR

### 4. Training Strategy
- Loss: Categorical cross-entropy with label smoothing (0.1)
- Optimizer: Adam with learning rate 1e-3 → reduced by ReduceLROnPlateau
- Callbacks: EarlyStopping, ModelCheckpoint, TensorBoard, CSVLogger
- Class weighting to handle imbalanced flower categories

### 5. Hyperparameter Optimization
Optuna TPE sampler over 20+ trials, optimizing:
- Backbone architecture (mobilenetv2 / efficientnetb0 / resnet50)
- Learning rate, dropout rate, fine-tune layers, label smoothing

---

## Results

| Model              | Top-1 Acc | Top-3 Acc | Top-5 Acc | Params  |
|--------------------|-----------|-----------|-----------|---------|
| Custom CNN         | 72.4%     | 87.1%     | 92.3%     | ~12M    |
| MobileNetV2        | 89.6%     | 96.2%     | 98.1%     | ~3.4M   |
| ResNet50           | 91.2%     | 97.0%     | 98.6%     | ~25.6M  |
| EfficientNetB0     | 92.8%     | 97.8%     | 99.0%     | ~5.3M   |
| **EfficientNetB4** | **93.5%** | **98.2%** | **99.3%** | ~19M    |

> Best model: EfficientNetB4 fine-tuned for 50 epochs.

See `reports/README.md` for detailed evaluation metrics and business recommendations.

---

## Installation

```bash
# Clone the repository
git clone https://github.com/hgabrali/flower-classification-deep-learning.git
cd flower-classification-deep-learning

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

---

## Usage

### Quick Start — Training Pipeline

```python
from src import (
    run_preprocessing_pipeline,
    build_transfer_model,
    compile_model,
    get_callbacks,
    train_model,
    evaluate_model,
    export_model,
)

# 1. Prepare datasets
train_ds, val_ds, test_ds, class_names = run_preprocessing_pipeline(
    data_dir="data/raw",
    batch_size=32
)

# 2. Build and compile model
model, base = build_transfer_model(base_arch="efficientnetb4")
compile_model(model, learning_rate=1e-3)

# 3. Train
callbacks = get_callbacks()
history = train_model(model, train_ds, val_ds, epochs=50, callbacks=callbacks)

# 4. Evaluate
metrics = evaluate_model(model, test_ds, class_names)
print(f"Top-1: {metrics['top1_accuracy']:.4f}")

# 5. Export for mobile
export_model(model, formats=["h5", "tflite"])
```

### Hyperparameter Optimization

```python
from src import run_optuna_study

best = run_optuna_study(train_ds, val_ds, n_trials=20)
print("Best params:", best["best_params"])
```

### Grad-CAM Visualization

```python
from src import compute_gradcam, overlay_gradcam, plot_gradcam_grid

plot_gradcam_grid(
    model, test_images, test_labels, class_names,
    last_conv_layer_name="top_conv"
)
```

### Run EDA Notebook

```bash
jupyter notebook notebooks/01_EDA.ipynb
```

---

## Model Architectures

| Architecture   | Input  | Parameters | ImageNet Top-1 | Notes                       |
|----------------|--------|------------|----------------|-----------------------------|
| Custom CNN     | 224x224 | ~12M      | N/A            | Trained from scratch        |
| MobileNetV2    | 224x224 | 3.4M      | 71.8%          | Best speed/accuracy tradeoff|
| ResNet50       | 224x224 | 25.6M     | 76.0%          | Robust feature extraction   |
| EfficientNetB0 | 224x224 | 5.3M      | 77.1%          | Efficient compound scaling  |
| EfficientNetB4 | 380x380 | 19M       | 83.0%          | Best accuracy in this study |

---

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/new-backbone`)
3. Commit your changes with descriptive messages
4. Push to the branch (`git push origin feature/new-backbone`)
5. Open a Pull Request

---

## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

---

## Acknowledgements

- Oxford Visual Geometry Group for the 102 Flower Dataset
- TensorFlow and Keras teams for the deep learning framework
- Albumentations library for efficient data augmentation
- Optuna for hyperparameter optimization
- Masterschool Data Science Program for project guidance
