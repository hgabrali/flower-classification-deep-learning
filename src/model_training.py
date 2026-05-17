"""
model_training.py

Model training module for flower classification deep learning project.
Implements custom CNN, transfer learning (MobileNetV2, ResNet50, EfficientNet),
Optuna hyperparameter optimization, and model evaluation utilities.
"""

import os
import json
import logging
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Callable

import numpy as np
import pandas as pd
import optuna
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers
from tensorflow.keras.applications import (
    MobileNetV2, ResNet50, EfficientNetB0, EfficientNetB4, VGG16
)
from tensorflow.keras.callbacks import (
    ModelCheckpoint, EarlyStopping, ReduceLROnPlateau,
    TensorBoard, CSVLogger
)
from tensorflow.keras.optimizers import Adam, SGD, RMSprop
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score,
    top_k_accuracy_score
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

NUM_CLASSES = 102  # Oxford 102 Flower Dataset
IMAGE_SIZE = (224, 224)


# ─────────────────────────────────────────────────────────────
# 1. Custom CNN Architecture
# ─────────────────────────────────────────────────────────────

def build_custom_cnn(
    input_shape: Tuple[int, int, int] = (224, 224, 3),
    num_classes: int = NUM_CLASSES,
    dropout_rate: float = 0.4,
    l2_lambda: float = 1e-4
) -> keras.Model:
    """Build a custom CNN from scratch for flower classification.

    Architecture: 4 Conv blocks with BatchNorm + MaxPool, then dense head.

    Args:
        input_shape: Input image shape (H, W, C).
        num_classes: Number of output flower classes.
        dropout_rate: Dropout rate for regularization.
        l2_lambda: L2 regularization strength.

    Returns:
        keras.Model: Compiled custom CNN model.
    """
    reg = regularizers.l2(l2_lambda)
    inputs = keras.Input(shape=input_shape, name="input")

    # Block 1
    x = layers.Conv2D(32, 3, padding="same", kernel_regularizer=reg, name="conv1_1")(inputs)
    x = layers.BatchNormalization(name="bn1_1")(x)
    x = layers.Activation("relu")(x)
    x = layers.Conv2D(32, 3, padding="same", kernel_regularizer=reg, name="conv1_2")(x)
    x = layers.BatchNormalization(name="bn1_2")(x)
    x = layers.Activation("relu")(x)
    x = layers.MaxPooling2D(2, strides=2)(x)
    x = layers.Dropout(dropout_rate * 0.5)(x)

    # Block 2
    x = layers.Conv2D(64, 3, padding="same", kernel_regularizer=reg, name="conv2_1")(x)
    x = layers.BatchNormalization(name="bn2_1")(x)
    x = layers.Activation("relu")(x)
    x = layers.Conv2D(64, 3, padding="same", kernel_regularizer=reg, name="conv2_2")(x)
    x = layers.BatchNormalization(name="bn2_2")(x)
    x = layers.Activation("relu")(x)
    x = layers.MaxPooling2D(2, strides=2)(x)
    x = layers.Dropout(dropout_rate * 0.5)(x)

    # Block 3
    x = layers.Conv2D(128, 3, padding="same", kernel_regularizer=reg, name="conv3_1")(x)
    x = layers.BatchNormalization(name="bn3_1")(x)
    x = layers.Activation("relu")(x)
    x = layers.Conv2D(128, 3, padding="same", kernel_regularizer=reg, name="conv3_2")(x)
    x = layers.BatchNormalization(name="bn3_2")(x)
    x = layers.Activation("relu")(x)
    x = layers.MaxPooling2D(2, strides=2)(x)
    x = layers.Dropout(dropout_rate * 0.75)(x)

    # Block 4
    x = layers.Conv2D(256, 3, padding="same", kernel_regularizer=reg, name="conv4_1")(x)
    x = layers.BatchNormalization(name="bn4_1")(x)
    x = layers.Activation("relu")(x)
    x = layers.Conv2D(256, 3, padding="same", kernel_regularizer=reg, name="conv4_2")(x)
    x = layers.BatchNormalization(name="bn4_2")(x)
    x = layers.Activation("relu")(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(dropout_rate)(x)

    # Classification head
    x = layers.Dense(512, kernel_regularizer=reg, name="fc1")(x)
    x = layers.BatchNormalization(name="bn_fc1")(x)
    x = layers.Activation("relu")(x)
    x = layers.Dropout(dropout_rate)(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="predictions")(x)

    model = keras.Model(inputs, outputs, name="FlowerCustomCNN")
    return model


# ─────────────────────────────────────────────────────────────
# 2. Transfer Learning Models
# ─────────────────────────────────────────────────────────────

def build_transfer_model(
    base_arch: str = "mobilenetv2",
    input_shape: Tuple[int, int, int] = (224, 224, 3),
    num_classes: int = NUM_CLASSES,
    fine_tune_layers: int = 30,
    dropout_rate: float = 0.3
) -> keras.Model:
    """Build a transfer learning model using a pre-trained backbone.

    Args:
        base_arch: One of mobilenetv2, resnet50, efficientnetb0, efficientnetb4, vgg16.
        input_shape: Input image shape (H, W, C).
        num_classes: Number of output classes.
        fine_tune_layers: Number of top layers of the backbone to unfreeze.
        dropout_rate: Dropout rate for the classification head.

    Returns:
        keras.Model: Transfer learning model ready for training.
    """
    arch_map = {
        "mobilenetv2": MobileNetV2,
        "resnet50": ResNet50,
        "efficientnetb0": EfficientNetB0,
        "efficientnetb4": EfficientNetB4,
        "vgg16": VGG16,
    }
    if base_arch not in arch_map:
        raise ValueError(f"Unknown architecture: {base_arch}. Choose from {list(arch_map)}")

    base_model = arch_map[base_arch](
        input_shape=input_shape,
        include_top=False,
        weights="imagenet"
    )
    base_model.trainable = False

    inputs = keras.Input(shape=input_shape, name="input")
    x = base_model(inputs, training=False)
    x = layers.GlobalAveragePooling2D(name="gap")(x)
    x = layers.Dense(512, activation="relu", name="fc1")(x)
    x = layers.BatchNormalization(name="bn_fc1")(x)
    x = layers.Dropout(dropout_rate, name="dropout1")(x)
    x = layers.Dense(256, activation="relu", name="fc2")(x)
    x = layers.Dropout(dropout_rate * 0.5, name="dropout2")(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="predictions")(x)

    model = keras.Model(inputs, outputs, name=f"Flower_{base_arch.upper()}")
    return model, base_model


def unfreeze_top_layers(base_model: keras.Model, n_layers: int = 30) -> None:
    """Unfreeze the top N layers of a base model for fine-tuning.

    Args:
        base_model: The backbone model.
        n_layers: Number of layers from the top to unfreeze.
    """
    base_model.trainable = True
    for layer in base_model.layers[:-n_layers]:
        layer.trainable = False
    trainable_count = sum(1 for l in base_model.layers if l.trainable)
    logger.info("Fine-tuning: %d trainable layers in backbone.", trainable_count)


# ─────────────────────────────────────────────────────────────
# 3. Model Compilation and Training
# ─────────────────────────────────────────────────────────────

def compile_model(
    model: keras.Model,
    learning_rate: float = 1e-3,
    optimizer_name: str = "adam",
    label_smoothing: float = 0.1
) -> None:
    """Compile a Keras model with loss, optimizer, and metrics.

    Args:
        model: Keras model to compile.
        learning_rate: Learning rate for the optimizer.
        optimizer_name: One of adam, sgd, rmsprop.
        label_smoothing: Label smoothing for cross-entropy loss.
    """
    optimizers = {
        "adam": Adam(learning_rate=learning_rate),
        "sgd": SGD(learning_rate=learning_rate, momentum=0.9, nesterov=True),
        "rmsprop": RMSprop(learning_rate=learning_rate),
    }
    optimizer = optimizers.get(optimizer_name, Adam(learning_rate=learning_rate))
    model.compile(
        optimizer=optimizer,
        loss=keras.losses.CategoricalCrossentropy(label_smoothing=label_smoothing),
        metrics=[
            "accuracy",
            keras.metrics.TopKCategoricalAccuracy(k=3, name="top3_accuracy"),
            keras.metrics.TopKCategoricalAccuracy(k=5, name="top5_accuracy"),
        ],
    )
    logger.info("Model compiled: optimizer=%s, lr=%.5f", optimizer_name, learning_rate)


def get_callbacks(
    checkpoint_path: str = "models/best_model.h5",
    log_dir: str = "logs/tensorboard",
    csv_log_path: str = "logs/training_log.csv",
    patience: int = 10,
    min_lr: float = 1e-7
) -> List[keras.callbacks.Callback]:
    """Create standard training callbacks.

    Args:
        checkpoint_path: Path to save best model checkpoint.
        log_dir: Directory for TensorBoard logs.
        csv_log_path: Path to CSV training log file.
        patience: Patience for EarlyStopping and ReduceLROnPlateau.
        min_lr: Minimum learning rate for ReduceLROnPlateau.

    Returns:
        List of Keras callbacks.
    """
    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(os.path.dirname(csv_log_path), exist_ok=True)
    return [
        ModelCheckpoint(
            checkpoint_path,
            monitor="val_accuracy",
            save_best_only=True,
            save_weights_only=False,
            verbose=1,
        ),
        EarlyStopping(
            monitor="val_accuracy",
            patience=patience,
            restore_best_weights=True,
            verbose=1,
        ),
        ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=patience // 2,
            min_lr=min_lr,
            verbose=1,
        ),
        TensorBoard(log_dir=log_dir, histogram_freq=1),
        CSVLogger(csv_log_path, append=True),
    ]


def train_model(
    model: keras.Model,
    train_dataset: tf.data.Dataset,
    val_dataset: tf.data.Dataset,
    epochs: int = 50,
    callbacks: Optional[List] = None,
    class_weights: Optional[Dict[int, float]] = None
) -> keras.callbacks.History:
    """Train the model on the given datasets.

    Args:
        model: Compiled Keras model.
        train_dataset: Training tf.data.Dataset.
        val_dataset: Validation tf.data.Dataset.
        epochs: Maximum number of training epochs.
        callbacks: List of Keras callbacks.
        class_weights: Optional dict mapping class indices to weights.

    Returns:
        keras.callbacks.History: Training history object.
    """
    logger.info("Starting training for up to %d epochs...", epochs)
    start_time = time.time()
    history = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=epochs,
        callbacks=callbacks or [],
        class_weight=class_weights,
        verbose=1,
    )
    duration = time.time() - start_time
    logger.info("Training complete in %.1f seconds.", duration)
    return history


# ─────────────────────────────────────────────────────────────
# 4. Hyperparameter Optimization with Optuna
# ─────────────────────────────────────────────────────────────

def create_optuna_objective(
    train_dataset: tf.data.Dataset,
    val_dataset: tf.data.Dataset,
    input_shape: Tuple[int, int, int] = (224, 224, 3),
    num_classes: int = NUM_CLASSES,
    n_epochs: int = 10
) -> Callable:
    """Create an Optuna objective function for hyperparameter optimization.

    Args:
        train_dataset: Training dataset.
        val_dataset: Validation dataset.
        input_shape: Model input shape.
        num_classes: Number of output classes.
        n_epochs: Number of training epochs per trial.

    Returns:
        Callable: Optuna objective function.
    """
    def objective(trial: optuna.Trial) -> float:
        arch = trial.suggest_categorical("arch", ["mobilenetv2", "efficientnetb0", "resnet50"])
        lr = trial.suggest_float("learning_rate", 1e-5, 1e-2, log=True)
        dropout = trial.suggest_float("dropout_rate", 0.1, 0.6)
        fine_tune = trial.suggest_int("fine_tune_layers", 10, 50)
        label_smooth = trial.suggest_float("label_smoothing", 0.0, 0.2)

        try:
            model, base_model = build_transfer_model(
                base_arch=arch,
                input_shape=input_shape,
                num_classes=num_classes,
                fine_tune_layers=fine_tune,
                dropout_rate=dropout
            )
            compile_model(model, learning_rate=lr, label_smoothing=label_smooth)
            callbacks = [
                EarlyStopping(monitor="val_accuracy", patience=3, restore_best_weights=True)
            ]
            history = model.fit(
                train_dataset,
                validation_data=val_dataset,
                epochs=n_epochs,
                callbacks=callbacks,
                verbose=0,
            )
            val_acc = max(history.history.get("val_accuracy", [0]))
            tf.keras.backend.clear_session()
            return val_acc
        except Exception as exc:
            logger.warning("Trial failed: %s", exc)
            return 0.0

    return objective


def run_optuna_study(
    train_dataset: tf.data.Dataset,
    val_dataset: tf.data.Dataset,
    n_trials: int = 20,
    n_epochs_per_trial: int = 10,
    study_name: str = "flower_classification",
    storage: Optional[str] = None
) -> Dict:
    """Run Optuna hyperparameter optimization study.

    Args:
        train_dataset: Training dataset.
        val_dataset: Validation dataset.
        n_trials: Number of Optuna trials to run.
        n_epochs_per_trial: Epochs per trial.
        study_name: Name of the Optuna study.
        storage: Optional SQLite or other Optuna storage URL.

    Returns:
        dict: Best hyperparameters and best validation accuracy.
    """
    objective_fn = create_optuna_objective(
        train_dataset, val_dataset, n_epochs=n_epochs_per_trial
    )
    study = optuna.create_study(
        direction="maximize",
        study_name=study_name,
        storage=storage,
        load_if_exists=True,
        sampler=optuna.samplers.TPESampler(seed=42),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5),
    )
    study.optimize(objective_fn, n_trials=n_trials, show_progress_bar=True)
    logger.info("Best trial: val_accuracy=%.4f", study.best_value)
    logger.info("Best params: %s", study.best_params)
    return {"best_params": study.best_params, "best_val_accuracy": study.best_value}


# ─────────────────────────────────────────────────────────────
# 5. Evaluation
# ─────────────────────────────────────────────────────────────

def evaluate_model(
    model: keras.Model,
    test_dataset: tf.data.Dataset,
    class_names: List[str],
    output_dir: str = "reports"
) -> Dict:
    """Evaluate model on test dataset and save evaluation metrics.

    Args:
        model: Trained Keras model.
        test_dataset: Test tf.data.Dataset.
        class_names: List of class name strings.
        output_dir: Directory to save evaluation results.

    Returns:
        dict: Evaluation metrics including accuracy, top-k accuracy, and report.
    """
    os.makedirs(output_dir, exist_ok=True)
    all_preds, all_labels = [], []
    for batch_images, batch_labels in test_dataset:
        preds = model.predict(batch_images, verbose=0)
        all_preds.append(preds)
        all_labels.append(batch_labels.numpy())

    y_pred_proba = np.concatenate(all_preds, axis=0)
    y_true = np.concatenate(all_labels, axis=0).astype(int)
    y_pred = np.argmax(y_pred_proba, axis=1)

    top1_acc = accuracy_score(y_true, y_pred)
    top3_acc = top_k_accuracy_score(y_true, y_pred_proba, k=3)
    top5_acc = top_k_accuracy_score(y_true, y_pred_proba, k=5)

    target_names = [class_names[i] if i < len(class_names) else str(i)
                    for i in range(max(y_true.max() + 1, len(class_names)))]
    report = classification_report(y_true, y_pred, target_names=target_names, output_dict=True)
    cm = confusion_matrix(y_true, y_pred)

    metrics = {
        "top1_accuracy": round(top1_acc, 4),
        "top3_accuracy": round(top3_acc, 4),
        "top5_accuracy": round(top5_acc, 4),
        "classification_report": report,
        "confusion_matrix": cm.tolist(),
    }

    with open(os.path.join(output_dir, "evaluation_metrics.json"), "w") as f:
        json.dump({k: v for k, v in metrics.items() if k != "confusion_matrix"}, f, indent=2)

    np.save(os.path.join(output_dir, "confusion_matrix.npy"), cm)
    logger.info("Top-1 Acc: %.4f | Top-3 Acc: %.4f | Top-5 Acc: %.4f",
                top1_acc, top3_acc, top5_acc)
    return metrics


def compute_class_weights(df: pd.DataFrame, label_col: str = "label") -> Dict[int, float]:
    """Compute balanced class weights to handle class imbalance.

    Args:
        df: DataFrame with a label column.
        label_col: Name of the column containing class labels.

    Returns:
        dict: Mapping from class index to weight.
    """
    from sklearn.utils.class_weight import compute_class_weight
    class_names = sorted(df[label_col].unique())
    class_to_idx = {cls: idx for idx, cls in enumerate(class_names)}
    y = df[label_col].map(class_to_idx).values
    weights = compute_class_weight("balanced", classes=np.unique(y), y=y)
    return {int(i): float(w) for i, w in zip(np.unique(y), weights)}


def export_model(
    model: keras.Model,
    export_dir: str = "models/export",
    formats: List[str] = None
) -> None:
    """Export trained model in multiple formats (SavedModel, H5, TFLite, ONNX).

    Args:
        model: Trained Keras model.
        export_dir: Base directory for exports.
        formats: List of formats to export. Defaults to [savedmodel, h5, tflite].
    """
    formats = formats or ["savedmodel", "h5", "tflite"]
    os.makedirs(export_dir, exist_ok=True)

    if "savedmodel" in formats:
        sm_path = os.path.join(export_dir, "saved_model")
        model.save(sm_path)
        logger.info("Saved TF SavedModel to %s", sm_path)

    if "h5" in formats:
        h5_path = os.path.join(export_dir, "model.h5")
        model.save(h5_path)
        logger.info("Saved H5 model to %s", h5_path)

    if "tflite" in formats:
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        tflite_model = converter.convert()
        tflite_path = os.path.join(export_dir, "model.tflite")
        with open(tflite_path, "wb") as f:
            f.write(tflite_model)
        logger.info("Saved TFLite model to %s (%.2f MB)", tflite_path,
                    os.path.getsize(tflite_path) / 1e6)


if __name__ == "__main__":
    logger.info("Building model architectures...")
    custom_model = build_custom_cnn()
    custom_model.summary()
    transfer_model, base = build_transfer_model(base_arch="mobilenetv2")
    compile_model(transfer_model, learning_rate=1e-3)
    transfer_model.summary()
    logger.info("Models built successfully.")
