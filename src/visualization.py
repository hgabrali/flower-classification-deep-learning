"""
visualization.py

Visualization utilities for flower classification deep learning project.
Creates training curves, confusion matrices, sample predictions, Grad-CAM
heatmaps, class distribution plots, and interactive Plotly dashboards.
"""

import os
import logging
from typing import List, Optional, Tuple, Dict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as mpl_cm
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import cv2
import tensorflow as tf
from tensorflow import keras

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

sns.set_style("whitegrid")
plt.rcParams["figure.dpi"] = 120


# ─────────────────────────────────────────────────────────────
# 1. Training History Plots
# ─────────────────────────────────────────────────────────────

def plot_training_history(
    history: keras.callbacks.History,
    save_path: Optional[str] = "reports/training_history.png"
) -> None:
    """Plot training and validation accuracy and loss curves.

    Args:
        history: Keras training history object.
        save_path: Path to save the figure.
    """
    hist = history.history
    epochs = range(1, len(hist["loss"]) + 1)
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    axes[0].plot(epochs, hist["accuracy"], label="Train Accuracy")
    axes[0].plot(epochs, hist["val_accuracy"], label="Val Accuracy")
    axes[0].set_title("Accuracy")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()
    axes[0].grid(True)

    axes[1].plot(epochs, hist["loss"], label="Train Loss")
    axes[1].plot(epochs, hist["val_loss"], label="Val Loss")
    axes[1].set_title("Loss")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()
    axes[1].grid(True)

    if "top3_accuracy" in hist:
        axes[2].plot(epochs, hist["top3_accuracy"], label="Train Top-3 Acc")
        axes[2].plot(epochs, hist["val_top3_accuracy"], label="Val Top-3 Acc")
        axes[2].set_title("Top-3 Accuracy")
        axes[2].set_xlabel("Epoch")
        axes[2].legend()
        axes[2].grid(True)
    else:
        axes[2].axis("off")

    plt.suptitle("Training History", fontsize=16, fontweight="bold")
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches="tight")
        logger.info("Training history plot saved to %s", save_path)
    plt.show()
    plt.close()


def plot_training_history_interactive(
    history_dict: Dict,
    save_path: Optional[str] = "reports/training_history_interactive.html"
) -> go.Figure:
    """Create an interactive Plotly plot of training history.

    Args:
        history_dict: Dictionary with training metrics (from history.history).
        save_path: Optional path to save the HTML file.

    Returns:
        go.Figure: Plotly figure.
    """
    epochs = list(range(1, len(history_dict["loss"]) + 1))
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=("Model Accuracy", "Model Loss")
    )
    fig.add_trace(go.Scatter(x=epochs, y=history_dict["accuracy"],
                             name="Train Accuracy", line=dict(color="blue")), row=1, col=1)
    fig.add_trace(go.Scatter(x=epochs, y=history_dict["val_accuracy"],
                             name="Val Accuracy", line=dict(color="orange", dash="dash")), row=1, col=1)
    fig.add_trace(go.Scatter(x=epochs, y=history_dict["loss"],
                             name="Train Loss", line=dict(color="red")), row=1, col=2)
    fig.add_trace(go.Scatter(x=epochs, y=history_dict["val_loss"],
                             name="Val Loss", line=dict(color="purple", dash="dash")), row=1, col=2)
    fig.update_layout(title="Training History Dashboard", height=450)
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.write_html(save_path)
        logger.info("Interactive training history saved to %s", save_path)
    return fig


# ─────────────────────────────────────────────────────────────
# 2. Confusion Matrix
# ─────────────────────────────────────────────────────────────

def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: List[str],
    top_n: int = 20,
    save_path: Optional[str] = "reports/confusion_matrix.png"
) -> None:
    """Plot a normalized confusion matrix heatmap.

    Args:
        cm: Confusion matrix array (n_classes x n_classes).
        class_names: List of class name strings.
        top_n: Show only top N most confused classes if total > top_n.
        save_path: Path to save the figure.
    """
    n = cm.shape[0]
    if n > top_n:
        # Show only the top_n classes with the most misclassifications
        row_errors = cm.sum(axis=1) - np.diag(cm)
        top_indices = np.argsort(row_errors)[::-1][:top_n]
        top_indices = np.sort(top_indices)
        cm = cm[np.ix_(top_indices, top_indices)]
        class_names = [class_names[i] for i in top_indices]

    cm_norm = cm.astype(float) / (cm.sum(axis=1, keepdims=True) + 1e-9)
    fig_size = max(10, len(class_names) // 2)
    plt.figure(figsize=(fig_size, fig_size))
    sns.heatmap(
        cm_norm,
        annot=len(class_names) <= 25,
        fmt=".2f",
        xticklabels=class_names,
        yticklabels=class_names,
        cmap="Blues",
        linewidths=0.5,
    )
    plt.title("Normalized Confusion Matrix", fontsize=14, fontweight="bold")
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches="tight")
        logger.info("Confusion matrix saved to %s", save_path)
    plt.show()
    plt.close()


# ─────────────────────────────────────────────────────────────
# 3. Sample Predictions
# ─────────────────────────────────────────────────────────────

def plot_sample_predictions(
    images: np.ndarray,
    true_labels: List[int],
    pred_labels: List[int],
    pred_probs: np.ndarray,
    class_names: List[str],
    n_samples: int = 16,
    save_path: Optional[str] = "reports/sample_predictions.png"
) -> None:
    """Display a grid of sample images with true vs predicted labels.

    Args:
        images: Array of images (N, H, W, C), values in [0, 1].
        true_labels: True class indices.
        pred_labels: Predicted class indices.
        pred_probs: Prediction probability arrays (N, num_classes).
        class_names: List of class name strings.
        n_samples: Number of samples to display.
        save_path: Path to save the figure.
    """
    n = min(n_samples, len(images))
    cols = 4
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 4))
    axes = axes.flatten() if rows > 1 else axes
    for i in range(n):
        ax = axes[i]
        img = images[i]
        if img.max() <= 1.0:
            img = (img * 255).clip(0, 255).astype(np.uint8)
        ax.imshow(img)
        true_name = class_names[true_labels[i]] if true_labels[i] < len(class_names) else str(true_labels[i])
        pred_name = class_names[pred_labels[i]] if pred_labels[i] < len(class_names) else str(pred_labels[i])
        conf = float(pred_probs[i][pred_labels[i]])
        color = "green" if true_labels[i] == pred_labels[i] else "red"
        ax.set_title(f"T: {true_name}\nP: {pred_name} ({conf:.1%})", color=color, fontsize=8)
        ax.axis("off")
    for j in range(n, len(axes)):
        axes[j].axis("off")
    plt.suptitle("Sample Predictions (Green=Correct, Red=Wrong)", fontsize=14, fontweight="bold")
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches="tight")
        logger.info("Sample predictions saved to %s", save_path)
    plt.show()
    plt.close()


# ─────────────────────────────────────────────────────────────
# 4. Class Distribution
# ─────────────────────────────────────────────────────────────

def plot_class_distribution(
    df: pd.DataFrame,
    label_col: str = "label",
    top_n: int = 30,
    save_path: Optional[str] = "reports/class_distribution.png"
) -> None:
    """Plot bar chart of class distribution in the dataset.

    Args:
        df: DataFrame with a label column.
        label_col: Name of the label column.
        top_n: Number of top classes to show.
        save_path: Path to save the figure.
    """
    counts = df[label_col].value_counts().head(top_n)
    plt.figure(figsize=(14, 6))
    colors = sns.color_palette("husl", len(counts))
    bars = plt.bar(counts.index, counts.values, color=colors, edgecolor="black", linewidth=0.5)
    for bar in bars:
        h = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2., h + 0.5, str(int(h)),
                 ha="center", va="bottom", fontsize=7)
    plt.title(f"Top {top_n} Flower Class Distribution", fontsize=14, fontweight="bold")
    plt.xlabel("Flower Class")
    plt.ylabel("Number of Images")
    plt.xticks(rotation=45, ha="right", fontsize=8)
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches="tight")
        logger.info("Class distribution plot saved to %s", save_path)
    plt.show()
    plt.close()


# ─────────────────────────────────────────────────────────────
# 5. Grad-CAM Heatmap
# ─────────────────────────────────────────────────────────────

def compute_gradcam(
    model: keras.Model,
    image: np.ndarray,
    last_conv_layer_name: str,
    pred_index: Optional[int] = None
) -> np.ndarray:
    """Compute Grad-CAM heatmap for a given image and model.

    Args:
        model: Trained Keras model.
        image: Input image array of shape (H, W, C), values in [0, 1].
        last_conv_layer_name: Name of the last convolutional layer.
        pred_index: Class index to compute Grad-CAM for (default: predicted class).

    Returns:
        np.ndarray: Grad-CAM heatmap of shape (H, W), values in [0, 1].
    """
    grad_model = keras.Model(
        inputs=model.inputs,
        outputs=[model.get_layer(last_conv_layer_name).output, model.output]
    )
    img_array = np.expand_dims(image, axis=0).astype(np.float32)
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        if pred_index is None:
            pred_index = int(tf.argmax(predictions[0]))
        class_channel = predictions[:, pred_index]
    grads = tape.gradient(class_channel, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-8)
    heatmap = heatmap.numpy()
    heatmap = cv2.resize(heatmap, (image.shape[1], image.shape[0]))
    return heatmap


def overlay_gradcam(
    image: np.ndarray,
    heatmap: np.ndarray,
    alpha: float = 0.4
) -> np.ndarray:
    """Overlay Grad-CAM heatmap on the original image.

    Args:
        image: Original image array (H, W, C), values in [0, 1] or [0, 255].
        heatmap: Grad-CAM heatmap array (H, W), values in [0, 1].
        alpha: Blending coefficient.

    Returns:
        np.ndarray: Blended image with Grad-CAM overlay, values in [0, 255].
    """
    if image.max() <= 1.0:
        image = (image * 255).astype(np.uint8)
    heatmap_colored = (mpl_cm.jet(heatmap)[:, :, :3] * 255).astype(np.uint8)
    overlay = cv2.addWeighted(image, 1 - alpha, heatmap_colored, alpha, 0)
    return overlay


def plot_gradcam_grid(
    model: keras.Model,
    images: np.ndarray,
    true_labels: List[int],
    class_names: List[str],
    last_conv_layer_name: str,
    n_samples: int = 8,
    save_path: Optional[str] = "reports/gradcam_grid.png"
) -> None:
    """Plot a grid of original images with Grad-CAM overlays.

    Args:
        model: Trained Keras model.
        images: Array of images (N, H, W, C).
        true_labels: True class indices.
        class_names: List of class name strings.
        last_conv_layer_name: Name of the last conv layer for Grad-CAM.
        n_samples: Number of samples to visualize.
        save_path: Path to save the figure.
    """
    n = min(n_samples, len(images))
    fig, axes = plt.subplots(n, 2, figsize=(8, n * 3))
    if n == 1:
        axes = [axes]
    for i in range(n):
        img = images[i]
        heatmap = compute_gradcam(model, img, last_conv_layer_name)
        overlay = overlay_gradcam(img, heatmap)
        display_img = (img * 255).clip(0, 255).astype(np.uint8) if img.max() <= 1 else img
        axes[i][0].imshow(display_img)
        cls_name = class_names[true_labels[i]] if true_labels[i] < len(class_names) else str(true_labels[i])
        axes[i][0].set_title(f"Original: {cls_name}", fontsize=9)
        axes[i][0].axis("off")
        axes[i][1].imshow(overlay)
        axes[i][1].set_title("Grad-CAM Overlay", fontsize=9)
        axes[i][1].axis("off")
    plt.suptitle("Grad-CAM Explanations", fontsize=14, fontweight="bold")
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches="tight")
        logger.info("Grad-CAM grid saved to %s", save_path)
    plt.show()
    plt.close()


# ─────────────────────────────────────────────────────────────
# 6. Model Comparison Dashboard
# ─────────────────────────────────────────────────────────────

def plot_model_comparison(
    model_metrics: Dict[str, Dict],
    save_path: Optional[str] = "reports/model_comparison.html"
) -> go.Figure:
    """Create an interactive bar chart comparing multiple models.

    Args:
        model_metrics: Dict mapping model name to dict of metrics
                       (e.g., {"MobileNetV2": {"top1": 0.92, "top5": 0.98}}).
        save_path: Optional path to save interactive HTML.

    Returns:
        go.Figure: Plotly figure.
    """
    model_names = list(model_metrics.keys())
    top1_scores = [model_metrics[m].get("top1_accuracy", 0) for m in model_names]
    top3_scores = [model_metrics[m].get("top3_accuracy", 0) for m in model_names]
    top5_scores = [model_metrics[m].get("top5_accuracy", 0) for m in model_names]

    fig = go.Figure(data=[
        go.Bar(name="Top-1 Accuracy", x=model_names, y=top1_scores, marker_color="steelblue"),
        go.Bar(name="Top-3 Accuracy", x=model_names, y=top3_scores, marker_color="seagreen"),
        go.Bar(name="Top-5 Accuracy", x=model_names, y=top5_scores, marker_color="coral"),
    ])
    fig.update_layout(
        barmode="group",
        title="Model Accuracy Comparison",
        xaxis_title="Model",
        yaxis_title="Accuracy",
        yaxis=dict(range=[0, 1.05]),
        height=500,
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.write_html(save_path)
        logger.info("Model comparison chart saved to %s", save_path)
    return fig


if __name__ == "__main__":
    logger.info("Running visualization demo with synthetic data...")
    n_classes = 10
    class_names = [f"flower_{i}" for i in range(n_classes)]

    history_mock = {
        "accuracy": list(np.linspace(0.1, 0.92, 30)),
        "val_accuracy": list(np.linspace(0.08, 0.87, 30)),
        "loss": list(np.linspace(2.5, 0.3, 30)),
        "val_loss": list(np.linspace(2.6, 0.5, 30)),
    }
    fig = plot_training_history_interactive(history_mock, save_path=None)
    logger.info("Interactive training history plot created.")

    cm_mock = np.random.randint(0, 50, (n_classes, n_classes))
    np.fill_diagonal(cm_mock, np.random.randint(100, 200, n_classes))
    logger.info("Visualization demo complete.")
