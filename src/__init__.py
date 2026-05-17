"""
Flower Classification Deep Learning Package

This package provides a complete deep learning pipeline for automated
flower species classification using CNNs and transfer learning.

Modules:
    data_preprocessing: Image loading, augmentation, and dataset preparation.
    model_training: CNN architectures, training, evaluation, and export utilities.
    visualization: Training history, confusion matrices, Grad-CAM, and dashboards.
"""

from .data_preprocessing import (
    get_augmentation_pipeline,
    load_image,
    preprocess_image,
    scan_dataset_directory,
    build_dataframe,
    split_dataset,
    create_tf_dataset,
    create_keras_generators,
    generate_sample_data,
    compute_dataset_statistics,
    run_preprocessing_pipeline,
    FLOWER_CLASSES,
    IMAGE_SIZE,
    NUM_CLASSES,
)

from .model_training import (
    build_custom_cnn,
    build_transfer_model,
    unfreeze_top_layers,
    compile_model,
    get_callbacks,
    train_model,
    create_optuna_objective,
    run_optuna_study,
    evaluate_model,
    compute_class_weights,
    export_model,
)

from .visualization import (
    plot_training_history,
    plot_training_history_interactive,
    plot_confusion_matrix,
    plot_sample_predictions,
    plot_class_distribution,
    compute_gradcam,
    overlay_gradcam,
    plot_gradcam_grid,
    plot_model_comparison,
)

__version__ = "1.0.0"
__author__ = "Hande Gabrali-Knobloch"
__email__ = "hgabrali@example.com"
__description__ = "Deep learning pipeline for automated flower species classification"

__all__ = [
    # Data preprocessing
    "get_augmentation_pipeline",
    "load_image",
    "preprocess_image",
    "scan_dataset_directory",
    "build_dataframe",
    "split_dataset",
    "create_tf_dataset",
    "create_keras_generators",
    "generate_sample_data",
    "compute_dataset_statistics",
    "run_preprocessing_pipeline",
    "FLOWER_CLASSES",
    "IMAGE_SIZE",
    "NUM_CLASSES",
    # Model training
    "build_custom_cnn",
    "build_transfer_model",
    "unfreeze_top_layers",
    "compile_model",
    "get_callbacks",
    "train_model",
    "create_optuna_objective",
    "run_optuna_study",
    "evaluate_model",
    "compute_class_weights",
    "export_model",
    # Visualization
    "plot_training_history",
    "plot_training_history_interactive",
    "plot_confusion_matrix",
    "plot_sample_predictions",
    "plot_class_distribution",
    "compute_gradcam",
    "overlay_gradcam",
    "plot_gradcam_grid",
    "plot_model_comparison",
]
