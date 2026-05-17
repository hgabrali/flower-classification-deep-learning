"""Flower Classification — Deep Learning.

Public API for the flower_classification package.
Imports the most useful symbols from the three sub-modules so that
callers can write, e.g.::

    from src import load_datasets, build_model, plot_training_history

instead of reaching into the individual sub-modules.
"""

from .data_preprocessing import (
    load_datasets,
    get_data_pipelines,
    FLOWER_CLASSES,
    IMAGE_SIZE,
    NUM_CLASSES,
    BATCH_SIZE,
)

from .model_training import (
    build_model,
    compile_model,
    train_model,
    evaluate_model,
    get_callbacks,
    run_optuna_study,
    unfreeze_top_layers,
)

from .visualization import (
    get_last_conv_layer_name,
    make_gradcam_heatmap,
    overlay_gradcam,
    plot_augmentation_examples,
    plot_class_distribution,
    plot_confusion_matrix,
    plot_gradcam_grid,
    plot_image_size_distribution,
    plot_model_comparison,
    plot_per_class_metrics,
    plot_sample_images,
    plot_training_history,
)

__version__ = "2.0.0"
__author__ = "Hande Gabrali-Knobloch"
__license__ = "MIT"
