"""
data_preprocessing.py

Data preprocessing pipeline for flower classification deep learning project.
Handles image loading, augmentation, normalization, and dataset preparation.
"""

import os
import json
import random
import logging
from pathlib import Path
from typing import Tuple, List, Optional, Dict, Generator

import numpy as np
import pandas as pd
from PIL import Image
import cv2
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical
import albumentations as A

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Flower class names (Oxford 102 Flower Dataset compatible)
FLOWER_CLASSES = [
    "pink primrose", "hard-leaved pocket orchid", "canterbury bells",
    "sweet pea", "english marigold", "tiger lily", "moon orchid",
    "bird of paradise", "monkshood", "globe thistle", "snapdragon",
    "colts foot", "king protea", "spear thistle", "yellow iris",
    "globe-flower", "purple coneflower", "peruvian lily", "balloon flower",
    "giant white arum lily", "fire lily", "pincushion flower", "fritillary",
    "red ginger", "grape hyacinth", "corn poppy", "prince of wales feathers",
    "stemless gentian", "artichoke", "sweet william", "carnation",
    "garden phlox", "love in the mist", "mexican aster", "alpine sea holly",
    "ruby-lipped cattleya", "cape flower", "great masterwort", "siam tulip",
    "lenten rose", "barbeton daisy", "daffodil", "sword lily", "poinsettia",
    "bolero deep blue", "wallflower", "marigold", "buttercup", "oxeye daisy",
    "common dandelion", "petunia", "wild pansy", "primula", "sunflower",
    "pelargonium", "bishop of llandaff", "gaura", "geranium", "orange dahlia",
    "pink-yellow dahlia", "cautleya spicata", "japanese anemone", "black-eyed susan",
    "silverbush", "californian poppy", "osteospermum", "spring crocus",
    "bearded iris", "windflower", "tree poppy", "gazania", "azalea",
    "water lilly", "rose", "thorn apple", "morning glory", "passion flower",
    "lotus", "toad lily", "anthurium", "frangipani", "clematis",
    "hibiscus", "columbine", "desert-rose", "tree mallow", "magnolia",
    "cyclamen", "watercress", "canna lily", "hippeastrum", "bee balm",
    "ball moss", "foxglove", "bougainvillea", "camellia", "mallow",
    "mexican petunia", "bromelia", "blanket flower", "trumpet creeper",
    "blackberry lily"
]

IMAGE_SIZE = (224, 224)
NUM_CLASSES = len(FLOWER_CLASSES)


def get_augmentation_pipeline(training: bool = True) -> A.Compose:
    """Create augmentation pipeline using albumentations.

    Args:
        training: If True, apply aggressive augmentation; otherwise minimal.

    Returns:
        A.Compose: Albumentations augmentation pipeline.
    """
    if training:
        return A.Compose([
            A.RandomRotate90(p=0.5),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.2),
            A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.15, rotate_limit=45, p=0.5),
            A.OneOf([
                A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2),
                A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20),
            ], p=0.6),
            A.OneOf([
                A.GaussianBlur(blur_limit=3),
                A.MotionBlur(blur_limit=3),
            ], p=0.3),
            A.GaussNoise(var_limit=(10.0, 50.0), p=0.2),
            A.ElasticTransform(alpha=1, sigma=50, alpha_affine=50, p=0.2),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ])
    else:
        return A.Compose([
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ])


def load_image(image_path: str, target_size: Tuple[int, int] = IMAGE_SIZE) -> np.ndarray:
    """Load and resize a single image.

    Args:
        image_path: Path to the image file.
        target_size: Desired (width, height) of output image.

    Returns:
        np.ndarray: Loaded and resized image array of shape (H, W, 3).
    """
    img = Image.open(image_path).convert("RGB")
    img = img.resize(target_size, Image.LANCZOS)
    return np.array(img, dtype=np.uint8)


def preprocess_image(
    image: np.ndarray,
    augment: bool = False,
    aug_pipeline: Optional[A.Compose] = None
) -> np.ndarray:
    """Preprocess a single image: resize, augment, normalize.

    Args:
        image: Input image array (H, W, 3).
        augment: Whether to apply augmentation.
        aug_pipeline: Optional custom albumentations pipeline.

    Returns:
        np.ndarray: Preprocessed image array.
    """
    if image.shape[:2] != IMAGE_SIZE:
        image = cv2.resize(image, IMAGE_SIZE)
    if augment:
        pipeline = aug_pipeline or get_augmentation_pipeline(training=True)
        augmented = pipeline(image=image)
        image = augmented["image"]
    else:
        norm_pipeline = get_augmentation_pipeline(training=False)
        image = norm_pipeline(image=image)["image"]
    return image.astype(np.float32)


def scan_dataset_directory(data_dir: str) -> Dict[str, List[str]]:
    """Scan dataset directory organized in class subfolders.

    Expected structure:
        data_dir/
            class_name_1/
                img001.jpg
            class_name_2/
                ...

    Args:
        data_dir: Path to root dataset directory. 

    Returns:
        dict: Mapping from class name to list of image file paths.
    """
    data_dir = Path(data_dir)
    class_to_images: Dict[str, List[str]] = {}
    valid_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}
    for class_dir in sorted(data_dir.iterdir()):
        if class_dir.is_dir():
            images = [
                str(f) for f in class_dir.iterdir()
                if f.suffix.lower() in valid_extensions
            ]
            if images:
                class_to_images[class_dir.name] = images
    logger.info("Found %d classes with %d total images.",
                len(class_to_images),
                sum(len(v) for v in class_to_images.values()))
    return class_to_images


def build_dataframe(class_to_images: Dict[str, List[str]]) -> pd.DataFrame:
    """Build a Pandas DataFrame from class-to-images mapping.

    Args:
        class_to_images: Dict mapping class names to image paths.

    Returns:
        pd.DataFrame: DataFrame with columns [filepath, label, class_index].
    """
    records = []
    class_names = sorted(class_to_images.keys())
    class_to_idx = {cls: idx for idx, cls in enumerate(class_names)}
    for class_name, paths in class_to_images.items():
        for path in paths:
            records.append({
                "filepath": path,
                "label": class_name,
                "class_index": class_to_idx[class_name],
            })
    df = pd.DataFrame(records)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    logger.info("Dataset: %d samples, %d classes", len(df), df["label"].nunique())
    return df


def split_dataset(
    df: pd.DataFrame,
    val_size: float = 0.15,
    test_size: float = 0.10,
    random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Stratified split of dataset into train / validation / test sets.

    Args:
        df: Full dataset DataFrame.
        val_size: Proportion for validation.
        test_size: Proportion for test.
        random_state: Random seed for reproducibility.

    Returns:
        Tuple of (train_df, val_df, test_df).
    """
    train_df, test_df = train_test_split(
        df, test_size=test_size, stratify=df["label"], random_state=random_state
    )
    adjusted_val_size = val_size / (1 - test_size)
    train_df, val_df = train_test_split(
        train_df, test_size=adjusted_val_size, stratify=train_df["label"],
        random_state=random_state
    )
    logger.info("Split: train=%d, val=%d, test=%d", len(train_df), len(val_df), len(test_df))
    return (train_df.reset_index(drop=True),
            val_df.reset_index(drop=True),
            test_df.reset_index(drop=True))


def create_tf_dataset(
    df: pd.DataFrame,
    batch_size: int = 32,
    training: bool = True,
    cache: bool = False
) -> tf.data.Dataset:
    """Create a tf.data.Dataset from a DataFrame with on-the-fly augmentation.

    Args:
        df: DataFrame with filepath and class_index columns.
        batch_size: Number of samples per batch.
        training: If True, apply training augmentations.
        cache: If True, cache the dataset in memory.

    Returns:
        tf.data.Dataset: Batched and prefetched dataset.
    """
    aug_pipeline = get_augmentation_pipeline(training=training)

    def load_and_preprocess(filepath, label):
        def _load(fp, lbl):
            img = load_image(fp.decode("utf-8"))
            img = preprocess_image(img, augment=training, aug_pipeline=aug_pipeline)
            return img, lbl
        img, lbl = tf.numpy_function(_load, [filepath, label], [tf.float32, tf.int64])
        img.set_shape([IMAGE_SIZE[0], IMAGE_SIZE[1], 3])
        lbl.set_shape([])
        return img, lbl

    filepaths = df["filepath"].values
    labels = df["class_index"].values.astype(np.int64)
    dataset = tf.data.Dataset.from_tensor_slices((filepaths, labels))
    if training:
        dataset = dataset.shuffle(buffer_size=len(df), seed=42)
    dataset = dataset.map(load_and_preprocess, num_parallel_calls=tf.data.AUTOTUNE)
    if cache:
        dataset = dataset.cache()
    dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return dataset


def create_keras_generators(
    train_dir: str,
    val_dir: str,
    test_dir: str,
    batch_size: int = 32,
    image_size: Tuple[int, int] = IMAGE_SIZE
) -> Tuple[object, object, object]:
    """Create Keras ImageDataGenerators for train/val/test directories.

    Args:
        train_dir: Path to training images directory.
        val_dir: Path to validation images directory.
        test_dir: Path to test images directory.
        batch_size: Batch size for generators.
        image_size: Target image (width, height).

    Returns:
        Tuple of (train_gen, val_gen, test_gen).
    """
    train_datagen = ImageDataGenerator(
        rescale=1.0 / 255,
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode="nearest",
        brightness_range=[0.8, 1.2],
    )
    val_test_datagen = ImageDataGenerator(rescale=1.0 / 255)
    train_gen = train_datagen.flow_from_directory(
        train_dir, target_size=image_size, batch_size=batch_size,
        class_mode="categorical", shuffle=True, seed=42
    )
    val_gen = val_test_datagen.flow_from_directory(
        val_dir, target_size=image_size, batch_size=batch_size,
        class_mode="categorical", shuffle=False
    )
    test_gen = val_test_datagen.flow_from_directory(
        test_dir, target_size=image_size, batch_size=batch_size,
        class_mode="categorical", shuffle=False
    )
    return train_gen, val_gen, test_gen


def generate_sample_data(
    output_dir: str = "data/sample",
    n_classes: int = 5,
    n_images_per_class: int = 20,
    image_size: Tuple[int, int] = IMAGE_SIZE
) -> pd.DataFrame:
    """Generate synthetic sample image data for testing the pipeline.

    Args:
        output_dir: Directory where sample images are saved.
        n_classes: Number of flower classes to generate.
        n_images_per_class: Number of images per class.
        image_size: Image dimensions (width, height).

    Returns:
        pd.DataFrame: DataFrame of generated image paths and labels.
    """
    output_path = Path(output_dir)
    records = []
    selected_classes = FLOWER_CLASSES[:n_classes]
    for cls in selected_classes:
        cls_dir = output_path / cls.replace(" ", "_")
        cls_dir.mkdir(parents=True, exist_ok=True)
        for i in range(n_images_per_class):
            color = tuple(np.random.randint(0, 255, 3).tolist())
            img = Image.new("RGB", image_size, color=color)
            pixels = np.array(img)
            pixels = np.clip(pixels + np.random.randint(-30, 30, pixels.shape), 0, 255).astype(np.uint8)
            img = Image.fromarray(pixels)
            filepath = cls_dir / f"img_{i:04d}.jpg"
            img.save(filepath)
            records.append({"filepath": str(filepath), "label": cls})
    df = pd.DataFrame(records)
    logger.info("Generated %d sample images in %s", len(df), output_dir)
    return df


def compute_dataset_statistics(df: pd.DataFrame) -> Dict[str, object]:
    """Compute statistics for the dataset: class distribution, image properties.

    Args:
        df: DataFrame with filepath and label columns.

    Returns:
        dict: Dataset statistics including class counts and balance metrics.
    """
    class_counts = df["label"].value_counts().to_dict()
    total = len(df)
    num_classes = df["label"].nunique()
    min_count = min(class_counts.values())
    max_count = max(class_counts.values())
    mean_count = total / num_classes if num_classes else 0
    stats = {
        "total_samples": total,
        "num_classes": num_classes,
        "class_counts": class_counts,
        "min_samples_per_class": min_count,
        "max_samples_per_class": max_count,
        "mean_samples_per_class": round(mean_count, 2),
        "imbalance_ratio": round(max_count / min_count, 2) if min_count > 0 else float("inf"),
    }
    return stats


def run_preprocessing_pipeline(
    data_dir: str,
    output_dir: str = "data/processed",
    val_size: float = 0.15,
    test_size: float = 0.10,
    batch_size: int = 32
) -> Tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset, List[str]]:
    """End-to-end data preprocessing pipeline.

    Args:
        data_dir: Root directory with class subfolders.
        output_dir: Directory to save split DataFrames.
        val_size: Validation set proportion.
        test_size: Test set proportion.
        batch_size: Batch size for tf.data.Dataset.

    Returns:
        Tuple of (train_ds, val_ds, test_ds, class_names).
    """
    logger.info("Starting preprocessing pipeline for: %s", data_dir)
    class_to_images = scan_dataset_directory(data_dir)
    class_names = sorted(class_to_images.keys())
    df = build_dataframe(class_to_images)
    stats = compute_dataset_statistics(df)
    logger.info("Dataset stats: total=%d, classes=%d", stats["total_samples"], stats["num_classes"])
    train_df, val_df, test_df = split_dataset(df, val_size=val_size, test_size=test_size)
    os.makedirs(output_dir, exist_ok=True)
    train_df.to_csv(os.path.join(output_dir, "train.csv"), index=False)
    val_df.to_csv(os.path.join(output_dir, "val.csv"), index=False)
    test_df.to_csv(os.path.join(output_dir, "test.csv"), index=False)
    logger.info("Saved split CSVs to %s", output_dir)
    train_ds = create_tf_dataset(train_df, batch_size=batch_size, training=True)
    val_ds = create_tf_dataset(val_df, batch_size=batch_size, training=False)
    test_ds = create_tf_dataset(test_df, batch_size=batch_size, training=False)
    return train_ds, val_ds, test_ds, class_names


if __name__ == "__main__":
    logger.info("Generating sample data for pipeline demonstration...")
    sample_df = generate_sample_data(output_dir="data/sample", n_classes=5, n_images_per_class=10)
    class_map = scan_dataset_directory("data/sample")
    full_df = build_dataframe(class_map)
    train_df, val_df, test_df = split_dataset(full_df)
    stats = compute_dataset_statistics(full_df)
    logger.info("Pipeline demo complete. Stats: %s", stats)

# v2
