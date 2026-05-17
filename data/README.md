# Data Directory

This directory contains datasets for the flower classification deep learning project.

---

## Directory Structure

```
data/
├── raw/                    # Original unprocessed images (organized by class)
│   ├── pink_primrose/
│   ├── hard-leaved_pocket_orchid/
│   ├── rose/
│   └── ...                 # 102 flower class folders
├── processed/              # Preprocessed split CSVs
│   ├── train.csv           # Training set file paths and labels
│   ├── val.csv             # Validation set file paths and labels
│   └── test.csv            # Test set file paths and labels
├── sample/                 # Small sample dataset for testing the pipeline
│   ├── pink_primrose/
│   └── ...                 # 5 class folders with 20 synthetic images each
└── README.md               # This file
```

---

## Data Sources

| Source | Description | Classes | Images | Link |
|--------|-------------|---------|--------|------|
| Oxford 102 Flowers | Academic benchmark | 102 | 8,189 | [Link](https://www.robots.ox.ac.uk/~vgg/data/flowers/102/) |
| Kaggle Flowers | Community dataset | 5 | ~4,317 | [Link](https://www.kaggle.com/datasets/alxmamaev/flowers-recognition) |
| TF Datasets | tf_flowers subset | 5 | ~3,670 | [Link](https://www.tensorflow.org/datasets/catalog/tf_flowers) |

---

## Data Schema

### Raw Image Structure

Images are organized in class-named subdirectories:

```
data/raw/
    {class_name}/           # Class folder (e.g., "rose", "sunflower")
        {image_id}.jpg      # JPEG images, variable resolution
```

### Processed CSV Columns

| Column | Type | Description |
|--------|------|-------------|
| filepath | string | Absolute or relative path to image file |
| label | string | Flower class name (e.g., "rose", "sunflower") |
| class_index | int | Integer class index (0-101 for Oxford 102) |

### Label Assignment

Class indices are assigned alphabetically:
```
0: alpine sea holly
1: anthurium
2: artichoke
...
101: yellow iris
```

---

## Dataset Statistics

**Oxford 102 Flower Dataset**

| Metric | Value |
|--------|-------|
| Total images | 8,189 |
| Number of classes | 102 |
| Min images per class | 40 |
| Max images per class | 258 |
| Mean images per class | 80.3 |
| Imbalance ratio | 6.45 |
| Image format | JPEG |
| Average resolution | ~500x666 px |

**Train / Val / Test Split**

| Split | Images | Proportion |
|-------|--------|------------|
| Training | 6,149 | 75.1% |
| Validation | 1,021 | 12.5% |
| Test | 1,019 | 12.4% |
| **Total** | **8,189** | **100%** |

---

## Data Preparation

### Step 1: Download Dataset

```bash
# Option A: Oxford 102 Flowers (recommended)
wget https://www.robots.ox.ac.uk/~vgg/data/flowers/102/102flowers.tgz
tar -xzf 102flowers.tgz -C data/raw/

# Option B: TensorFlow Datasets
python -c "import tensorflow_datasets as tfds; tfds.load(name='oxford_flowers102', split='train')"

# Option C: Kaggle
kaggle datasets download alxmamaev/flowers-recognition
unzip flowers-recognition.zip -d data/raw/
```

### Step 2: Run Preprocessing Pipeline

```python
from src import run_preprocessing_pipeline

train_ds, val_ds, test_ds, class_names = run_preprocessing_pipeline(
    data_dir="data/raw",
    output_dir="data/processed",
    val_size=0.15,
    test_size=0.10,
    batch_size=32
)
```

### Step 3: Verify Data

```python
from src import compute_dataset_statistics
import pandas as pd

df = pd.read_csv("data/processed/train.csv")
stats = compute_dataset_statistics(df)
print(f"Training samples: {stats['total_samples']}")
print(f"Classes: {stats['num_classes']}")
print(f"Imbalance ratio: {stats['imbalance_ratio']:.2f}")
```

---

## Image Augmentation

During training, the following augmentations are applied on-the-fly:

| Augmentation | Probability | Parameters |
|---|---|---|
| Random Rotate 90 | 0.50 | — |
| Horizontal Flip | 0.50 | — |
| Vertical Flip | 0.20 | — |
| Shift/Scale/Rotate | 0.50 | shift=0.1, scale=0.15, rotate=45 |
| Brightness/Contrast | 0.60 | limit=0.2 |
| Gaussian Blur | 0.30 | blur_limit=3 |
| Gaussian Noise | 0.20 | var_limit=(10, 50) |
| Elastic Transform | 0.20 | alpha=1, sigma=50 |
| Normalization | 1.00 | mean=(0.485,0.456,0.406), std=(0.229,0.224,0.225) |

---

## Notes

- Images in `data/raw/` and `data/processed/` are **gitignored** to keep the repo lightweight.
- Use the `generate_sample_data()` function from `src/data_preprocessing.py` to create
  a small synthetic dataset for testing the pipeline without downloading real data.
- All paths in the CSV files should be absolute or relative to the repo root.
- Class imbalance is handled via `compute_class_weights()` during model training.
