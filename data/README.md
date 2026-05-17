# Data

This project uses the **Oxford Flowers 102** dataset. No files are stored in
this folder — the dataset is downloaded automatically through
[TensorFlow Datasets (TFDS)](https://www.tensorflow.org/datasets/catalog/oxford_flowers102)
the first time a notebook runs.

## Loading

```python
import tensorflow_datasets as tfds

dataset, info = tfds.load(
    "oxford_flowers102",
    with_info=True,
    as_supervised=True,
)

train_set      = dataset["train"]
validation_set = dataset["validation"]
test_set       = dataset["test"]
```

## Dataset facts

| Split      | Examples |
|------------|----------|
| train      | 1,020    |
| validation | 1,020    |
| test       | 6,149    |
| **total**  | **8,189** |

*Note:* The Oxford benchmark deliberately inverts the usual train/test
size ratio — the *test* split is the largest.  Models are therefore
evaluated on more data than they were trained on, which is the
standard protocol for this benchmark.

## Notes

- **Splits.** The project uses the official TFDS splits directly. Note that the
  *test* split is the largest — this is the standard Oxford Flowers 102
  benchmark protocol, where models are trained on a small set and evaluated on a
  large one.
- **Integer labels.** Because labels are integers rather than one-hot vectors,
  every model is compiled with `sparse_categorical_crossentropy`.
- **Class names.** TFDS exposes numeric label IDs. The standard human-readable
  Oxford 102 species names are provided in `src/data_preprocessing.py`
  (`FLOWER_CLASSES`) for plot labelling only; models are trained on the integer
  labels.
- **Local cache.** TFDS caches the downloaded data under `~/tensorflow_datasets/`
  (or a `tensorflow_datasets/` folder), which is excluded from version control.
