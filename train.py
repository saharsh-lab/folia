"""
Train the plant disease classifier on the PlantVillage dataset.

Transfer learning with MobileNetV2 (ImageNet weights) on the *colour* images:
  1. train only the new classification head on top of the frozen backbone
  2. unfreeze the top of the backbone and fine-tune with a small learning rate

The data is split 70 / 15 / 15 (train / val / test), stratified per class.
The test split is never seen during training and is used for the final score.

Outputs (written to the project root):
  plant_disease_model.keras   trained model (takes raw 0-255 RGB, 224x224)
  class_indices.json          {index: class_name}
  metrics.json                test accuracy / loss + per-class report

Usage:
  python train.py                                 # uses ./plantvillagedataset/color
  python train.py --data-dir /path/to/color       # or point at the folder yourself
  python train.py --download                      # fetch the dataset with kagglehub
"""

import argparse
import json
import random
from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models

ROOT = Path(__file__).resolve().parent
IMG_SIZE = 224
SEED = 42
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


def set_seeds(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def find_dataset(data_dir: str | None, download: bool) -> Path:
    if data_dir:
        return Path(data_dir)
    local = ROOT / "plantvillagedataset" / "color"
    if local.is_dir() and not download:
        return local
    import kagglehub  # needs ~/.kaggle/kaggle.json or KAGGLE_USERNAME/KAGGLE_KEY

    base = Path(kagglehub.dataset_download("abdallahalidev/plantvillage-dataset"))
    matches = [p for p in base.rglob("color") if p.is_dir()]
    if not matches:
        raise FileNotFoundError(f"No 'color' folder found under {base}")
    return matches[0]


def split_files(data_dir: Path, val_frac: float = 0.15, test_frac: float = 0.15,
                max_per_class: int | None = None):
    """Stratified train/val/test split of image paths. Returns (splits, class_names)."""
    class_names = sorted(d.name for d in data_dir.iterdir() if d.is_dir())
    rng = random.Random(SEED)
    splits = {"train": [], "val": [], "test": []}

    for label, name in enumerate(class_names):
        files = sorted(
            str(p) for p in (data_dir / name).iterdir() if p.suffix.lower() in IMAGE_EXTS
        )
        rng.shuffle(files)
        files = files[:max_per_class]
        n_test = int(len(files) * test_frac)
        n_val = int(len(files) * val_frac)
        splits["test"] += [(f, label) for f in files[:n_test]]
        splits["val"] += [(f, label) for f in files[n_test:n_test + n_val]]
        splits["train"] += [(f, label) for f in files[n_test + n_val:]]

    return splits, class_names


def make_dataset(items, batch_size: int, shuffle: bool) -> tf.data.Dataset:
    paths = [p for p, _ in items]
    labels = [l for _, l in items]
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    if shuffle:
        ds = ds.shuffle(len(items), seed=SEED, reshuffle_each_iteration=True)

    def load(path, label):
        img = tf.io.read_file(path)
        img = tf.io.decode_image(img, channels=3, expand_animations=False)
        img = tf.image.resize(img, (IMG_SIZE, IMG_SIZE))
        return tf.cast(img, tf.float32), label

    return (
        ds.map(load, num_parallel_calls=tf.data.AUTOTUNE)
        .batch(batch_size)
        .prefetch(tf.data.AUTOTUNE)
    )


def build_model(num_classes: int) -> tuple[tf.keras.Model, tf.keras.Model]:
    augment = tf.keras.Sequential(
        [
            layers.RandomFlip("horizontal_and_vertical"),
            layers.RandomRotation(0.15),
            layers.RandomZoom(0.15),
            layers.RandomContrast(0.15),
        ],
        name="augmentation",
    )

    base = tf.keras.applications.MobileNetV2(
        input_shape=(IMG_SIZE, IMG_SIZE, 3), include_top=False, weights="imagenet"
    )
    base.trainable = False

    inputs = layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3))  # raw 0-255 RGB
    x = augment(inputs)
    x = layers.Rescaling(1.0 / 127.5, offset=-1.0)(x)  # MobileNetV2 expects [-1, 1]
    x = base(x, training=False)  # keep BatchNorm in inference mode while fine-tuning
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)
    return models.Model(inputs, outputs), base


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", help="folder with one sub-folder per class")
    parser.add_argument("--download", action="store_true", help="download with kagglehub")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--head-epochs", type=int, default=5)
    parser.add_argument("--fine-tune-epochs", type=int, default=10)
    parser.add_argument("--unfreeze", type=int, default=40, help="backbone layers to fine-tune")
    parser.add_argument("--max-per-class", type=int, help="cap images per class (quick test)")
    args = parser.parse_args()

    set_seeds()
    data_dir = find_dataset(args.data_dir, args.download)
    splits, class_names = split_files(data_dir, max_per_class=args.max_per_class)
    print(f"Dataset: {data_dir}")
    print(f"{len(class_names)} classes | "
          + " | ".join(f"{k}: {len(v)}" for k, v in splits.items()))

    train_ds = make_dataset(splits["train"], args.batch_size, shuffle=True)
    val_ds = make_dataset(splits["val"], args.batch_size, shuffle=False)
    test_ds = make_dataset(splits["test"], args.batch_size, shuffle=False)

    model, base = build_model(len(class_names))
    model_path = ROOT / "plant_disease_model.keras"
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=3, restore_best_weights=True
        ),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.3, patience=1),
        tf.keras.callbacks.ModelCheckpoint(model_path, monitor="val_loss", save_best_only=True),
    ]

    # Phase 1: train the new head only
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    model.summary()
    model.fit(train_ds, validation_data=val_ds, epochs=args.head_epochs, callbacks=callbacks,
              verbose=2)

    # Phase 2: fine-tune the top of the backbone
    base.trainable = True
    for layer in base.layers[:-args.unfreeze]:
        layer.trainable = False
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-5),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.head_epochs + args.fine_tune_epochs,
        initial_epoch=args.head_epochs,
        callbacks=callbacks,
        verbose=2,
    )

    # Final evaluation on the untouched test split
    test_loss, test_acc = model.evaluate(test_ds)
    print(f"\nTest accuracy: {test_acc:.2%}  (loss {test_loss:.4f})")

    y_true = np.array([label for _, label in splits["test"]])
    y_pred = np.argmax(model.predict(test_ds), axis=1)
    per_class = {
        name: float(np.mean(y_pred[y_true == i] == i)) for i, name in enumerate(class_names)
    }
    worst = sorted(per_class.items(), key=lambda kv: kv[1])[:5]
    print("Lowest per-class accuracy:")
    for name, acc in worst:
        print(f"  {acc:.2%}  {name}")

    model.save(model_path)
    with open(ROOT / "class_indices.json", "w") as f:
        json.dump({str(i): name for i, name in enumerate(class_names)}, f, indent=2)
    with open(ROOT / "metrics.json", "w") as f:
        json.dump(
            {
                "test_accuracy": float(test_acc),
                "test_loss": float(test_loss),
                "split_sizes": {k: len(v) for k, v in splits.items()},
                "per_class_accuracy": per_class,
            },
            f,
            indent=2,
        )
    print(f"Saved {model_path.name}, class_indices.json, metrics.json")


if __name__ == "__main__":
    main()
