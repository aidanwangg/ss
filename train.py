"""Train the digit-recognition CNN on MNIST and save the weights.

Usage:
    python train.py [--epochs N] [--out models/digit_model.h5]

Requires internet on first run to download the MNIST dataset (~11 MB),
which Keras caches under ~/.keras/datasets afterwards.
"""

from __future__ import annotations

import argparse
import os

from sudoku.model import DEFAULT_MODEL_PATH, INPUT_SIZE, build_model


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train the digit-recognition CNN for the Sudoku scanner."
    )
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--out", default=DEFAULT_MODEL_PATH)
    parser.add_argument(
        "--printed-samples",
        type=int,
        default=4000,
        help="Synthetic printed digits per class (real fonts). 0 disables.",
    )
    parser.add_argument(
        "--with-mnist",
        action="store_true",
        help="Also train on handwritten MNIST. Off by default: for a printed "
        "Sudoku scanner, handwriting hurts accuracy (printed '1' read as '7').",
    )
    args = parser.parse_args()

    # Imported here so `--help` works without TensorFlow installed.
    import numpy as np
    import tensorflow as tf
    from tensorflow import keras

    x_parts, y_parts, xt_parts, yt_parts = [], [], [], []

    # Printed digits (real fonts) are the primary signal for typeset puzzles.
    if args.printed_samples > 0:
        from sudoku.printed_digits import generate

        px_train, py_train = generate(args.printed_samples, seed=0)
        px_test, py_test = generate(max(1, args.printed_samples // 5), seed=1)
        x_parts.append(px_train); y_parts.append(py_train)
        xt_parts.append(px_test); yt_parts.append(py_test)
        print(f"Added {len(px_train)} printed-digit training samples.")

    # MNIST handwriting is optional and off by default; it can pull printed
    # '1's toward '7'. Enable with --with-mnist if you also scan handwriting.
    if args.with_mnist:
        (mx, my), (mxt, myt) = keras.datasets.mnist.load_data()
        mx = (mx.astype("float32") / 255.0).reshape(-1, INPUT_SIZE, INPUT_SIZE, 1)
        mxt = (mxt.astype("float32") / 255.0).reshape(-1, INPUT_SIZE, INPUT_SIZE, 1)
        x_parts.append(mx); y_parts.append(my)
        xt_parts.append(mxt); yt_parts.append(myt)
        print(f"Added {len(mx)} handwritten MNIST training samples.")

    if not x_parts:
        raise SystemExit("No training data: set --printed-samples > 0 or --with-mnist.")

    x_train = np.concatenate(x_parts, axis=0)
    y_train = np.concatenate(y_parts, axis=0)
    x_test = np.concatenate(xt_parts, axis=0)
    y_test = np.concatenate(yt_parts, axis=0)

    # Light augmentation for tolerance to slight scan skew/shift. Built from
    # Keras preprocessing layers (ImageDataGenerator was removed in Keras 3 /
    # TensorFlow 2.16+). RandomRotation's factor is a fraction of 2*pi.
    augment = keras.Sequential(
        [
            keras.layers.RandomRotation(0.03),
            keras.layers.RandomTranslation(0.1, 0.1),
            keras.layers.RandomZoom(0.1),
        ],
        name="augmentation",
    )

    train_ds = (
        tf.data.Dataset.from_tensor_slices((x_train, y_train))
        .shuffle(10_000)
        .batch(args.batch_size)
        .map(
            lambda x, y: (augment(x, training=True), y),
            num_parallel_calls=tf.data.AUTOTUNE,
        )
        .prefetch(tf.data.AUTOTUNE)
    )

    model = build_model()
    model.summary()
    model.fit(
        train_ds,
        validation_data=(x_test, y_test),
        epochs=args.epochs,
    )

    loss, acc = model.evaluate(x_test, y_test, verbose=0)
    print(f"\nTest accuracy: {acc:.4f}")

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    model.save(args.out)
    print(f"Saved trained model to {args.out}")


if __name__ == "__main__":
    main()
