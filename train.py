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
    parser = argparse.ArgumentParser(description="Train the digit CNN on MNIST.")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--out", default=DEFAULT_MODEL_PATH)
    args = parser.parse_args()

    # Imported here so `--help` works without TensorFlow installed.
    import numpy as np
    from tensorflow import keras

    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

    # Normalize to [0, 1] and add the channel dimension.
    x_train = (x_train.astype("float32") / 255.0).reshape(-1, INPUT_SIZE, INPUT_SIZE, 1)
    x_test = (x_test.astype("float32") / 255.0).reshape(-1, INPUT_SIZE, INPUT_SIZE, 1)

    # Light augmentation helps the model generalize from handwritten MNIST
    # to the printed/warped digits coming out of the vision pipeline.
    datagen = keras.preprocessing.image.ImageDataGenerator(
        rotation_range=10, width_shift_range=0.1, height_shift_range=0.1, zoom_range=0.1
    )

    model = build_model()
    model.summary()
    model.fit(
        datagen.flow(x_train, y_train, batch_size=args.batch_size),
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
