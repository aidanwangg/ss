"""Digit-recognition CNN (Keras) used to read cells from a Sudoku photo."""

from __future__ import annotations

import os

# The trained weights live here by default. `train.py` writes this file.
DEFAULT_MODEL_PATH = os.path.join(
    os.path.dirname(os.path.dirname(__file__)), "models", "digit_model.keras"
)

# Cells are normalized to this size before classification (MNIST convention).
INPUT_SIZE = 28


def build_model():
    """Construct the CNN architecture.

    Kept in one place so `train.py` and inference agree on the topology.
    Output is 10 classes (digits 0-9); class 0 is effectively unused for
    Sudoku since empty cells are filtered out before classification.
    """
    from tensorflow import keras
    from tensorflow.keras import layers

    model = keras.Sequential(
        [
            keras.Input(shape=(INPUT_SIZE, INPUT_SIZE, 1)),
            layers.Conv2D(32, 3, activation="relu", padding="same"),
            layers.BatchNormalization(),
            layers.MaxPooling2D(),
            layers.Conv2D(64, 3, activation="relu", padding="same"),
            layers.BatchNormalization(),
            layers.MaxPooling2D(),
            layers.Flatten(),
            layers.Dropout(0.3),
            layers.Dense(128, activation="relu"),
            layers.Dropout(0.3),
            layers.Dense(10, activation="softmax"),
        ],
        name="digit_cnn",
    )
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def load_model(path: str = DEFAULT_MODEL_PATH):
    """Load a trained model from disk.

    Raises a clear error if the model hasn't been trained yet.
    """
    from tensorflow import keras

    # Prefer the requested path; fall back to a legacy .h5 sibling so models
    # trained before the switch to the native .keras format still load.
    if not os.path.exists(path):
        legacy = os.path.splitext(path)[0] + ".h5"
        if os.path.exists(legacy):
            path = legacy
        else:
            raise FileNotFoundError(
                f"No trained model found at {path!r}. "
                "Run `python train.py` first to train and save one."
            )
    return keras.models.load_model(path)
