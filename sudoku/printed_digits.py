"""Generate synthetic *printed* digit images to augment MNIST training.

The recognition model trained on MNIST alone learns handwritten digits, which
misreads printed Sudoku digits (most notably reading a printed "1" as "7").
Mixing in rendered printed digits closes that domain gap. Images are produced
in the same format as MNIST (28x28, white digit on black) and centered the same
way the vision pipeline centers extracted cells, so training matches inference.
"""

from __future__ import annotations

import cv2
import numpy as np

# OpenCV's built-in vector fonts give us several distinct printed-style shapes
# for free (no font files or extra dependencies needed).
_FONTS = [
    cv2.FONT_HERSHEY_SIMPLEX,
    cv2.FONT_HERSHEY_DUPLEX,
    cv2.FONT_HERSHEY_COMPLEX,
    cv2.FONT_HERSHEY_TRIPLEX,
    cv2.FONT_HERSHEY_COMPLEX_SMALL,
    cv2.FONT_HERSHEY_PLAIN,
]

_CANVAS = 64  # render large, then crop+downscale for clean anti-aliased strokes


def _render(digit: int, font: int, scale: float, thickness: int) -> np.ndarray:
    canvas = np.zeros((_CANVAS, _CANVAS), dtype=np.uint8)
    text = str(digit)
    (w, h), _ = cv2.getTextSize(text, font, scale, thickness)
    x = (_CANVAS - w) // 2
    y = (_CANVAS + h) // 2
    cv2.putText(canvas, text, (x, y), font, scale, 255, thickness, cv2.LINE_AA)
    return canvas


def _center_28(img: np.ndarray) -> np.ndarray | None:
    """Crop to the digit and center it as a 20px glyph in 28x28 (MNIST-style).

    Mirrors `sudoku.vision._center_in_square` so synthetic samples look like
    the cells the model will actually see at inference time.
    """
    contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    x, y, w, h = cv2.boundingRect(max(contours, key=cv2.contourArea))
    crop = img[y : y + h, x : x + w]
    scale = 20.0 / max(w, h)
    rw, rh = max(1, int(w * scale)), max(1, int(h * scale))
    resized = cv2.resize(crop, (rw, rh), interpolation=cv2.INTER_AREA)
    out = np.zeros((28, 28), dtype=np.uint8)
    y0, x0 = (28 - rh) // 2, (28 - rw) // 2
    out[y0 : y0 + rh, x0 : x0 + rw] = resized
    return out


def generate(samples_per_digit: int = 2000, seed: int = 0):
    """Return (x, y) printed-digit data: x is (N,28,28,1) float32 in [0,1].

    Digits 1-9 only (Sudoku has no 0). Each sample uses a random font, scale,
    stroke thickness, and small rotation for variety.
    """
    rng = np.random.default_rng(seed)
    xs, ys = [], []
    for digit in range(1, 10):
        made = 0
        while made < samples_per_digit:
            font = _FONTS[int(rng.integers(len(_FONTS)))]
            scale = float(rng.uniform(1.5, 2.6))
            thickness = int(rng.integers(2, 5))
            img = _render(digit, font, scale, thickness)

            # Small random rotation so the model tolerates slightly skewed scans.
            angle = float(rng.uniform(-8, 8))
            matrix = cv2.getRotationMatrix2D((_CANVAS / 2, _CANVAS / 2), angle, 1.0)
            img = cv2.warpAffine(img, matrix, (_CANVAS, _CANVAS))

            centered = _center_28(img)
            if centered is None:
                continue
            xs.append(centered)
            ys.append(digit)
            made += 1

    x = (np.array(xs, dtype="float32") / 255.0).reshape(-1, 28, 28, 1)
    y = np.array(ys, dtype="int64")
    return x, y
