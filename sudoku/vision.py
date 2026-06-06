"""Computer-vision pipeline: photo of a Sudoku grid -> 9x9 array of digits.

Pipeline:
    1. Pre-process (grayscale, blur, adaptive threshold).
    2. Find the largest 4-corner contour -> the puzzle outline.
    3. Warp it to a flat top-down square (perspective transform).
    4. Slice into a 9x9 grid of cells.
    5. For each cell, decide empty vs. digit; classify digits with the CNN.
"""

from __future__ import annotations

from typing import List

import cv2
import numpy as np

from .model import INPUT_SIZE


def _preprocess(gray: np.ndarray) -> np.ndarray:
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    thresh = cv2.adaptiveThreshold(
        blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2
    )
    return thresh


def _find_grid_corners(thresh: np.ndarray) -> np.ndarray:
    """Locate the four outer corners of the Sudoku grid."""
    contours, _ = cv2.findContours(
        thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    if not contours:
        raise ValueError("No contours found — is this a photo of a grid?")

    biggest = max(contours, key=cv2.contourArea)
    peri = cv2.arcLength(biggest, True)
    approx = cv2.approxPolyDP(biggest, 0.02 * peri, True)
    if len(approx) != 4:
        raise ValueError(
            "Could not find a 4-corner grid outline. Try a clearer, "
            "flatter, well-lit photo."
        )
    return _order_corners(approx.reshape(4, 2).astype("float32"))


def _order_corners(pts: np.ndarray) -> np.ndarray:
    """Order four points as [top-left, top-right, bottom-right, bottom-left]."""
    ordered = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1)
    ordered[0] = pts[np.argmin(s)]      # top-left: smallest x+y
    ordered[2] = pts[np.argmax(s)]      # bottom-right: largest x+y
    ordered[1] = pts[np.argmin(diff)]   # top-right: smallest y-x
    ordered[3] = pts[np.argmax(diff)]   # bottom-left: largest y-x
    return ordered


def _warp(gray: np.ndarray, corners: np.ndarray, size: int = 450) -> np.ndarray:
    """Perspective-transform the grid to a flat `size`x`size` square."""
    dst = np.array(
        [[0, 0], [size, 0], [size, size], [0, size]], dtype="float32"
    )
    matrix = cv2.getPerspectiveTransform(corners, dst)
    return cv2.warpPerspective(gray, matrix, (size, size))


def _extract_digit(cell: np.ndarray) -> np.ndarray | None:
    """Return a 28x28 normalized digit image, or None if the cell is empty."""
    # Threshold so digit strokes are white on black.
    thresh = cv2.adaptiveThreshold(
        cell, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2
    )
    h, w = thresh.shape
    # Strip away grid-line borders that bleed into the cell.
    margin = int(0.12 * h)
    inner = thresh[margin : h - margin, margin : w - margin]

    contours, _ = cv2.findContours(
        inner, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    if not contours:
        return None

    largest = max(contours, key=cv2.contourArea)
    x, y, cw, ch = cv2.boundingRect(largest)
    # Reject specks (noise) and full-cell blobs (leftover grid lines).
    area_ratio = (cw * ch) / float(inner.shape[0] * inner.shape[1])
    if area_ratio < 0.04 or ch < 0.25 * inner.shape[0]:
        return None

    digit = inner[y : y + ch, x : x + cw]
    return _center_in_square(digit)


def _center_in_square(digit: np.ndarray) -> np.ndarray:
    """Resize and center a cropped digit into a 28x28 frame, MNIST-style."""
    h, w = digit.shape
    scale = 20.0 / max(h, w)
    resized = cv2.resize(
        digit, (max(1, int(w * scale)), max(1, int(h * scale))),
        interpolation=cv2.INTER_AREA,
    )
    canvas = np.zeros((INPUT_SIZE, INPUT_SIZE), dtype=np.uint8)
    rh, rw = resized.shape
    y0 = (INPUT_SIZE - rh) // 2
    x0 = (INPUT_SIZE - rw) // 2
    canvas[y0 : y0 + rh, x0 : x0 + rw] = resized
    return canvas


def extract_grid(image_path: str, model, debug: bool = False) -> List[List[int]]:
    """Read a Sudoku photo and return a 9x9 list of ints (0 = empty cell)."""
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Could not read image: {image_path!r}")

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    thresh = _preprocess(gray)
    corners = _find_grid_corners(thresh)
    warped = _warp(gray, corners)

    cell_size = warped.shape[0] // 9
    digit_images = []        # (row, col, 28x28 array)
    board = [[0] * 9 for _ in range(9)]

    for r in range(9):
        for c in range(9):
            y, x = r * cell_size, c * cell_size
            cell = warped[y : y + cell_size, x : x + cell_size]
            digit = _extract_digit(cell)
            if digit is not None:
                digit_images.append((r, c, digit))

    # Batch-classify all non-empty cells in one model call.
    if digit_images:
        batch = np.array([d for _, _, d in digit_images], dtype="float32")
        batch = (batch / 255.0).reshape(-1, INPUT_SIZE, INPUT_SIZE, 1)
        preds = model.predict(batch, verbose=0)
        for (r, c, _), pred in zip(digit_images, preds):
            value = int(np.argmax(pred))
            # MNIST class 0 shouldn't appear in Sudoku; treat as empty.
            board[r][c] = value if value != 0 else 0

    if debug:
        print(f"Detected {len(digit_images)} filled cells.")
    return board
