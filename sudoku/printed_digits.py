"""Generate synthetic *printed* digit images to augment MNIST training.

A model trained on MNIST alone learns handwritten digits and misreads printed
Sudoku digits (most notably a printed "1" read as "7"). This module renders
digits 1-9 using real TrueType fonts found on the system (Arial/Helvetica/
Liberation/Times/etc.), which closely match the typefaces used in printed and
on-screen Sudoku grids. Output matches MNIST's format (28x28, white digit on
black) and is centered the same way the vision pipeline centers extracted
cells, so training data matches inference inputs.

If no system fonts or Pillow are available, it falls back to OpenCV's built-in
vector fonts so training still runs (just less accurately on real fonts).
"""

from __future__ import annotations

import glob
import os

import cv2
import numpy as np

_CANVAS = 64  # render large, then crop+downscale for clean anti-aliased strokes

# Where TrueType fonts live on macOS and Linux.
_FONT_DIRS = [
    "/System/Library/Fonts",
    "/System/Library/Fonts/Supplemental",
    "/Library/Fonts",
    os.path.expanduser("~/Library/Fonts"),
    "/usr/share/fonts",
    "/usr/local/share/fonts",
]

# Skip fonts that don't render plain Latin digits well (emoji, symbols, icons,
# non-Latin scripts) — they'd inject garbage glyphs into training.
_FONT_BLOCKLIST = (
    "emoji", "symbol", "icon", "webdings", "wingding", "dingbat", "ornament",
    "batang", "gothic", "mincho", "gulim", "kai", "hei", "song", "noto",
    "marker", "brush", "music", "math", "code39", "barcode",
)


def _discover_fonts() -> list[str]:
    found: list[str] = []
    for d in _FONT_DIRS:
        if not os.path.isdir(d):
            continue
        for ext in ("*.ttf", "*.ttc", "*.otf"):
            found.extend(glob.glob(os.path.join(d, "**", ext), recursive=True))
    fonts = [f for f in found if not any(b in os.path.basename(f).lower()
                                         for b in _FONT_BLOCKLIST)]
    return sorted(set(fonts))


def _render_truetype(digit: int, font_path: str, size: int, ImageFont, Image, ImageDraw):
    try:
        font = ImageFont.truetype(font_path, size)
    except Exception:
        return None
    img = Image.new("L", (_CANVAS, _CANVAS), 0)
    draw = ImageDraw.Draw(img)
    text = str(digit)
    bbox = draw.textbbox((0, 0), text, font=font)
    w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
    if w == 0 or h == 0:
        return None
    pos = ((_CANVAS - w) / 2 - bbox[0], (_CANVAS - h) / 2 - bbox[1])
    draw.text(pos, text, fill=255, font=font)
    arr = np.array(img, dtype=np.uint8)
    # Reject glyphs that rendered as near-empty or a solid block (bad font).
    fill_ratio = (arr > 64).mean()
    if fill_ratio < 0.02 or fill_ratio > 0.6:
        return None
    return arr


# OpenCV vector fonts, used only as a fallback when no TrueType fonts exist.
_HERSHEY = [
    cv2.FONT_HERSHEY_SIMPLEX, cv2.FONT_HERSHEY_DUPLEX, cv2.FONT_HERSHEY_COMPLEX,
    cv2.FONT_HERSHEY_TRIPLEX, cv2.FONT_HERSHEY_COMPLEX_SMALL, cv2.FONT_HERSHEY_PLAIN,
]


def _render_hershey(digit: int, font: int, scale: float, thickness: int) -> np.ndarray:
    canvas = np.zeros((_CANVAS, _CANVAS), dtype=np.uint8)
    text = str(digit)
    (w, h), _ = cv2.getTextSize(text, font, scale, thickness)
    cv2.putText(canvas, text, ((_CANVAS - w) // 2, (_CANVAS + h) // 2),
                font, scale, 255, thickness, cv2.LINE_AA)
    return canvas


def _center_28(img: np.ndarray) -> np.ndarray | None:
    """Crop to the digit and center it as a 20px glyph in 28x28 (MNIST-style)."""
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

    Digits 1-9 only (Sudoku has no 0). Prefers real TrueType fonts; falls back
    to OpenCV vector fonts if none are available.
    """
    rng = np.random.default_rng(seed)

    fonts = _discover_fonts()
    pil = None
    if fonts:
        try:
            from PIL import Image, ImageDraw, ImageFont
            pil = (Image, ImageDraw, ImageFont)
        except Exception:
            pil = None
    use_truetype = bool(fonts and pil)

    xs, ys = [], []
    for digit in range(1, 10):
        made = 0
        attempts = 0
        while made < samples_per_digit:
            attempts += 1
            if attempts > samples_per_digit * 20:
                break  # safety valve against a pathological font set
            if use_truetype:
                Image, ImageDraw, ImageFont = pil
                font_path = fonts[int(rng.integers(len(fonts)))]
                size = int(rng.integers(34, 52))
                img = _render_truetype(digit, font_path, size, ImageFont, Image, ImageDraw)
                if img is None:
                    continue
            else:
                img = _render_hershey(
                    digit, _HERSHEY[int(rng.integers(len(_HERSHEY)))],
                    float(rng.uniform(1.5, 2.6)), int(rng.integers(2, 5)),
                )

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
