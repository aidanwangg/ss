# Running the Sudoku Scanner & Solver

Step-by-step setup → train → scan, including the gotchas we hit on macOS.

## 1. Environment

Python 3.11 recommended. From the repo root:

```bash
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
```

This installs TensorFlow, OpenCV, NumPy, and Pillow.

**Apple Silicon (M1/M2/M3):** if `tensorflow` won't install/import, use the Mac
build instead: `pip install tensorflow-macos` (optionally `tensorflow-metal`
for GPU).

Verify everything imports:

```bash
python -c "import cv2, numpy, tensorflow, PIL; print('ok')"
```

## 2. macOS certificate fix (one-time)

If training fails downloading data with
`SSL: CERTIFICATE_VERIFY_FAILED: unable to get local issuer certificate`, the
python.org Python is missing its root certificates. Fix it once:

```bash
/Applications/Python\ 3.11/Install\ Certificates.command
```

(Only relevant if you train `--with-mnist`, which downloads MNIST. The default
printed-only training needs no download.)

## 3. Train the digit model

```bash
python train.py
```

- Trains **printed-only by default** (digits rendered from real system fonts),
  which is what typeset Sudoku grids look like. Takes a few minutes on CPU.
- Saves to `models/digit_model.keras`.
- You should see `Added 36000 printed-digit training samples.` near the start.

Useful flags:

| Flag | Purpose |
|------|---------|
| `--printed-samples N` | Printed images per digit (default 4000). |
| `--with-mnist` | Also train on handwritten MNIST (downloads MNIST). Off by default — handwriting hurts printed-digit accuracy. |
| `--epochs N` | Training epochs (default 12). |
| `--out PATH` | Where to save the model. |

> The reported "Test accuracy" (~0.90) is measured across dozens of very
> different fonts and is intentionally pessimistic. On a single clean puzzle
> font, accuracy is effectively 100%.

## 4. Scan and solve a photo

Put a Sudoku image in the project folder, then:

```bash
python solve.py puzzle.png
python solve.py puzzle.png --debug      # also writes debug_cells.png
```

`--debug` prints the number of detected cells and writes **`debug_cells.png`**:
a 9x9 montage of the exact cell images fed to the model, each labeled with its
prediction — invaluable for diagnosing any misread.

## 5. Run the solver tests

No ML dependencies needed:

```bash
python -m unittest discover -s tests -v
```

## Troubleshooting

| Symptom | Fix |
|---|---|
| `ModuleNotFoundError: No module named 'sudoku'` | Run from the repo root with the venv active. |
| `No trained model found...` | Run `python train.py` first. |
| `CERTIFICATE_VERIFY_FAILED` while training | Run the cert command in step 2. |
| `Could not find a 4-corner grid outline` | Use a straighter, well-lit photo with the full grid + border visible. |
| Detected grid has wrong digits | Inspect `debug_cells.png`. If cells look clean but labels are wrong, retrain; if cells look mangled, it's a photo/vision issue. |
| "No solution exists" | Usually one misread digit — compare the detected grid to the photo. |

## Tips for good scans

- Photograph straight-on, grid filling most of the frame.
- Even lighting, minimal glare/shadow.
- The grid's outer border must be clearly visible (used to locate the puzzle).
