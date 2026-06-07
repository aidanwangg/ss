# Sudoku Scanner & Solver

Scan a photo of a Sudoku grid, read the digits with a CNN, and solve the
puzzle with backtracking.

```
photo.jpg ──▶ OpenCV pipeline ──▶ CNN digit reader ──▶ backtracking solver ──▶ solution
            (find & warp grid)   (Keras, real-font)
```

## How it works

1. **Vision** (`sudoku/vision.py`) — OpenCV finds the grid's outer contour,
   perspective-warps it to a flat square, slices it into 81 cells, and crops &
   centers each digit.
2. **Recognition** (`sudoku/model.py`, `train.py`) — a small Keras CNN reads
   each non-empty cell. It is trained on **printed digits rendered from real
   system fonts** (`sudoku/printed_digits.py`), which match typeset Sudoku
   grids. (Handwritten MNIST is available via `--with-mnist` but off by
   default — handwriting makes printed `1`s look like `7`s.)
3. **Solving** (`sudoku/solver.py`) — a backtracking search with the
   minimum-remaining-values heuristic fills the grid.

## Setup

```bash
pip install -r requirements.txt
```

See **[RUNNING.md](RUNNING.md)** for full setup (incl. Apple Silicon and the
macOS SSL-certificate fix) and troubleshooting.

## Train the digit model

Trains on printed digits (real fonts) and saves to `models/digit_model.keras`:

```bash
python train.py                 # printed-only, 12 epochs (recommended)
python train.py --with-mnist    # also include handwritten MNIST
python train.py --printed-samples 6000 --epochs 15
```

> The reported "Test accuracy" (~0.90) spans dozens of very different fonts and
> is intentionally pessimistic; on a single clean puzzle font it is effectively
> 100%.

## Solve a puzzle from a photo

```bash
python solve.py path/to/photo.jpg
python solve.py path/to/photo.jpg --debug   # also writes debug_cells.png
```

Example output:

```
Detected puzzle:

5 3 . | . 7 . | . . .
6 . . | 1 9 5 | . . .
...

Solution:

5 3 4 | 6 7 8 | 9 1 2
6 7 2 | 1 9 5 | 3 4 8
...
```

`--debug` writes `debug_cells.png`: a 9x9 montage of the exact cell images fed
to the model, each labeled with its prediction — handy for diagnosing misreads.

## Run the tests

The solver is fully tested and needs no ML dependencies:

```bash
python -m unittest discover -s tests -v
```

## Tips for good scans

- Photograph the grid straight-on, filling most of the frame.
- Even lighting, minimal shadows and glare.
- The grid's outer border should be clearly visible (it's used to locate the
  puzzle).

## Notes & limitations

- The model is trained on **printed** digits from real fonts, so it targets
  typeset/on-screen puzzles. For handwritten puzzles, retrain with
  `--with-mnist`.
- A single misread digit can make a valid puzzle unsolvable — `solve.py` will
  say so and suggest re-scanning. Use `--debug` to see what the model read.

## Project layout

```
solve.py                  # CLI entrypoint: photo -> solution
train.py                  # train the digit CNN (printed fonts; --with-mnist optional)
sudoku/
  solver.py               # backtracking solver
  vision.py               # OpenCV grid detection + cell extraction + debug montage
  model.py                # CNN architecture + load/save
  printed_digits.py       # synthetic printed-digit generator (real fonts)
tests/
  test_solver.py          # solver unit tests
RUNNING.md                # detailed setup, training, scanning, troubleshooting
```
