# Sudoku Scanner & Solver

Scan a photo of a Sudoku grid, read the digits with a CNN, and solve the
puzzle with backtracking.

```
photo.jpg ──▶ OpenCV pipeline ──▶ CNN digit reader ──▶ backtracking solver ──▶ solution
            (find & warp grid)   (Keras, MNIST-trained)
```

## How it works

1. **Vision** (`sudoku/vision.py`) — OpenCV finds the grid's outer contour,
   perspective-warps it to a flat square, slices it into 81 cells, and crops &
   centers each digit MNIST-style.
2. **Recognition** (`sudoku/model.py`, `train.py`) — a small Keras CNN trained
   on MNIST classifies each non-empty cell.
3. **Solving** (`sudoku/solver.py`) — a backtracking search with the
   minimum-remaining-values heuristic fills the grid.

## Setup

```bash
pip install -r requirements.txt
```

## Train the digit model

Downloads MNIST (cached after the first run) and saves weights to
`models/digit_model.h5`:

```bash
python train.py            # defaults: 8 epochs
python train.py --epochs 15
```

## Solve a puzzle from a photo

```bash
python solve.py path/to/photo.jpg
python solve.py path/to/photo.jpg --debug
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

- The model is trained on **handwritten** MNIST digits. It works on printed
  puzzles too, but accuracy varies; the training script adds rotation/zoom
  augmentation to help generalize. For best results on printed grids you can
  retrain on a printed-font digit dataset using the same `build_model()`.
- A single misread digit can make a valid puzzle unsolvable — `solve.py` will
  say so and suggest re-scanning.

## Project layout

```
solve.py              # CLI entrypoint: photo -> solution
train.py              # train the CNN on MNIST
sudoku/
  solver.py           # backtracking solver
  vision.py           # OpenCV grid detection + cell extraction
  model.py            # CNN architecture + load/save
tests/
  test_solver.py      # solver unit tests
```
