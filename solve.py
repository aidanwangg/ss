"""CLI: scan a photo of a Sudoku grid and solve it.

Usage:
    python solve.py path/to/photo.jpg
    python solve.py path/to/photo.jpg --model models/digit_model.h5 --debug
"""

from __future__ import annotations

import argparse
import sys

from sudoku.model import DEFAULT_MODEL_PATH, load_model
from sudoku.solver import SolveError, format_board, solve


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Scan a Sudoku photo and solve it with backtracking."
    )
    parser.add_argument("image", help="Path to a photo of a Sudoku grid.")
    parser.add_argument("--model", default=DEFAULT_MODEL_PATH,
                        help="Path to the trained digit model.")
    parser.add_argument("--debug", action="store_true",
                        help="Print extra diagnostics from the vision step.")
    args = parser.parse_args()

    # Imported lazily so `--help` doesn't require OpenCV/TensorFlow.
    from sudoku.vision import extract_grid

    try:
        model = load_model(args.model)
    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

    try:
        grid = extract_grid(args.image, model, debug=args.debug)
    except (FileNotFoundError, ValueError) as e:
        print(f"Error reading the grid: {e}", file=sys.stderr)
        return 1

    print("Detected puzzle:\n")
    print(format_board(grid))
    print()

    try:
        solution = solve(grid)
    except SolveError as e:
        print(f"Could not solve: {e}", file=sys.stderr)
        print(
            "\nThis is often a digit-recognition error. Re-take the photo "
            "straighter and better-lit, or train the model for more epochs.",
            file=sys.stderr,
        )
        return 2

    print("Solution:\n")
    print(format_board(solution))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
