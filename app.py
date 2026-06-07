"""Flask web app for the Sudoku scanner & solver.

Two ways to use it:
  * Upload a photo  -> the grid is pre-filled with detected digits to review/fix.
  * Type values     -> enter givens directly into the 9x9 grid.
Then click Solve.

Run:
    pip install -r requirements.txt
    python train.py            # once, to create models/digit_model.keras
    python app.py              # serves http://127.0.0.1:5000
"""

from __future__ import annotations

import os
import tempfile

from flask import Flask, jsonify, render_template, request

from sudoku.solver import SolveError, is_valid_board, solve

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024  # 10 MB upload cap

# The recognition model is loaded lazily on first scan and cached, so the app
# starts instantly and manual-entry solving works even without a trained model.
_model = None


def get_model():
    global _model
    if _model is None:
        from sudoku.model import load_model

        _model = load_model()
    return _model


def _parse_grid(grid):
    """Validate and coerce an incoming 9x9 grid into ints (0 = empty)."""
    if (
        not isinstance(grid, list)
        or len(grid) != 9
        or any(not isinstance(row, list) or len(row) != 9 for row in grid)
    ):
        raise ValueError("Grid must be 9x9.")
    out = []
    for row in grid:
        new_row = []
        for v in row:
            v = 0 if v in (None, "") else int(v)
            if not 0 <= v <= 9:
                raise ValueError("Cells must be digits 1-9 (or empty).")
            new_row.append(v)
        out.append(new_row)
    return out


@app.get("/")
def index():
    return render_template("index.html")


@app.post("/api/scan")
def scan():
    """Accept an uploaded image and return the detected 9x9 grid."""
    file = request.files.get("image")
    if file is None or not file.filename:
        return jsonify(error="No image uploaded."), 400

    try:
        model = get_model()
    except FileNotFoundError as exc:
        return jsonify(error=str(exc)), 503

    suffix = os.path.splitext(file.filename)[1] or ".png"
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    try:
        file.save(tmp.name)
        tmp.close()
        from sudoku.vision import extract_grid

        grid = extract_grid(tmp.name, model)
    except (ValueError, FileNotFoundError) as exc:
        return jsonify(error=f"Could not read the grid: {exc}"), 422
    finally:
        os.unlink(tmp.name)

    return jsonify(grid=grid)


@app.post("/api/solve")
def solve_endpoint():
    """Accept a 9x9 grid (JSON) and return the solution."""
    data = request.get_json(silent=True) or {}
    try:
        grid = _parse_grid(data.get("grid"))
    except (ValueError, TypeError) as exc:
        return jsonify(error=str(exc)), 400

    if not is_valid_board(grid):
        return jsonify(error="The grid has conflicting values (check rows, columns, boxes)."), 422

    try:
        solution = solve(grid)
    except SolveError as exc:
        return jsonify(error=str(exc)), 422

    return jsonify(solution=solution)


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)
