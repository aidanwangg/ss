"""Sudoku solver using backtracking.

A board is represented as a 9x9 list of lists (or numpy array) of ints,
where 0 denotes an empty cell and 1-9 are filled values.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

Board = List[List[int]]


class SolveError(Exception):
    """Raised when a board is invalid or cannot be solved."""


def _to_list(board) -> Board:
    """Normalize numpy arrays / tuples into a plain list-of-lists of ints."""
    return [[int(v) for v in row] for row in board]


def is_valid_board(board) -> bool:
    """Return True if the given (possibly partial) board has no conflicts.

    Empty cells (0) are ignored. This checks the *givens* are consistent;
    it does not check solvability.
    """
    board = _to_list(board)
    if len(board) != 9 or any(len(row) != 9 for row in board):
        return False

    for r in range(9):
        for c in range(9):
            v = board[r][c]
            if v == 0:
                continue
            if not (1 <= v <= 9):
                return False
            # Temporarily clear the cell so it doesn't conflict with itself.
            board[r][c] = 0
            if not _can_place(board, r, c, v):
                board[r][c] = v
                return False
            board[r][c] = v
    return True


def _can_place(board: Board, row: int, col: int, val: int) -> bool:
    """Check whether `val` can legally go in (row, col)."""
    # Row and column.
    for i in range(9):
        if board[row][i] == val or board[i][col] == val:
            return False
    # 3x3 box.
    box_r, box_c = 3 * (row // 3), 3 * (col // 3)
    for r in range(box_r, box_r + 3):
        for c in range(box_c, box_c + 3):
            if board[r][c] == val:
                return False
    return True


def _find_empty(board: Board) -> Optional[Tuple[int, int]]:
    """Return the next empty cell with the fewest candidates (MRV heuristic).

    Choosing the most-constrained cell first prunes the search tree far
    faster than scanning left-to-right.
    """
    best: Optional[Tuple[int, int]] = None
    best_count = 10
    for r in range(9):
        for c in range(9):
            if board[r][c] != 0:
                continue
            count = sum(1 for v in range(1, 10) if _can_place(board, r, c, v))
            if count == 0:
                # Dead end — return immediately so backtracking can prune.
                return (r, c)
            if count < best_count:
                best, best_count = (r, c), count
                if count == 1:
                    return best
    return best


def _backtrack(board: Board) -> bool:
    cell = _find_empty(board)
    if cell is None:
        return True  # No empty cells left -> solved.
    row, col = cell
    for val in range(1, 10):
        if _can_place(board, row, col, val):
            board[row][col] = val
            if _backtrack(board):
                return True
            board[row][col] = 0  # Undo and try the next value.
    return False


def solve(board) -> Board:
    """Solve a Sudoku puzzle in place-safe fashion and return the solution.

    Args:
        board: 9x9 grid of ints (0 = empty). Accepts lists or numpy arrays.

    Returns:
        A new 9x9 list-of-lists with the completed solution.

    Raises:
        SolveError: if the board is malformed, has conflicting givens, or
            has no solution.
    """
    work = _to_list(board)
    if not is_valid_board(work):
        raise SolveError("Board is malformed or has conflicting givens.")
    if not _backtrack(work):
        raise SolveError("No solution exists for this puzzle.")
    return work


def format_board(board) -> str:
    """Render a board as a human-readable grid string."""
    board = _to_list(board)
    lines = []
    for r in range(9):
        if r % 3 == 0 and r != 0:
            lines.append("------+-------+------")
        cells = []
        for c in range(9):
            if c % 3 == 0 and c != 0:
                cells.append("|")
            cells.append(str(board[r][c]) if board[r][c] != 0 else ".")
        lines.append(" ".join(cells))
    return "\n".join(lines)
