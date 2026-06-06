"""Tests for the backtracking solver (no ML dependencies required)."""

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sudoku.solver import SolveError, is_valid_board, solve  # noqa: E402


PUZZLE = [
    [5, 3, 0, 0, 7, 0, 0, 0, 0],
    [6, 0, 0, 1, 9, 5, 0, 0, 0],
    [0, 9, 8, 0, 0, 0, 0, 6, 0],
    [8, 0, 0, 0, 6, 0, 0, 0, 3],
    [4, 0, 0, 8, 0, 3, 0, 0, 1],
    [7, 0, 0, 0, 2, 0, 0, 0, 6],
    [0, 6, 0, 0, 0, 0, 2, 8, 0],
    [0, 0, 0, 4, 1, 9, 0, 0, 5],
    [0, 0, 0, 0, 8, 0, 0, 7, 9],
]

SOLUTION = [
    [5, 3, 4, 6, 7, 8, 9, 1, 2],
    [6, 7, 2, 1, 9, 5, 3, 4, 8],
    [1, 9, 8, 3, 4, 2, 5, 6, 7],
    [8, 5, 9, 7, 6, 1, 4, 2, 3],
    [4, 2, 6, 8, 5, 3, 7, 9, 1],
    [7, 1, 3, 9, 2, 4, 8, 5, 6],
    [9, 6, 1, 5, 3, 7, 2, 8, 4],
    [2, 8, 7, 4, 1, 9, 6, 3, 5],
    [3, 4, 5, 2, 8, 6, 1, 7, 9],
]


def _is_complete_valid(board):
    """A solved board: every row, column, and box is a permutation of 1-9."""
    target = set(range(1, 10))
    for i in range(9):
        if {board[i][j] for j in range(9)} != target:
            return False
        if {board[j][i] for j in range(9)} != target:
            return False
    for br in range(0, 9, 3):
        for bc in range(0, 9, 3):
            box = {board[br + r][bc + c] for r in range(3) for c in range(3)}
            if box != target:
                return False
    return True


class SolverTests(unittest.TestCase):
    def test_solves_known_puzzle(self):
        self.assertEqual(solve(PUZZLE), SOLUTION)

    def test_solution_is_internally_valid(self):
        self.assertTrue(_is_complete_valid(solve(PUZZLE)))

    def test_does_not_mutate_input(self):
        original = [row[:] for row in PUZZLE]
        solve(PUZZLE)
        self.assertEqual(PUZZLE, original)

    def test_already_solved_board(self):
        self.assertEqual(solve(SOLUTION), SOLUTION)

    def test_rejects_conflicting_givens(self):
        bad = [row[:] for row in PUZZLE]
        bad[0][1] = 5  # Duplicate 5 in the first row.
        with self.assertRaises(SolveError):
            solve(bad)

    def test_rejects_unsolvable_board(self):
        # Valid givens, but no completion exists: row 0 forces a 9 in the
        # last cell, while column 8 already contains a 9.
        contradiction = [[0] * 9 for _ in range(9)]
        for c in range(8):
            contradiction[0][c] = c + 1  # cells (0,0..7) = 1..8 -> (0,8) must be 9
        contradiction[1][8] = 9          # but column 8 already has a 9
        with self.assertRaises(SolveError):
            solve(contradiction)

    def test_is_valid_board(self):
        self.assertTrue(is_valid_board(PUZZLE))
        self.assertTrue(is_valid_board(SOLUTION))
        self.assertFalse(is_valid_board([[1] * 9 for _ in range(9)]))

    def test_rejects_malformed_shape(self):
        self.assertFalse(is_valid_board([[0] * 9 for _ in range(8)]))
        self.assertFalse(is_valid_board([[0] * 8 for _ in range(9)]))


if __name__ == "__main__":
    unittest.main()
