from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from trs.evaluators.math import evaluate_math_answer, normalize_math_answer


class MathEvaluatorTest(unittest.TestCase):
    def test_normalize_integer_and_fraction(self) -> None:
        self.assertEqual(normalize_math_answer("144"), "144")
        self.assertEqual(normalize_math_answer("0.5"), "1/2")
        self.assertEqual(normalize_math_answer("2/4"), "1/2")

    def test_evaluate_math_answer(self) -> None:
        self.assertTrue(evaluate_math_answer("The answer is 42.", "42"))
        self.assertFalse(evaluate_math_answer("43", "42"))


if __name__ == "__main__":
    unittest.main()
