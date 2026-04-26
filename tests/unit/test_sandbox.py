from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from trs.evaluators.sandbox import run_python_code_with_tests


class SandboxTest(unittest.TestCase):
    def test_python_code_passes(self) -> None:
        ok, reason = run_python_code_with_tests("def add(a, b):\n    return a + b\n", ["assert add(1, 2) == 3"])
        self.assertTrue(ok)
        self.assertEqual(reason, "passed")

    def test_unsafe_import_is_rejected(self) -> None:
        ok, reason = run_python_code_with_tests("import os\n", [])
        self.assertFalse(ok)
        self.assertIn("unsafe-import", reason)


if __name__ == "__main__":
    unittest.main()
