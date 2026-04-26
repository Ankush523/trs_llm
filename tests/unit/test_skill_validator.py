from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from trs.skills.validator import validate_skill_payload


class SkillValidatorTest(unittest.TestCase):
    def test_valid_skill_payload(self) -> None:
        payload = {
            "trigger": "When adding integers.",
            "keywords": ["addition"],
            "do": ["Add carefully."],
            "avoid": ["Arithmetic slips."],
            "check": ["Verify with estimation."],
            "risk": ["Rushing."],
            "heuristic": "Decompose and recombine.",
        }
        valid, reason = validate_skill_payload(payload)
        self.assertTrue(valid)
        self.assertEqual(reason, "valid")

    def test_invalid_skill_payload(self) -> None:
        payload = {"trigger": "", "keywords": []}
        valid, reason = validate_skill_payload(payload)
        self.assertFalse(valid)
        self.assertIn("missing fields", reason)


if __name__ == "__main__":
    unittest.main()
