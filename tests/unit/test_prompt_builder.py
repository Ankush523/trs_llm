from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from trs.domain.enums import TaskType
from trs.domain.schemas import ProblemRecord, RetrievedSkill
from trs.prompts.builders import build_trs_prompt


class PromptBuilderTest(unittest.TestCase):
    def test_trs_prompt_includes_skills(self) -> None:
        problem = ProblemRecord(
            id="math-1",
            task_type=TaskType.MATH,
            dataset="math_hendrycks",
            question="Compute 12 * 12.",
            answer="144",
        )
        prompt_config = {
            "system": "system",
            "user_template": "Retrieved Reasoning Skill(s):\n{skills_block}\nQuestion:\n{question}",
        }
        skills = [
            RetrievedSkill(
                skill_id="skill-1",
                score=1.0,
                trigger="Multiply",
                heuristic="Square the number carefully.",
                text="Trigger: multiply\nHeuristic: square carefully",
            )
        ]
        _, user_prompt = build_trs_prompt(prompt_config, problem, skills)
        self.assertIn("square carefully", user_prompt)
        self.assertIn("Compute 12 * 12.", user_prompt)


if __name__ == "__main__":
    unittest.main()
