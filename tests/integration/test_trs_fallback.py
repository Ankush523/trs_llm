from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from trs.domain.enums import OutcomeType, TaskType
from trs.domain.schemas import ProblemRecord, SkillCard
from trs.inference.trs_runner import TRSInferenceRunner
from trs.models.base import ModelConfig
from trs.models.openai_compatible import build_model_client
from trs.prompts.loader import load_mapping
from trs.retrieval.selector import build_retriever


class TRSFallbackTest(unittest.TestCase):
    def test_fallback_when_no_skill_matches_threshold(self) -> None:
        repo_root = Path(__file__).resolve().parents[2]
        client = build_model_client(ModelConfig.from_dict(load_mapping(repo_root / "configs/models/inference.yaml")))
        prompt_config = load_mapping(repo_root / "configs/prompts/trs_infer.yaml")
        problem = ProblemRecord(
            id="math-fallback",
            task_type=TaskType.MATH,
            dataset="math_hendrycks",
            question="Compute 1 + 1.",
            answer="2",
            metadata={"mock_direct_response": "2"},
        )
        card = SkillCard(
            id="skill-x",
            task_type=TaskType.MATH,
            dataset="math_hendrycks",
            source_problem_id="source",
            outcome_type=OutcomeType.SUCCESS,
            trigger="When solving geometry proofs.",
            keywords=["geometry"],
            do=["Use angle chasing."],
            avoid=["Arithmetic."],
            check=["Check the diagram."],
            risk=["Irrelevant retrieval."],
            heuristic="Use geometry patterns.",
            source_trace_ref="trace-x",
            generator_model="mock-generator",
            summarizer_model="mock-summarizer",
        )
        profile = {"name": "strict", "retriever_type": "bm25", "top_k": 1, "score_threshold": 999.0, "max_skill_chars": 200}
        runner = TRSInferenceRunner(client, prompt_config, build_retriever(profile), profile)
        result = runner.run(problem, [card])
        self.assertTrue(result.fallback_used)
        self.assertEqual(result.final_answer, "2")


if __name__ == "__main__":
    unittest.main()
