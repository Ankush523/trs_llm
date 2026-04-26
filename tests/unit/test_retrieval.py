from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from trs.domain.enums import OutcomeType, TaskType
from trs.domain.schemas import SkillCard
from trs.retrieval.bm25 import BM25Retriever
from trs.retrieval.hybrid import HybridRetriever


def make_card(identifier: str, trigger: str, heuristic: str, keywords: list[str]) -> SkillCard:
    return SkillCard(
        id=identifier,
        task_type=TaskType.MATH,
        dataset="math_hendrycks",
        source_problem_id=identifier,
        outcome_type=OutcomeType.SUCCESS,
        trigger=trigger,
        keywords=keywords,
        do=["Use the right operation."],
        avoid=["Careless mistakes."],
        check=["Verify the result."],
        risk=["Rushing."],
        heuristic=heuristic,
        source_trace_ref=f"trace-{identifier}",
        generator_model="mock-generator",
        summarizer_model="mock-summarizer",
    )


class RetrievalTest(unittest.TestCase):
    def test_bm25_prefers_relevant_skill(self) -> None:
        cards = [
            make_card("a", "When multiplying numbers.", "Use multiplication.", ["multiply", "product"]),
            make_card("b", "When checking parity.", "Use modulo 2.", ["parity", "modulo"]),
        ]
        hits = BM25Retriever().retrieve("How do I multiply 12 and 12?", cards, top_k=1)
        self.assertEqual(len(hits), 1)
        self.assertEqual(hits[0].skill_id, "a")

    def test_hybrid_returns_results(self) -> None:
        cards = [
            make_card("a", "When multiplying numbers.", "Use multiplication.", ["multiply", "product"]),
            make_card("b", "When checking parity.", "Use modulo 2.", ["parity", "modulo"]),
        ]
        hits = HybridRetriever().retrieve("How do I check parity with modulo 2?", cards, top_k=2)
        self.assertTrue(hits)
        self.assertEqual(hits[0].skill_id, "b")


if __name__ == "__main__":
    unittest.main()
