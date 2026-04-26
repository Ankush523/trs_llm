from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from trs.data.registry import get_dataset_adapter
from trs.domain.enums import RunMode
from trs.experiments.runner import ExperimentRunner
from trs.models.base import ModelConfig
from trs.models.openai_compatible import build_model_client
from trs.prompts.loader import load_mapping
from trs.retrieval.selector import build_retriever
from trs.skills.compiler import build_library_snapshot, load_skill_cards, save_skill_cards
from trs.skills.distiller import SkillDistiller
from trs.traces.generator import TraceGenerator


class MathPipelineIntegrationTest(unittest.TestCase):
    def setUp(self) -> None:
        self.repo_root = Path(__file__).resolve().parents[2]
        self.problems = get_dataset_adapter("math_hendrycks").load(self.repo_root / "data/raw/smoke/math_hendrycks.jsonl")
        self.generator_client = build_model_client(ModelConfig.from_dict(load_mapping(self.repo_root / "configs/models/generator.yaml")))
        self.summarizer_client = build_model_client(ModelConfig.from_dict(load_mapping(self.repo_root / "configs/models/summarizer.yaml")))
        self.inference_client = build_model_client(ModelConfig.from_dict(load_mapping(self.repo_root / "configs/models/inference.yaml")))

    def test_direct_to_skill_to_trs(self) -> None:
        trace_prompt = load_mapping(self.repo_root / "configs/prompts/direct.yaml")
        skill_prompt = load_mapping(self.repo_root / "configs/prompts/skill_extract.yaml")
        trs_prompt = load_mapping(self.repo_root / "configs/prompts/trs_infer.yaml")
        traces = TraceGenerator(self.generator_client, trace_prompt).generate_batch(self.problems)
        cards = SkillDistiller(self.summarizer_client, skill_prompt).distill_batch(self.problems, traces)

        with tempfile.TemporaryDirectory() as tmpdir:
            cards_file = Path(tmpdir) / "cards.jsonl"
            save_skill_cards(cards_file, cards)
            profile = load_mapping(self.repo_root / "configs/retrieval/math_bm25.yaml")
            snapshot = build_library_snapshot(load_skill_cards(cards_file), Path(tmpdir) / "library", profile)
            self.assertTrue(Path(snapshot.cards_file).exists())
            retriever = build_retriever(profile)
            direct_results, direct_summary = ExperimentRunner(self.inference_client, trace_prompt, RunMode.DIRECT).run(self.problems)
            trs_results, trs_summary = ExperimentRunner(
                self.inference_client,
                trs_prompt,
                RunMode.TRS,
                retriever=retriever,
                profile=profile,
                cards=cards,
            ).run(self.problems)
            self.assertLess(direct_summary.accuracy, 1.0)
            self.assertEqual(trs_summary.accuracy, 1.0)
            self.assertTrue(any(result.used_skills for result in trs_results))
            self.assertEqual(len(direct_results), len(trs_results))


if __name__ == "__main__":
    unittest.main()
