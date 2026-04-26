from __future__ import annotations

from trs.domain.enums import RunMode
from trs.domain.results import MetricsSummary
from trs.domain.schemas import InferenceResult, ProblemRecord, SkillCard
from trs.evaluators.metrics import summarize_results
from trs.inference.direct_runner import DirectInferenceRunner
from trs.inference.trs_runner import TRSInferenceRunner
from trs.models.base import ModelClient
from trs.retrieval.base import Retriever


class ExperimentRunner:
    def __init__(
        self,
        client: ModelClient,
        prompt_config: dict[str, str],
        mode: RunMode,
        retriever: Retriever | None = None,
        profile: dict[str, object] | None = None,
        cards: list[SkillCard] | None = None,
    ):
        self.client = client
        self.prompt_config = prompt_config
        self.mode = mode
        self.retriever = retriever
        self.profile = profile or {}
        self.cards = cards or []

    def run(self, problems: list[ProblemRecord]) -> tuple[list[InferenceResult], MetricsSummary]:
        results: list[InferenceResult] = []
        if self.mode == RunMode.DIRECT:
            runner = DirectInferenceRunner(self.client, self.prompt_config)
            results = [runner.run(problem) for problem in problems]
        else:
            if self.retriever is None:
                raise ValueError("TRS mode requires a retriever.")
            runner = TRSInferenceRunner(self.client, self.prompt_config, self.retriever, self.profile)
            results = [runner.run(problem, self.cards) for problem in problems]
        return results, summarize_results(results)
