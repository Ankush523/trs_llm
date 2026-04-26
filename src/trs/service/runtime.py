from __future__ import annotations

from pathlib import Path

from trs.domain.enums import RunMode, TaskType
from trs.domain.schemas import ProblemRecord
from trs.inference.direct_runner import DirectInferenceRunner
from trs.inference.trs_runner import TRSInferenceRunner
from trs.models.base import ModelConfig
from trs.models.openai_compatible import build_model_client
from trs.prompts.loader import load_mapping
from trs.retrieval.selector import build_retriever
from trs.skills.compiler import load_skill_cards
from trs.storage.manifest import read_manifest
from trs.utils.ids import make_run_id


class ReasoningService:
    def __init__(self, model_config_path: str, prompt_config_path: str, library_dir: str | None = None):
        self.model_config = ModelConfig.from_dict(load_mapping(model_config_path))
        self.client = build_model_client(self.model_config)
        self.prompt_config = load_mapping(prompt_config_path)
        self.library_dir = library_dir
        self.profile = {"name": "service-default", "retriever_type": "bm25", "top_k": 1, "score_threshold": 0.0, "max_skill_chars": 800}
        self.cards = []
        self.retriever = None
        if library_dir:
            manifest = read_manifest(Path(library_dir) / "library_manifest.json")
            self.profile = dict(manifest.get("metadata", {}).get("profile", self.profile))
            self.cards = load_skill_cards(Path(library_dir) / "cards.jsonl")
            self.retriever = build_retriever(self.profile)

    def reason(self, payload: dict) -> dict:
        task_type = TaskType(payload["task_type"])
        problem = ProblemRecord(
            id=payload.get("problem_id") or make_run_id("problem", payload["question"]),
            task_type=task_type,
            dataset=payload.get("dataset", "api"),
            question=payload["question"],
            answer=payload.get("answer", ""),
            tests=list(payload.get("tests", [])),
            entry_point=payload.get("entry_point", ""),
            metadata=dict(payload.get("metadata", {})),
        )
        mode = RunMode(payload.get("mode", "direct"))
        if mode == RunMode.DIRECT or not self.cards or self.retriever is None:
            result = DirectInferenceRunner(self.client, self.prompt_config).run(problem)
        else:
            result = TRSInferenceRunner(self.client, self.prompt_config, self.retriever, self.profile).run(problem, self.cards)
        return {
            "answer": result.final_answer,
            "used_skills": [skill.to_dict() for skill in result.used_skills],
            "usage": result.usage.to_dict(),
            "cost_estimate": result.cost_estimate,
            "latency_ms": result.usage.latency_ms,
            "run_id": make_run_id("reason", problem.id),
        }
