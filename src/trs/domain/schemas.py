from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any

from trs.domain.enums import OutcomeType, RunMode, TaskType


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass(slots=True)
class UsageStats:
    input_tokens: int = 0
    output_tokens: int = 0
    reasoning_tokens: int = 0
    latency_ms: int = 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "UsageStats":
        return cls(**data)


@dataclass(slots=True)
class ProblemRecord:
    id: str
    task_type: TaskType
    dataset: str
    question: str
    answer: str
    tests: list[str] = field(default_factory=list)
    reference_solution: str = ""
    entry_point: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["task_type"] = self.task_type.value
        return payload

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ProblemRecord":
        payload = dict(data)
        payload["task_type"] = TaskType(payload["task_type"])
        return cls(**payload)


@dataclass(slots=True)
class TraceRecord:
    id: str
    problem_id: str
    task_type: TaskType
    dataset: str
    mode: RunMode
    prompt_text: str
    response_text: str
    final_answer: str
    is_correct: bool
    usage: UsageStats
    model_name: str
    created_at: str = field(default_factory=utc_now_iso)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["task_type"] = self.task_type.value
        payload["mode"] = self.mode.value
        payload["usage"] = self.usage.to_dict()
        return payload

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TraceRecord":
        payload = dict(data)
        payload["task_type"] = TaskType(payload["task_type"])
        payload["mode"] = RunMode(payload["mode"])
        payload["usage"] = UsageStats.from_dict(payload["usage"])
        return cls(**payload)


@dataclass(slots=True)
class SkillCard:
    id: str
    task_type: TaskType
    dataset: str
    source_problem_id: str
    outcome_type: OutcomeType
    trigger: str
    keywords: list[str]
    do: list[str]
    avoid: list[str]
    check: list[str]
    risk: list[str]
    heuristic: str
    source_trace_ref: str
    generator_model: str
    summarizer_model: str
    created_at: str = field(default_factory=utc_now_iso)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["task_type"] = self.task_type.value
        payload["outcome_type"] = self.outcome_type.value
        return payload

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SkillCard":
        payload = dict(data)
        payload["task_type"] = TaskType(payload["task_type"])
        payload["outcome_type"] = OutcomeType(payload["outcome_type"])
        return cls(**payload)


@dataclass(slots=True)
class RetrievedSkill:
    skill_id: str
    score: float
    trigger: str
    heuristic: str
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "RetrievedSkill":
        return cls(**data)


@dataclass(slots=True)
class LibrarySnapshot:
    id: str
    task_type: TaskType
    dataset: str
    retriever_type: str
    cards_file: str
    metadata_file: str
    built_at: str = field(default_factory=utc_now_iso)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["task_type"] = self.task_type.value
        return payload

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "LibrarySnapshot":
        payload = dict(data)
        payload["task_type"] = TaskType(payload["task_type"])
        return cls(**payload)


@dataclass(slots=True)
class InferenceRequest:
    task_type: TaskType
    question: str
    dataset: str
    problem_id: str
    mode: RunMode
    profile: str = ""
    library_id: str = ""
    tests: list[str] = field(default_factory=list)
    answer: str = ""
    entry_point: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class InferenceResult:
    request: InferenceRequest
    response_text: str
    final_answer: str
    usage: UsageStats
    used_skills: list[RetrievedSkill]
    is_correct: bool
    model_name: str
    fallback_used: bool = False
    cost_estimate: float = 0.0
    created_at: str = field(default_factory=utc_now_iso)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "request": {
                **self.request.metadata,
                "task_type": self.request.task_type.value,
                "question": self.request.question,
                "dataset": self.request.dataset,
                "problem_id": self.request.problem_id,
                "mode": self.request.mode.value,
                "profile": self.request.profile,
                "library_id": self.request.library_id,
                "tests": self.request.tests,
                "answer": self.request.answer,
                "entry_point": self.request.entry_point,
            },
            "response_text": self.response_text,
            "final_answer": self.final_answer,
            "usage": self.usage.to_dict(),
            "used_skills": [skill.to_dict() for skill in self.used_skills],
            "is_correct": self.is_correct,
            "model_name": self.model_name,
            "fallback_used": self.fallback_used,
            "cost_estimate": self.cost_estimate,
            "created_at": self.created_at,
            "metadata": self.metadata,
        }


@dataclass(slots=True)
class EvalResult:
    problem_id: str
    mode: RunMode
    is_correct: bool
    final_answer: str
    usage: UsageStats
    cost_estimate: float
    used_skills: list[RetrievedSkill] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "problem_id": self.problem_id,
            "mode": self.mode.value,
            "is_correct": self.is_correct,
            "final_answer": self.final_answer,
            "usage": self.usage.to_dict(),
            "cost_estimate": self.cost_estimate,
            "used_skills": [skill.to_dict() for skill in self.used_skills],
        }


@dataclass(slots=True)
class RunManifest:
    run_id: str
    task_type: TaskType
    dataset: str
    mode: RunMode
    model_name: str
    source_path: str
    output_path: str
    created_at: str = field(default_factory=utc_now_iso)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["task_type"] = self.task_type.value
        payload["mode"] = self.mode.value
        return payload
