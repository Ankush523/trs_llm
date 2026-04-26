from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(slots=True)
class ReasonRequest:
    task_type: str
    question: str
    mode: str
    profile: str = ""
    library_id: str = ""
    dataset: str = ""
    problem_id: str = ""
    tests: list[str] = field(default_factory=list)
    answer: str = ""
    entry_point: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class ReasonResponse:
    answer: str
    used_skills: list[dict[str, Any]]
    usage: dict[str, Any]
    cost_estimate: float
    latency_ms: int
    run_id: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
