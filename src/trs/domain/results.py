from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(slots=True)
class MetricsSummary:
    total: int
    correct: int
    accuracy: float
    avg_input_tokens: float
    avg_output_tokens: float
    avg_latency_ms: float
    total_cost: float
    avg_cost: float
    fallback_rate: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class DifficultySlice:
    label: str
    count: int
    accuracy: float
    avg_output_tokens: float
    avg_cost: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class ComparisonSummary:
    baseline: MetricsSummary
    candidate: MetricsSummary
    accuracy_delta: float
    avg_output_tokens_delta: float
    avg_cost_delta: float
    avg_latency_delta: float
    slices: list[DifficultySlice] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "baseline": self.baseline.to_dict(),
            "candidate": self.candidate.to_dict(),
            "accuracy_delta": self.accuracy_delta,
            "avg_output_tokens_delta": self.avg_output_tokens_delta,
            "avg_cost_delta": self.avg_cost_delta,
            "avg_latency_delta": self.avg_latency_delta,
            "slices": [slice_item.to_dict() for slice_item in self.slices],
        }
