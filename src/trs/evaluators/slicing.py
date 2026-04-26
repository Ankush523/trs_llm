from __future__ import annotations

from trs.domain.results import DifficultySlice
from trs.domain.schemas import InferenceResult


def build_quartile_slices(results: list[InferenceResult]) -> list[DifficultySlice]:
    if not results:
        return []
    ranked = sorted(results, key=lambda item: item.usage.output_tokens)
    size = max(1, len(ranked) // 4)
    labels = ["q1", "q2", "q3", "q4"]
    slices: list[DifficultySlice] = []
    for index, label in enumerate(labels):
        start = index * size
        end = None if index == len(labels) - 1 else (index + 1) * size
        chunk = ranked[start:end]
        if not chunk:
            continue
        slices.append(
            DifficultySlice(
                label=label,
                count=len(chunk),
                accuracy=sum(1 for item in chunk if item.is_correct) / len(chunk),
                avg_output_tokens=sum(item.usage.output_tokens for item in chunk) / len(chunk),
                avg_cost=sum(item.cost_estimate for item in chunk) / len(chunk),
            )
        )
    return slices
