from __future__ import annotations

from trs.domain.results import MetricsSummary
from trs.domain.schemas import InferenceResult


def summarize_results(results: list[InferenceResult]) -> MetricsSummary:
    total = len(results)
    correct = sum(1 for result in results if result.is_correct)
    total_input = sum(result.usage.input_tokens for result in results)
    total_output = sum(result.usage.output_tokens for result in results)
    total_latency = sum(result.usage.latency_ms for result in results)
    total_cost = sum(result.cost_estimate for result in results)
    fallback_count = sum(1 for result in results if result.fallback_used)
    return MetricsSummary(
        total=total,
        correct=correct,
        accuracy=(correct / total) if total else 0.0,
        avg_input_tokens=(total_input / total) if total else 0.0,
        avg_output_tokens=(total_output / total) if total else 0.0,
        avg_latency_ms=(total_latency / total) if total else 0.0,
        total_cost=total_cost,
        avg_cost=(total_cost / total) if total else 0.0,
        fallback_rate=(fallback_count / total) if total else 0.0,
    )
