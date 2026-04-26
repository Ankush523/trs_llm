from __future__ import annotations

from trs.domain.results import ComparisonSummary
from trs.evaluators.slicing import build_quartile_slices


def compare_summaries(baseline, candidate, candidate_results):
    return ComparisonSummary(
        baseline=baseline,
        candidate=candidate,
        accuracy_delta=candidate.accuracy - baseline.accuracy,
        avg_output_tokens_delta=candidate.avg_output_tokens - baseline.avg_output_tokens,
        avg_cost_delta=candidate.avg_cost - baseline.avg_cost,
        avg_latency_delta=candidate.avg_latency_ms - baseline.avg_latency_ms,
        slices=build_quartile_slices(candidate_results),
    )
