from __future__ import annotations

import csv
import json
from pathlib import Path

from trs.domain.results import ComparisonSummary, MetricsSummary
from trs.domain.schemas import InferenceResult


def write_run_outputs(output_dir: str | Path, results: list[InferenceResult], summary: MetricsSummary) -> None:
    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)
    (target / "results.json").write_text(
        json.dumps([result.to_dict() for result in results], indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    (target / "summary.json").write_text(json.dumps(summary.to_dict(), indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    with (target / "results.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["problem_id", "is_correct", "input_tokens", "output_tokens", "latency_ms", "cost_estimate"])
        for result in results:
            writer.writerow(
                [
                    result.request.problem_id,
                    result.is_correct,
                    result.usage.input_tokens,
                    result.usage.output_tokens,
                    result.usage.latency_ms,
                    result.cost_estimate,
                ]
            )


def write_comparison_report(output_dir: str | Path, comparison: ComparisonSummary) -> None:
    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)
    (target / "comparison.json").write_text(
        json.dumps(comparison.to_dict(), indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    markdown = [
        "# TRS Comparison",
        "",
        f"- Baseline accuracy: {comparison.baseline.accuracy:.4f}",
        f"- Candidate accuracy: {comparison.candidate.accuracy:.4f}",
        f"- Accuracy delta: {comparison.accuracy_delta:.4f}",
        f"- Avg output tokens delta: {comparison.avg_output_tokens_delta:.4f}",
        f"- Avg cost delta: {comparison.avg_cost_delta:.6f}",
        f"- Avg latency delta: {comparison.avg_latency_delta:.4f}",
        "",
        "## Difficulty slices",
    ]
    for slice_item in comparison.slices:
        markdown.append(
            f"- {slice_item.label}: count={slice_item.count}, accuracy={slice_item.accuracy:.4f}, "
            f"avg_output_tokens={slice_item.avg_output_tokens:.2f}, avg_cost={slice_item.avg_cost:.6f}"
        )
    (target / "comparison.md").write_text("\n".join(markdown) + "\n", encoding="utf-8")
