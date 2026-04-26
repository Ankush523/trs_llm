from __future__ import annotations

from trs.domain.schemas import UsageStats


def estimate_prompt_tokens(text: str) -> int:
    return max(1, len(text.split()))


def make_usage(input_text: str, output_text: str, latency_ms: int = 0) -> UsageStats:
    return UsageStats(
        input_tokens=estimate_prompt_tokens(input_text),
        output_tokens=estimate_prompt_tokens(output_text),
        reasoning_tokens=estimate_prompt_tokens(output_text),
        latency_ms=latency_ms,
    )
