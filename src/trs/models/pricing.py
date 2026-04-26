from __future__ import annotations

from trs.domain.schemas import UsageStats
from trs.models.base import ModelConfig


def estimate_cost(usage: UsageStats, config: ModelConfig) -> float:
    pricing = config.pricing or {}
    input_per_1k = float(pricing.get("input_per_1k", 0.0))
    output_per_1k = float(pricing.get("output_per_1k", 0.0))
    return round((usage.input_tokens / 1000.0) * input_per_1k + (usage.output_tokens / 1000.0) * output_per_1k, 8)
