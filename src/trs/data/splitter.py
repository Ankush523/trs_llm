from __future__ import annotations

import random
from typing import TypeVar

T = TypeVar("T")


def split_records(records: list[T], holdout_ratio: float = 0.2, seed: int = 42) -> tuple[list[T], list[T]]:
    items = list(records)
    rng = random.Random(seed)
    rng.shuffle(items)
    holdout_count = max(1, int(len(items) * holdout_ratio)) if items else 0
    return items[holdout_count:], items[:holdout_count]
