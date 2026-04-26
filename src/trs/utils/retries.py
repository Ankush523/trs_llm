from __future__ import annotations

import time
from collections.abc import Callable
from typing import TypeVar

T = TypeVar("T")


def retry(operation: Callable[[], T], attempts: int = 3, delay_s: float = 0.5) -> T:
    last_error: Exception | None = None
    for attempt in range(attempts):
        try:
            return operation()
        except Exception as exc:  # pragma: no cover - thin wrapper
            last_error = exc
            if attempt == attempts - 1:
                break
            time.sleep(delay_s)
    assert last_error is not None
    raise last_error
