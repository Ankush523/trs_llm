from __future__ import annotations

from enum import Enum


class StrEnum(str, Enum):
    def __str__(self) -> str:
        return self.value


class TaskType(StrEnum):
    MATH = "math"
    CODING = "coding"


class OutcomeType(StrEnum):
    SUCCESS = "success"
    FAILURE = "failure"


class RunMode(StrEnum):
    DIRECT = "direct"
    TRS = "trs"


class RetrieverType(StrEnum):
    BM25 = "bm25"
    DENSE = "dense"
    HYBRID = "hybrid"
