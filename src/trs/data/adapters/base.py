from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

from trs.domain.schemas import ProblemRecord


class DatasetAdapter(ABC):
    name: str

    @abstractmethod
    def load(self, source: str | Path) -> list[ProblemRecord]:
        raise NotImplementedError
