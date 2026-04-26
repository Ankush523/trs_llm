from __future__ import annotations

from pathlib import Path

from trs.data.adapters.base import DatasetAdapter
from trs.domain.enums import TaskType
from trs.domain.schemas import ProblemRecord
from trs.storage.jsonl_store import read_jsonl


class MathHendrycksAdapter(DatasetAdapter):
    name = "math_hendrycks"

    def load(self, source: str | Path) -> list[ProblemRecord]:
        rows = read_jsonl(source)
        return [
            ProblemRecord(
                id=str(row["id"]),
                task_type=TaskType.MATH,
                dataset=self.name,
                question=str(row["problem"]),
                answer=str(row["answer"]),
                reference_solution=str(row.get("solution", "")),
                metadata=dict(row.get("metadata", {})),
            )
            for row in rows
        ]
