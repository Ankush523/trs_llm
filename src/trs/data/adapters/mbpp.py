from __future__ import annotations

from pathlib import Path

from trs.data.adapters.base import DatasetAdapter
from trs.domain.enums import TaskType
from trs.domain.schemas import ProblemRecord
from trs.storage.jsonl_store import read_jsonl


class MBPPAdapter(DatasetAdapter):
    name = "mbpp"

    def load(self, source: str | Path) -> list[ProblemRecord]:
        rows = read_jsonl(source)
        return [
            ProblemRecord(
                id=str(row["task_id"]),
                task_type=TaskType.CODING,
                dataset=self.name,
                question=str(row["text"]),
                answer=str(row.get("code", "")),
                tests=list(row.get("tests", [])),
                metadata=dict(row.get("metadata", {})),
            )
            for row in rows
        ]
