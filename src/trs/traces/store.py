from __future__ import annotations

from pathlib import Path

from trs.domain.schemas import TraceRecord
from trs.storage.jsonl_store import read_jsonl, write_jsonl


def save_traces(path: str | Path, traces: list[TraceRecord]) -> None:
    write_jsonl(path, [trace.to_dict() for trace in traces])


def load_traces(path: str | Path) -> list[TraceRecord]:
    return [TraceRecord.from_dict(row) for row in read_jsonl(path)]
