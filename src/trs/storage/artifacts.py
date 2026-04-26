from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(slots=True)
class ArtifactLayout:
    root: Path

    def ensure(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True)

    def child(self, *parts: str) -> Path:
        return self.root.joinpath(*parts)
