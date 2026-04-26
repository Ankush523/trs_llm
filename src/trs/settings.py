from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(slots=True)
class AppSettings:
    project_root: Path
    data_dir: Path
    artifacts_dir: Path
    reports_dir: Path
    run_id_prefix: str = "trs"

    @classmethod
    def discover(cls, cwd: str | Path | None = None) -> "AppSettings":
        root = Path(cwd or Path.cwd()).resolve()
        return cls(
            project_root=root,
            data_dir=root / "data",
            artifacts_dir=root / "artifacts",
            reports_dir=root / "artifacts" / "reports",
        )
