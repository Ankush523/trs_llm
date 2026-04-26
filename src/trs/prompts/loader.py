from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def load_mapping(path: str | Path) -> dict[str, Any]:
    payload = Path(path).read_text(encoding="utf-8")
    try:
        return json.loads(payload)
    except json.JSONDecodeError:
        try:
            import yaml  # type: ignore
        except ImportError as exc:
            raise RuntimeError(
                f"Config file {path} is not JSON-shaped YAML and PyYAML is not installed."
            ) from exc
        data = yaml.safe_load(payload)
        if not isinstance(data, dict):
            raise ValueError(f"Config file {path} did not parse to a mapping.")
        return data
