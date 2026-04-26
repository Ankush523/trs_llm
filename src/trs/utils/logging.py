from __future__ import annotations

import json
import logging
import logging.config
from pathlib import Path


def configure_logging(config_path: str | Path | None = None) -> None:
    if config_path is None:
        logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
        return
    path = Path(config_path)
    if not path.exists():
        logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
        return
    payload = json.loads(path.read_text())
    logging.config.dictConfig(payload)
