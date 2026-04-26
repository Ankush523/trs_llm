from __future__ import annotations

from datetime import datetime, timezone

from trs.utils.hashing import stable_hash


def make_run_id(prefix: str, seed: str) -> str:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")
    return f"{prefix}-{timestamp}-{stable_hash(seed)[:8]}"


def make_record_id(prefix: str, seed: str) -> str:
    return f"{prefix}-{stable_hash(seed)[:12]}"
