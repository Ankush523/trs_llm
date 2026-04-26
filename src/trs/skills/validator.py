from __future__ import annotations

from trs.skills.schema import REQUIRED_SKILL_FIELDS


def validate_skill_payload(payload: dict[str, object]) -> tuple[bool, str]:
    missing = [field for field in REQUIRED_SKILL_FIELDS if field not in payload]
    if missing:
        return False, f"missing fields: {', '.join(sorted(missing))}"
    for key in ("trigger", "heuristic"):
        if not str(payload.get(key, "")).strip():
            return False, f"field {key} is empty"
    for key in ("keywords", "do", "avoid", "check", "risk"):
        value = payload.get(key)
        if not isinstance(value, list) or not value:
            return False, f"field {key} must be a non-empty list"
    return True, "valid"
