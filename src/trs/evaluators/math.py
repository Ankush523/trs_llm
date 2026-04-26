from __future__ import annotations

import re
from fractions import Fraction


def extract_math_answer(text: str) -> str:
    cleaned = text.strip()
    if not cleaned:
        return ""
    if "\n" in cleaned:
        cleaned = cleaned.splitlines()[-1].strip()
    match = re.search(r"(-?\d+(?:/\d+)?(?:\.\d+)?)", cleaned)
    return match.group(1) if match else cleaned


def normalize_math_answer(answer: str) -> str:
    candidate = extract_math_answer(answer).replace(",", "").strip()
    if not candidate:
        return ""
    try:
        if "/" in candidate:
            return str(Fraction(candidate))
        if "." in candidate:
            return str(Fraction(candidate).limit_denominator())
        return str(int(candidate))
    except (ValueError, ZeroDivisionError):
        return candidate.lower()


def evaluate_math_answer(predicted: str, expected: str) -> bool:
    return normalize_math_answer(predicted) == normalize_math_answer(expected)
