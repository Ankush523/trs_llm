from __future__ import annotations

from trs.evaluators.sandbox import run_python_code_with_tests


def extract_code_answer(text: str) -> str:
    cleaned = text.strip()
    if "```" not in cleaned:
        return cleaned
    parts = cleaned.split("```")
    if len(parts) >= 3:
        block = parts[1]
        lines = block.splitlines()
        if lines and lines[0].strip().startswith("python"):
            lines = lines[1:]
        return "\n".join(lines).strip()
    return cleaned


def evaluate_python_code(predicted: str, tests: list[str]) -> tuple[bool, str]:
    code = extract_code_answer(predicted)
    return run_python_code_with_tests(code, tests)
