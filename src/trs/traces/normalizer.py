from __future__ import annotations

from trs.domain.enums import TaskType
from trs.evaluators.coding import extract_code_answer
from trs.evaluators.math import extract_math_answer


def normalize_final_answer(task_type: TaskType, response_text: str) -> str:
    if task_type == TaskType.MATH:
        return extract_math_answer(response_text)
    if task_type == TaskType.CODING:
        return extract_code_answer(response_text)
    return response_text.strip()
