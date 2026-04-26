from __future__ import annotations

import json

from trs.domain.schemas import ProblemRecord, RetrievedSkill, SkillCard, TraceRecord


def build_direct_prompt(prompt_config: dict[str, str], problem: ProblemRecord) -> tuple[str, str]:
    system = prompt_config["system"]
    user = prompt_config["user_template"].format(
        task_type=problem.task_type.value,
        problem_id=problem.id,
        question=problem.question,
    )
    return system, user


def build_skill_extract_prompt(
    prompt_config: dict[str, str],
    problem: ProblemRecord,
    trace: TraceRecord,
) -> tuple[str, str]:
    system = prompt_config["system"]
    user = prompt_config["user_template"].format(
        task_type=problem.task_type.value,
        problem_id=problem.id,
        question=problem.question,
        reference_answer=problem.answer,
        model_answer=trace.final_answer,
        was_correct=str(trace.is_correct).lower(),
        trace_text=trace.response_text,
    )
    return system, user


def render_skill_text(card: SkillCard) -> str:
    return "\n".join(
        [
            f"Trigger: {card.trigger}",
            f"Heuristic: {card.heuristic}",
            f"Do: {'; '.join(card.do)}",
            f"Avoid: {'; '.join(card.avoid)}",
            f"Check: {'; '.join(card.check)}",
            f"Risk: {'; '.join(card.risk)}",
        ]
    )


def build_trs_prompt(
    prompt_config: dict[str, str],
    problem: ProblemRecord,
    skills: list[RetrievedSkill],
) -> tuple[str, str]:
    system = prompt_config["system"]
    if skills:
        items = []
        for index, skill in enumerate(skills, start=1):
            items.append(f"{index}. {skill.text}")
        skills_block = "\n".join(items)
    else:
        skills_block = "No retrieved skills available."
    user = prompt_config["user_template"].format(
        skills_block=skills_block,
        task_type=problem.task_type.value,
        problem_id=problem.id,
        question=problem.question,
    )
    return system, user


def parse_skill_payload(text: str) -> dict[str, object]:
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end < start:
        raise ValueError("Model output did not contain a JSON object for the skill card.")
    return json.loads(text[start : end + 1])
