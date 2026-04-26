from __future__ import annotations

from trs.domain.schemas import RetrievedSkill


def gate_skills(
    skills: list[RetrievedSkill],
    top_k: int,
    score_threshold: float,
    max_skill_chars: int,
) -> list[RetrievedSkill]:
    accepted: list[RetrievedSkill] = []
    total_chars = 0
    for skill in skills:
        if skill.score < score_threshold:
            continue
        skill_chars = len(skill.text)
        if total_chars + skill_chars > max_skill_chars and accepted:
            break
        accepted.append(skill)
        total_chars += skill_chars
        if len(accepted) >= top_k:
            break
    return accepted
