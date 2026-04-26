from __future__ import annotations

from trs.domain.enums import OutcomeType
from trs.domain.schemas import ProblemRecord, SkillCard, TraceRecord
from trs.models.base import ModelClient
from trs.prompts.builders import build_skill_extract_prompt, parse_skill_payload
from trs.skills.validator import validate_skill_payload
from trs.utils.ids import make_record_id


class SkillDistiller:
    def __init__(self, client: ModelClient, prompt_config: dict[str, str]):
        self.client = client
        self.prompt_config = prompt_config

    def distill(self, problem: ProblemRecord, trace: TraceRecord) -> SkillCard:
        system_prompt, user_prompt = build_skill_extract_prompt(self.prompt_config, problem, trace)
        response = self.client.complete(system_prompt, user_prompt, metadata=problem.metadata)
        payload = self._to_payload(problem, trace, response.text)
        valid, reason = validate_skill_payload(payload)
        if not valid:
            raise ValueError(f"Invalid skill card for problem {problem.id}: {reason}")
        outcome_type = OutcomeType.SUCCESS if trace.is_correct else OutcomeType.FAILURE
        return SkillCard(
            id=make_record_id("skill", f"{problem.id}:{trace.id}:{self.client.name}"),
            task_type=problem.task_type,
            dataset=problem.dataset,
            source_problem_id=problem.id,
            outcome_type=outcome_type,
            trigger=str(payload["trigger"]),
            keywords=[str(item) for item in payload["keywords"]],
            do=[str(item) for item in payload["do"]],
            avoid=[str(item) for item in payload["avoid"]],
            check=[str(item) for item in payload["check"]],
            risk=[str(item) for item in payload["risk"]],
            heuristic=str(payload["heuristic"]),
            source_trace_ref=trace.id,
            generator_model=trace.model_name,
            summarizer_model=self.client.name,
            metadata={
                "provider_metadata": response.metadata,
                "source_question": problem.question,
                "reference_answer": problem.answer,
                "trace_correct": trace.is_correct,
            },
        )

    def distill_batch(self, problems: list[ProblemRecord], traces: list[TraceRecord]) -> list[SkillCard]:
        trace_map = {trace.problem_id: trace for trace in traces}
        cards: list[SkillCard] = []
        for problem in problems:
            trace = trace_map.get(problem.id)
            if trace is None:
                continue
            cards.append(self.distill(problem, trace))
        return cards

    def _to_payload(self, problem: ProblemRecord, trace: TraceRecord, response_text: str) -> dict[str, object]:
        try:
            payload = parse_skill_payload(response_text)
        except Exception:
            payload = {}
        if payload:
            return payload
        mock_skill = problem.metadata.get("mock_skill_card")
        if isinstance(mock_skill, dict):
            return mock_skill
        return {
            "trigger": f"When solving a {problem.task_type.value} problem similar to {problem.id}.",
            "keywords": [problem.dataset, problem.task_type.value],
            "do": ["Follow the most direct valid solution path."],
            "avoid": ["Avoid redundant detours."],
            "check": ["Verify the final answer against the requirement."],
            "risk": ["Trace summarization fallback was used."],
            "heuristic": "Reuse the successful pattern when it clearly matches the new task."
            if trace.is_correct
            else "Avoid the failed pattern and re-check the core operation before finalizing.",
        }
