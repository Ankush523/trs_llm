from __future__ import annotations

from trs.domain.enums import RunMode
from trs.domain.schemas import ProblemRecord, TraceRecord
from trs.evaluators.coding import evaluate_python_code
from trs.evaluators.math import evaluate_math_answer
from trs.models.base import ModelClient
from trs.prompts.builders import build_direct_prompt
from trs.traces.normalizer import normalize_final_answer
from trs.utils.ids import make_record_id


class TraceGenerator:
    def __init__(self, client: ModelClient, prompt_config: dict[str, str]):
        self.client = client
        self.prompt_config = prompt_config

    def generate_for_problem(self, problem: ProblemRecord) -> TraceRecord:
        system_prompt, user_prompt = build_direct_prompt(self.prompt_config, problem)
        response = self.client.complete(system_prompt, user_prompt, metadata=problem.metadata)
        final_answer = normalize_final_answer(problem.task_type, response.text)
        if problem.task_type.value == "math":
            is_correct = evaluate_math_answer(final_answer, problem.answer)
        else:
            is_correct, _ = evaluate_python_code(final_answer, problem.tests)
        return TraceRecord(
            id=make_record_id("trace", f"{problem.id}:{self.client.name}:{RunMode.DIRECT.value}"),
            problem_id=problem.id,
            task_type=problem.task_type,
            dataset=problem.dataset,
            mode=RunMode.DIRECT,
            prompt_text=user_prompt,
            response_text=response.text,
            final_answer=final_answer,
            is_correct=is_correct,
            usage=response.usage,
            model_name=self.client.name,
            metadata={"provider_metadata": response.metadata},
        )

    def generate_batch(self, problems: list[ProblemRecord]) -> list[TraceRecord]:
        return [self.generate_for_problem(problem) for problem in problems]
