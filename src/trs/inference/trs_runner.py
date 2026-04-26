from __future__ import annotations

from trs.domain.enums import RunMode
from trs.domain.schemas import InferenceRequest, InferenceResult, ProblemRecord, RetrievedSkill, SkillCard
from trs.evaluators.coding import evaluate_python_code
from trs.evaluators.math import evaluate_math_answer
from trs.inference.gating import gate_skills
from trs.models.base import ModelClient
from trs.models.pricing import estimate_cost
from trs.prompts.builders import build_trs_prompt
from trs.retrieval.base import Retriever
from trs.traces.normalizer import normalize_final_answer


class TRSInferenceRunner:
    def __init__(
        self,
        client: ModelClient,
        prompt_config: dict[str, str],
        retriever: Retriever,
        profile: dict[str, object],
    ):
        self.client = client
        self.prompt_config = prompt_config
        self.retriever = retriever
        self.profile = profile

    def run(self, problem: ProblemRecord, cards: list[SkillCard]) -> InferenceResult:
        request = InferenceRequest(
            task_type=problem.task_type,
            question=problem.question,
            dataset=problem.dataset,
            problem_id=problem.id,
            mode=RunMode.TRS,
            profile=str(self.profile.get("name", "")),
            tests=problem.tests,
            answer=problem.answer,
            entry_point=problem.entry_point,
            metadata=problem.metadata,
        )
        retrieved = self.retriever.retrieve(problem.question, cards, top_k=int(self.profile.get("top_k", 1)))
        accepted = gate_skills(
            retrieved,
            top_k=int(self.profile.get("top_k", 1)),
            score_threshold=float(self.profile.get("score_threshold", 0.0)),
            max_skill_chars=int(self.profile.get("max_skill_chars", 800)),
        )
        if not accepted:
            return self._fallback_direct(problem, request)
        system_prompt, user_prompt = build_trs_prompt(self.prompt_config, problem, accepted)
        response = self.client.complete(system_prompt, user_prompt, metadata=problem.metadata)
        final_answer = normalize_final_answer(problem.task_type, response.text)
        if problem.task_type.value == "math":
            is_correct = evaluate_math_answer(final_answer, problem.answer)
        else:
            is_correct, _ = evaluate_python_code(final_answer, problem.tests)
        return InferenceResult(
            request=request,
            response_text=response.text,
            final_answer=final_answer,
            usage=response.usage,
            used_skills=accepted,
            is_correct=is_correct,
            model_name=self.client.name,
            fallback_used=False,
            cost_estimate=estimate_cost(response.usage, self.client.config),
            metadata={"provider_metadata": response.metadata},
        )

    def _fallback_direct(self, problem: ProblemRecord, request: InferenceRequest) -> InferenceResult:
        system_prompt, user_prompt = build_trs_prompt(self.prompt_config, problem, [])
        response = self.client.complete(system_prompt, user_prompt, metadata=problem.metadata)
        final_answer = normalize_final_answer(problem.task_type, response.text)
        if problem.task_type.value == "math":
            is_correct = evaluate_math_answer(final_answer, problem.answer)
        else:
            is_correct, _ = evaluate_python_code(final_answer, problem.tests)
        result = InferenceResult(
            request=request,
            response_text=response.text,
            final_answer=final_answer,
            usage=response.usage,
            used_skills=[],
            is_correct=is_correct,
            model_name=self.client.name,
            fallback_used=True,
            cost_estimate=estimate_cost(response.usage, self.client.config),
            metadata={"provider_metadata": response.metadata},
        )
        result.fallback_used = True
        return result
