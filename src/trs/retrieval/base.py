from __future__ import annotations

from abc import ABC, abstractmethod

from trs.domain.schemas import RetrievedSkill, SkillCard


class Retriever(ABC):
    @abstractmethod
    def retrieve(self, query: str, cards: list[SkillCard], top_k: int = 1) -> list[RetrievedSkill]:
        raise NotImplementedError
