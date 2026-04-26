from __future__ import annotations

import math
from collections import Counter

from trs.domain.schemas import RetrievedSkill, SkillCard
from trs.prompts.builders import render_skill_text
from trs.retrieval.base import Retriever
from trs.retrieval.bm25 import tokenize


def cosine_similarity(left: Counter[str], right: Counter[str]) -> float:
    if not left or not right:
        return 0.0
    overlap = set(left).intersection(right)
    numerator = sum(left[token] * right[token] for token in overlap)
    left_norm = math.sqrt(sum(value * value for value in left.values()))
    right_norm = math.sqrt(sum(value * value for value in right.values()))
    if left_norm == 0 or right_norm == 0:
        return 0.0
    return numerator / (left_norm * right_norm)


class DenseRetriever(Retriever):
    def retrieve(self, query: str, cards: list[SkillCard], top_k: int = 1) -> list[RetrievedSkill]:
        query_vec = Counter(tokenize(query))
        scored: list[tuple[float, SkillCard]] = []
        for card in cards:
            card_vec = Counter(tokenize(_card_text(card)))
            scored.append((cosine_similarity(query_vec, card_vec), card))
        scored.sort(key=lambda item: item[0], reverse=True)
        return [
            RetrievedSkill(
                skill_id=card.id,
                score=score,
                trigger=card.trigger,
                heuristic=card.heuristic,
                text=render_skill_text(card),
                metadata={"retriever": "dense"},
            )
            for score, card in scored[:top_k]
            if score > 0
        ]


def _card_text(card: SkillCard) -> str:
    return " ".join(
        [
            str(card.metadata.get("source_question", "")),
            card.trigger,
            card.heuristic,
            " ".join(card.keywords),
            " ".join(card.do),
            " ".join(card.avoid),
            " ".join(card.check),
            " ".join(card.risk),
        ]
    )
