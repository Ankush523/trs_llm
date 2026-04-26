from __future__ import annotations

from trs.domain.schemas import RetrievedSkill, SkillCard
from trs.retrieval.base import Retriever
from trs.retrieval.bm25 import BM25Retriever
from trs.retrieval.dense import DenseRetriever


class HybridRetriever(Retriever):
    def __init__(self, bm25_weight: float = 0.7, dense_weight: float = 0.3):
        self.bm25 = BM25Retriever()
        self.dense = DenseRetriever()
        self.bm25_weight = bm25_weight
        self.dense_weight = dense_weight

    def retrieve(self, query: str, cards: list[SkillCard], top_k: int = 1) -> list[RetrievedSkill]:
        bm25_hits = {hit.skill_id: hit for hit in self.bm25.retrieve(query, cards, top_k=max(top_k * 3, top_k))}
        dense_hits = {hit.skill_id: hit for hit in self.dense.retrieve(query, cards, top_k=max(top_k * 3, top_k))}
        merged: list[RetrievedSkill] = []
        for card in cards:
            left = bm25_hits.get(card.id)
            right = dense_hits.get(card.id)
            if not left and not right:
                continue
            score = 0.0
            text = ""
            trigger = card.trigger
            heuristic = card.heuristic
            if left:
                score += left.score * self.bm25_weight
                text = left.text
            if right:
                score += right.score * self.dense_weight
                text = right.text
            merged.append(
                RetrievedSkill(
                    skill_id=card.id,
                    score=score,
                    trigger=trigger,
                    heuristic=heuristic,
                    text=text,
                    metadata={"retriever": "hybrid"},
                )
            )
        merged.sort(key=lambda item: item.score, reverse=True)
        return [hit for hit in merged[:top_k] if hit.score > 0]
