from __future__ import annotations

import math
import re
from collections import Counter

from trs.domain.schemas import RetrievedSkill, SkillCard
from trs.prompts.builders import render_skill_text
from trs.retrieval.base import Retriever


TOKEN_RE = re.compile(r"[a-zA-Z0-9_]+")


def tokenize(text: str) -> list[str]:
    return TOKEN_RE.findall(text.lower())


class BM25Retriever(Retriever):
    def retrieve(self, query: str, cards: list[SkillCard], top_k: int = 1) -> list[RetrievedSkill]:
        docs = [tokenize(_card_text(card)) for card in cards]
        query_terms = tokenize(query)
        if not docs or not query_terms:
            return []
        avg_doc_len = sum(len(doc) for doc in docs) / len(docs)
        doc_freqs: Counter[str] = Counter()
        for doc in docs:
            doc_freqs.update(set(doc))
        scored: list[tuple[float, SkillCard]] = []
        for card, doc in zip(cards, docs, strict=True):
            term_counts = Counter(doc)
            score = 0.0
            for term in query_terms:
                df = doc_freqs.get(term, 0)
                if df == 0:
                    continue
                idf = math.log(1 + (len(docs) - df + 0.5) / (df + 0.5))
                tf = term_counts.get(term, 0)
                if tf == 0:
                    continue
                numerator = tf * (1.5 + 1)
                denominator = tf + 1.5 * (1 - 0.75 + 0.75 * (len(doc) / max(avg_doc_len, 1)))
                score += idf * (numerator / denominator)
            scored.append((score, card))
        scored.sort(key=lambda item: item[0], reverse=True)
        return [
            RetrievedSkill(
                skill_id=card.id,
                score=score,
                trigger=card.trigger,
                heuristic=card.heuristic,
                text=render_skill_text(card),
                metadata={"retriever": "bm25"},
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
