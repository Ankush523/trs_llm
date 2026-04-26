from __future__ import annotations

from trs.retrieval.base import Retriever
from trs.retrieval.bm25 import BM25Retriever
from trs.retrieval.dense import DenseRetriever
from trs.retrieval.hybrid import HybridRetriever


def build_retriever(profile: dict[str, object]) -> Retriever:
    retriever_type = str(profile.get("retriever_type", "bm25"))
    if retriever_type == "bm25":
        return BM25Retriever()
    if retriever_type == "dense":
        return DenseRetriever()
    if retriever_type == "hybrid":
        weights = dict(profile.get("hybrid_weights", {}))
        return HybridRetriever(
            bm25_weight=float(weights.get("bm25", 0.7)),
            dense_weight=float(weights.get("dense", 0.3)),
        )
    raise ValueError(f"Unsupported retriever type: {retriever_type}")
