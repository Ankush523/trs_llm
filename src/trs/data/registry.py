from __future__ import annotations

from trs.data.adapters.base import DatasetAdapter
from trs.data.adapters.humaneval_plus import HumanEvalPlusAdapter
from trs.data.adapters.math_hendrycks import MathHendrycksAdapter
from trs.data.adapters.mbpp import MBPPAdapter


def get_dataset_adapter(name: str) -> DatasetAdapter:
    registry: dict[str, DatasetAdapter] = {
        MathHendrycksAdapter.name: MathHendrycksAdapter(),
        MBPPAdapter.name: MBPPAdapter(),
        HumanEvalPlusAdapter.name: HumanEvalPlusAdapter(),
    }
    try:
        return registry[name]
    except KeyError as exc:
        raise ValueError(f"Unsupported dataset adapter: {name}") from exc
