from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from trs.domain.schemas import UsageStats


@dataclass(slots=True)
class ModelConfig:
    provider: str
    name: str
    base_url: str
    api_key_env: str = "OPENAI_API_KEY"
    temperature: float = 0.0
    max_tokens: int = 1200
    pricing: dict[str, float] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ModelConfig":
        return cls(
            provider=str(data["provider"]),
            name=str(data["name"]),
            base_url=str(data["base_url"]),
            api_key_env=str(data.get("api_key_env", "OPENAI_API_KEY")),
            temperature=float(data.get("temperature", 0.0)),
            max_tokens=int(data.get("max_tokens", 1200)),
            pricing=dict(data.get("pricing", {})),
        )


@dataclass(slots=True)
class ModelResponse:
    text: str
    usage: UsageStats
    metadata: dict[str, Any] = field(default_factory=dict)


class ModelClient(ABC):
    def __init__(self, config: ModelConfig):
        self.config = config

    @property
    def name(self) -> str:
        return self.config.name

    @abstractmethod
    def complete(self, system_prompt: str, user_prompt: str, metadata: dict[str, Any] | None = None) -> ModelResponse:
        raise NotImplementedError
