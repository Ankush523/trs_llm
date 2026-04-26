from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from typing import Any

from trs.models.base import ModelClient, ModelConfig, ModelResponse
from trs.models.usage import make_usage


class OpenAICompatibleClient(ModelClient):
    def complete(self, system_prompt: str, user_prompt: str, metadata: dict[str, Any] | None = None) -> ModelResponse:
        metadata = metadata or {}
        if self.config.provider == "mock" or self.config.base_url.startswith("mock://"):
            return self._complete_mock(system_prompt, user_prompt, metadata)
        return self._complete_http(system_prompt, user_prompt, metadata)

    def _complete_mock(self, system_prompt: str, user_prompt: str, metadata: dict[str, Any]) -> ModelResponse:
        start = time.time()
        prompt_bundle = f"{system_prompt}\n{user_prompt}"
        mock_output = metadata.get("mock_response")
        if mock_output is None:
            if "Retrieved Reasoning Skill(s):" in user_prompt:
                mock_output = metadata.get("mock_trs_response", metadata.get("mock_direct_response"))
            elif "Return a JSON object with fields trigger" in user_prompt:
                mock_output = json.dumps(metadata.get("mock_skill_card", {}))
            else:
                mock_output = metadata.get("mock_direct_response", "")
        text = str(mock_output or "")
        usage = make_usage(prompt_bundle, text, latency_ms=int((time.time() - start) * 1000))
        return ModelResponse(text=text, usage=usage, metadata={"provider": "mock"})

    def _complete_http(self, system_prompt: str, user_prompt: str, metadata: dict[str, Any]) -> ModelResponse:
        start = time.time()
        api_key = os.environ.get(self.config.api_key_env, "")
        if not api_key:
            raise RuntimeError(f"Missing API key in env var: {self.config.api_key_env}")
        base_url = self.config.base_url.rstrip("/")
        url = f"{base_url}/chat/completions"
        payload = {
            "model": self.config.name,
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        }
        request = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=120) as response:
                body = json.loads(response.read().decode("utf-8"))
        except urllib.error.URLError as exc:  # pragma: no cover - network path
            raise RuntimeError(f"OpenAI-compatible request failed: {exc}") from exc
        text = str(body["choices"][0]["message"]["content"])
        usage_body = body.get("usage", {})
        usage = make_usage(f"{system_prompt}\n{user_prompt}", text, latency_ms=int((time.time() - start) * 1000))
        usage.input_tokens = int(usage_body.get("prompt_tokens", usage.input_tokens))
        usage.output_tokens = int(usage_body.get("completion_tokens", usage.output_tokens))
        usage.reasoning_tokens = int(usage_body.get("reasoning_tokens", usage.output_tokens))
        return ModelResponse(text=text, usage=usage, metadata={"provider": "openai-compatible", "raw_usage": usage_body})


def build_model_client(config: ModelConfig) -> ModelClient:
    return OpenAICompatibleClient(config)
