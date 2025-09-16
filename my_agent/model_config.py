from __future__ import annotations

import os
from dataclasses import dataclass, field
from autogen_core.models import ModelInfo


def _get_int_env(name: str, default: int) -> int:
    value = os.environ.get(name)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError:
        return default


@dataclass(frozen=True)
class OpenAIConfig:
    model: str = "Qwen3_Coder_480B_A35B_Instruct_FP8"
    base_url: str = "https://ur-llm-proxy.wb.ru/v1"
    api_key: str = "sk-tSKxWGj74Hjr-5LxW7BQYg"
    request_timeout: int = 60
    max_retries: int = 3
    model_info: ModelInfo = field(
        default_factory=lambda: ModelInfo(
            vision=False,
            function_calling=True,
            json_output=True,
            family="qwen",
            structured_output=True,
        )
    )

    @classmethod
    def from_env(cls) -> "OpenAIConfig":
        return cls(
            model=os.environ.get("CODE_MODEL", cls.model),
            base_url=os.environ.get("CODE_BASE_URL", cls.base_url),
            api_key=os.environ.get("CODE_API_KEY", cls.api_key),
            request_timeout=_get_int_env("CODE_REQUEST_TIMEOUT", cls.request_timeout),
            max_retries=_get_int_env("CODE_MAX_RETRIES", cls.max_retries),
        )
