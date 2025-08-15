from __future__ import annotations

import os
from dataclasses import dataclass, field
from autogen_core.models import ModelInfo


@dataclass(frozen=True)
class OpenAIConfig:
    model: str = "openai/Devstral"
    base_url: str = "tmp"
    api_key: str = "tmp"
    model_info: ModelInfo = field(
        default_factory=lambda: ModelInfo(
            vision=False,
            function_calling=True,
            json_output=False,
            family="mistral",
            structured_output=True,
        )
    )

    @classmethod
    def from_env(cls) -> "OpenAIConfig":
        return cls(
            model=os.environ.get("CODE_MODEL", cls.model),
            base_url=os.environ.get("CODE_BASE_URL", cls.base_url),
            api_key=os.environ.get("CODE_API_KEY", cls.api_key),
        )