from __future__ import annotations

from pathlib import Path

import yaml
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.messages import TextMessage
from autogen_core import CancellationToken
from autogen_core.models import ChatCompletionClient
from autogen_ext.models.openai import OpenAIChatCompletionClient
from my_agent.model_config import OpenAIConfig


_MODEL_CONFIG_PATH = Path(__file__).with_name("model_config.yml")


def _create_model_client() -> ChatCompletionClient:
    """Create the chat completion client used by the Streamlit agent."""
    if _MODEL_CONFIG_PATH.exists():
        with _MODEL_CONFIG_PATH.open("r", encoding="utf-8") as config_file:
            model_config = yaml.safe_load(config_file)
        return ChatCompletionClient.load_component(model_config)

    config = OpenAIConfig.from_env()
    return OpenAIChatCompletionClient(
        model=config.model,
        base_url=config.base_url,
        api_key=config.api_key,
        request_timeout=config.request_timeout,
        max_retries=config.max_retries,
        model_info=config.model_info,
    )


class Agent:
    def __init__(self) -> None:
        model_client = _create_model_client()
        self.agent = AssistantAgent(
            name="assistant",
            model_client=model_client,
            system_message="You are a helpful AI assistant.",
        )

    async def chat(self, prompt: str) -> str:
        response = await self.agent.on_messages(
            [TextMessage(content=prompt, source="user")],
            CancellationToken(),
        )
        assert isinstance(response.chat_message, TextMessage)
        return response.chat_message.content
