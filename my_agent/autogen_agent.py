"""Simple Codex-like agent built with AutoGen.

This module recreates the lightweight coding agent from the Codex CLI
using the AutoGen 0.2 library.  It wires together three agents:

* **Test Writing Assistant** – writes Python tests for the requested
  functionality.
* **Verification Assistant** – runs the written tests and verifies their
  correctness.
* **Summary Agent** – presents the final summary to the user once the
  work is complete.

The configuration mirrors ``team-config.json`` in this directory but is
constructed programmatically so the agent can be executed without a UI.

Example
-------
Run the team on a simple request against the current repository:

``python my_agent/autogen_agent.py "Write tests for a fibonacci function" .``"""

from __future__ import annotations

import argparse
import asyncio
import os
from pathlib import Path

from autogen_core import CancellationToken
from autogen_core.tools import StaticWorkbench
from autogen_agentchat.agents import AssistantAgent, UserProxyAgent
from autogen_agentchat.conditions import TextMentionTermination
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_agentchat.ui import Console
from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_ext.tools.code_execution import PythonCodeExecutionTool
from autogen_ext.code_executors.local import LocalCommandLineCodeExecutor

# ---------------------------------------------------------------------------
# System messages are copied from ``team-config.json`` so behaviour mirrors
# the configuration used for the notebook prototype.
# ---------------------------------------------------------------------------

TEST_WRITER_SYSTEM_MESSAGE = (
    "You are an expert Python developer with deep knowledge of software testing, "
    "including unit tests, integration tests, and best practices for test-driven "
    "development (TDD). Your task is to analyze Python code and generate high-quality, "
    "maintainable, and efficient test cases. Instructions:  1. Understand the Code:    "
    "- Carefully review the provided Python code, including functions, classes, and "
    "dependencies.    - Identify edge cases, expected behavior, and potential failure "
    "points.  2. Choose a Testing Framework:     - Default to `pytest` if no framework "
    "is specified.     - Ensure compatibility with the code’s dependencies (e.g., `mock` "
    "for external services).  3. Generate Tests:  - Write tests that cover:       - Happy "
    "Path (normal operation).       - Edge Cases (empty inputs, invalid types, boundary "
    "conditions).       - Error Handling (exceptions, invalid states).     - Include "
    "descriptive docstrings or comments explaining each test’s purpose.     - Use "
    "fixtures (`pytest`) or `setUp/tearDown` (`unittest`) where needed.  4. Output Format: "
    "    - Return the test code in a complete, executable format.     - If the original "
    "code has bugs, note them and write tests to catch them.  When done, say TERMINATE."
)

VERIFICATION_SYSTEM_MESSAGE = (
    "You are a task verification assistant who is working with a test writer agent to "
    "solve tasks. At each point, check if the task has been completed as requested by the "
    "user. If the test_assistant_agent responds and the task has not yet been completed, "
    "respond with what is left to do and then say 'keep going'. If and only when the task "
    "has been completed, summarize and present a final answer that directly addresses the "
    "user task in detail and then respond with TERMINATE."
)

# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def _create_model_client() -> OpenAIChatCompletionClient:
    """Create the model client used by all agents.

    The model name, base URL and API key can be provided via environment
    variables ``OPENAI_MODEL``, ``OPENAI_BASE_URL`` and ``OPENAI_API_KEY``.
    Defaults match the placeholder values in ``team-config.json``.
    """

    model = os.environ.get("OPENAI_MODEL", "DeepSeek-R1")
    base_url = os.environ.get("OPENAI_BASE_URL", "https://corellm.wb.ru/deepseek/v1")
    api_key = os.environ.get("OPENAI_API_KEY", "")
    return OpenAIChatCompletionClient(model=model, base_url=base_url, api_key=api_key, temperature=0.15)


def build_team(code_dir: Path) -> RoundRobinGroupChat:
    """Construct the three-agent team mimicking the Codex arrangement.

    Parameters
    ----------
    code_dir:
        Directory containing the code under test.  All file operations and
        execution performed by the agents occur relative to this directory.
    """

    model_client = _create_model_client()

    executor = LocalCommandLineCodeExecutor(
        timeout=360,
        work_dir=str(code_dir),
        functions_module="functions",
    )
    python_tool_writer = PythonCodeExecutionTool(
        executor=executor,
        description="Execute Python code blocks and edit files.",
        name="Python_Code_Execution_Tool",
    )
    python_tool_verifier = PythonCodeExecutionTool(
        executor=executor,
        description="Execute Python code blocks and edit files.",
        name="Python_Code_Execution_Tool",
    )

    test_writer = AssistantAgent(
        "test_writing_assistant",
        model_client=model_client,
        system_message=TEST_WRITER_SYSTEM_MESSAGE,
        workbench=StaticWorkbench(tools=[python_tool_writer]),
    )

    verifier = AssistantAgent(
        "verification_assistant",
        model_client=model_client,
        system_message=VERIFICATION_SYSTEM_MESSAGE,
        workbench=StaticWorkbench(tools=[python_tool_verifier]),
        reflect_on_tool_use=True,
    )

    summary_agent = UserProxyAgent(
        "summary_agent",
        description=(
            "a human user that should be consulted only when the verification_assistant "
            "is unable to verify the information provided by the test_writing_assistant"
        ),
    )

    termination = TextMentionTermination("TERMINATE")
    team = RoundRobinGroupChat(
        [test_writer, verifier, summary_agent],
        termination_condition=termination,
        allow_repeated_speaker=True,
    )
    return team


async def run(task: str, code_dir: Path) -> None:
    """Run the agent team on the given task and stream output to the console."""

    team = build_team(code_dir)
    await team.reset()
    await Console(
        team.run_stream(task=task, cancellation_token=CancellationToken())
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a Codex-like AutoGen agent")
    parser.add_argument("task", help="Task for the agents to solve")
    parser.add_argument(
        "code_dir",
        type=Path,
        help="Path to the code repository the agents should modify and execute",
    )
    args = parser.parse_args()
    asyncio.run(run(args.task, args.code_dir))


if __name__ == "__main__":
    main()
