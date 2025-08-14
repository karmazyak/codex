# Custom AutoGen Agent

This folder contains a minimal agent built on top of [AutoGen](https://github.com/microsoft/autogen) that mimics the behaviour of the Codex CLI.

## Quick start

```bash
python -m py_compile my_agent/autogen_agent.py my_agent/sample_project/math_utils.py
# Command prints nothing when the files compile successfully.

python my_agent/autogen_agent.py "Write tests for a fibonacci function" my_agent/sample_project --log-file my_agent/conversation.log
```

The `--log-file` option stores the task history so that subsequent runs can take
previous context into account. After each run a `git diff` of the target
repository is printed so you can easily review the changes.

## Sample project

```
python my_agent/sample_project/math_utils.py 5
python -m pytest my_agent/sample_project/test_math_utils.py
```
