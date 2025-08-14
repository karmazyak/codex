import os
from autogen_core.models import ModelInfo

MODEL = os.environ.get("OPENAI_MODEL", "openai/Devstral")
BASE_URL = os.environ.get("OPENAI_BASE_URL", "tmp")
API_KEY = os.environ.get("OPENAI_API_KEY", "tmp")

MODEL_INFO = ModelInfo(
    vision=False,
    function_calling=False,
    json_output=False,
    family="mistral",
    structured_output=True,
)
