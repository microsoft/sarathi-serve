from dataclasses import dataclass, field
from typing import List

from sarathi.entrypoints.config import APIServerConfig


@dataclass
class OpenAIServerConfig(APIServerConfig):
    api_key: str = None
    chat_template: str = None
    response_role: str = "assistant"
