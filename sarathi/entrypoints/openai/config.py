from dataclasses import dataclass, field
from typing import List, Optional

from sarathi.entrypoints.config import APIServerConfig


@dataclass
class OpenAIServerConfig(APIServerConfig):
    api_key: Optional[str] = field(
        default=None, metadata={"help": "API key for authentication with the server."}
    )
    chat_template: Optional[str] = field(
        default=None, metadata={"help": "Template for formatting chat messages."}
    )
    response_role: str = field(
        default="assistant",
        metadata={
            "help": "Role to be assigned to the model's responses in the chat format."
        },
    )
