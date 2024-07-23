import json
from dataclasses import dataclass
from http import HTTPStatus
from typing import List, Optional, Union

from sarathi.config import ModelConfig
from sarathi.engine.async_llm_engine import AsyncLLMEngine
from sarathi.entrypoints.openai.protocol import (
    ChatCompletionRequest,
    CompletionRequest,
    ErrorResponse,
    ModelCard,
    ModelList,
    ModelPermission,
)
from sarathi.logger import init_logger
from sarathi.transformers_utils.tokenizer import get_tokenizer

logger = init_logger(__name__)


@dataclass
class LoRAModulePath:
    name: str
    local_path: str


class OpenAIServing:

    def __init__(
        self,
        engine: AsyncLLMEngine,
        model_config: ModelConfig,
        served_model_names: List[str],
    ):
        super().__init__()

        self.engine = engine
        self.max_model_len = model_config.max_model_len

        # A separate tokenizer to map token IDs to strings.
        self.tokenizer = get_tokenizer(
            model_config.model,
            tokenizer_revision=model_config.revision,
            trust_remote_code=model_config.trust_remote_code,
            truncation_side="left",
        )

        self.served_model_names = served_model_names

    async def show_available_models(self) -> ModelList:
        """Show available models. Right now we only have one model."""
        model_cards = [
            ModelCard(
                id=served_model_name,
                max_model_len=self.max_model_len,
                root=self.served_model_names[0],
                permission=[ModelPermission()],
            )
            for served_model_name in self.served_model_names
        ]
        return ModelList(data=model_cards)

    def create_error_response(
        self,
        message: str,
        err_type: str = "BadRequestError",
        status_code: HTTPStatus = HTTPStatus.BAD_REQUEST,
    ) -> ErrorResponse:
        return ErrorResponse(message=message, type=err_type, code=status_code.value)

    def create_streaming_error_response(
        self,
        message: str,
        err_type: str = "BadRequestError",
        status_code: HTTPStatus = HTTPStatus.BAD_REQUEST,
    ) -> str:
        json_str = json.dumps(
            {
                "error": self.create_error_response(
                    message=message, err_type=err_type, status_code=status_code
                ).model_dump()
            }
        )
        return json_str

    async def _check_model(
        self, request: Union[CompletionRequest, ChatCompletionRequest]
    ) -> Optional[ErrorResponse]:
        if request.model in self.served_model_names:
            return None
        return self.create_error_response(
            message=f"The model `{request.model}` does not exist.",
            err_type="NotFoundError",
            status_code=HTTPStatus.NOT_FOUND,
        )
