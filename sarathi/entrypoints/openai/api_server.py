import asyncio
from http import HTTPStatus
from typing import Optional

import fastapi
import uvicorn
from fastapi import Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, StreamingResponse

from sarathi.engine.async_llm_engine import AsyncLLMEngine
from sarathi.entrypoints.openai.config import OpenAIServerConfig
from sarathi.entrypoints.openai.protocol import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    CompletionRequest,
    ErrorResponse,
)
from sarathi.entrypoints.openai.serving_chat import OpenAIServingChat
from sarathi.entrypoints.openai.serving_completion import OpenAIServingCompletion
from sarathi.logger import init_logger

TIMEOUT_KEEP_ALIVE = 5  # seconds

openai_serving_chat: OpenAIServingChat
openai_serving_completion: OpenAIServingCompletion

logger = init_logger(__name__)


app = fastapi.FastAPI()


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(_, exc):
    err = openai_serving_chat.create_error_response(message=str(exc))
    return JSONResponse(err.model_dump(), status_code=HTTPStatus.BAD_REQUEST)


@app.get("/v1/models")
async def show_available_models():
    models = await openai_serving_chat.show_available_models()
    return JSONResponse(content=models.model_dump())


@app.post("/v1/chat/completions")
async def create_chat_completion(request: ChatCompletionRequest, raw_request: Request):
    generator = await openai_serving_chat.create_chat_completion(request, raw_request)
    if isinstance(generator, ErrorResponse):
        return JSONResponse(content=generator.model_dump(), status_code=generator.code)
    if request.stream:
        return StreamingResponse(content=generator, media_type="text/event-stream")
    else:
        assert isinstance(generator, ChatCompletionResponse)
        return JSONResponse(content=generator.model_dump())


@app.post("/v1/completions")
async def create_completion(request: CompletionRequest, raw_request: Request):
    generator = await openai_serving_completion.create_completion(request, raw_request)
    if isinstance(generator, ErrorResponse):
        return JSONResponse(content=generator.model_dump(), status_code=generator.code)
    if request.stream:
        return StreamingResponse(content=generator, media_type="text/event-stream")
    else:
        return JSONResponse(content=generator.model_dump())


if __name__ == "__main__":
    config = OpenAIServerConfig.create_from_cli_args()

    if config.api_key:

        @app.middleware("http")
        async def authentication(request: Request, call_next):
            root_path = (
                "" if config.server_root_path is None else config.server_root_path
            )
            if request.method == "OPTIONS":
                return await call_next(request)
            if not request.url.path.startswith(f"{root_path}/v1"):
                return await call_next(request)
            if request.headers.get("Authorization") != "Bearer " + config.api_key:
                return JSONResponse(content={"error": "Unauthorized"}, status_code=401)
            return await call_next(request)

    logger.info(f"Launching OpenAI compatible server with config: {config}")

    served_model_names = [config.model_config.model]

    engine = AsyncLLMEngine.from_system_config(
        config.create_system_config(), verbose=(config.log_level == "debug")
    )

    event_loop: Optional[asyncio.AbstractEventLoop]
    try:
        event_loop = asyncio.get_running_loop()
    except RuntimeError:
        event_loop = None

    if event_loop is not None and event_loop.is_running():
        # If the current is instanced by Ray Serve,
        # there is already a running event loop
        model_config = event_loop.run_until_complete(engine.get_model_config())
    else:
        # When using single Sarathi instance
        model_config = asyncio.run(engine.get_model_config())

    openai_serving_chat = OpenAIServingChat(
        engine,
        model_config,
        served_model_names,
        config.response_role,
        config.chat_template,
    )
    openai_serving_completion = OpenAIServingCompletion(
        engine, model_config, served_model_names
    )

    app.root_path = config.server_root_path
    uvicorn.run(
        app,
        host=config.host,
        port=config.port,
        log_level=config.log_level,
        ssl_keyfile=config.ssl_keyfile,
        ssl_certfile=config.ssl_certfile,
        ssl_ca_certs=config.ssl_ca_certs,
        ssl_cert_reqs=config.ssl_cert_reqs,
        timeout_keep_alive=TIMEOUT_KEEP_ALIVE,
    )
