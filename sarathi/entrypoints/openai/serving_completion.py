import time
from typing import AsyncGenerator, AsyncIterator, List, Optional, Tuple

from fastapi import Request

from sarathi.config import ModelConfig
from sarathi.core.datatypes.request_output import RequestOutput
from sarathi.engine.async_llm_engine import AsyncLLMEngine

# yapf conflicts with isort for this block
# yapf: disable
from sarathi.entrypoints.openai.protocol import (
    CompletionRequest,
    CompletionResponse,
    CompletionResponseChoice,
    CompletionResponseStreamChoice,
    CompletionStreamResponse,
    UsageInfo,
)
from sarathi.entrypoints.openai.serving_engine import OpenAIServing
from sarathi.logger import init_logger
from sarathi.utils import merge_async_iterators, random_uuid

logger = init_logger(__name__)


def parse_prompt_format(prompt) -> Tuple[bool, list]:
    # get the prompt, openai supports the following
    # "a string, array of strings, array of tokens, or array of token arrays."
    prompt_is_tokens = False
    prompts = [prompt]  # case 1: a string
    if isinstance(prompt, list):
        if len(prompt) == 0:
            raise ValueError("please provide at least one prompt")
        elif isinstance(prompt[0], str):
            prompt_is_tokens = False
            prompts = prompt  # case 2: array of strings
        elif isinstance(prompt[0], int):
            prompt_is_tokens = True
            prompts = [prompt]  # case 3: array of tokens
        elif isinstance(prompt[0], list) and isinstance(prompt[0][0], int):
            prompt_is_tokens = True
            prompts = prompt  # case 4: array of token arrays
        else:
            raise ValueError("prompt must be a string, array of strings, "
                             "array of tokens, or array of token arrays")
    return prompt_is_tokens, prompts


class OpenAIServingCompletion(OpenAIServing):

    def __init__(self, engine: AsyncLLMEngine, model_config: ModelConfig,
                 served_model_names: List[str],):
        super().__init__(engine=engine,
                         model_config=model_config,
                         served_model_names=served_model_names,)

    async def create_completion(self, request: CompletionRequest,
                                raw_request: Request):
        """Completion API similar to OpenAI's API.

        See https://platform.openai.com/docs/api-reference/completions/create
        for the API specification. This API mimics the OpenAI Completion API.

        NOTE: Currently we do not support the following feature:
            - suffix (the language models we currently support do not support
            suffix)
        """
        error_check_ret = await self._check_model(request)
        if error_check_ret is not None:
            return error_check_ret

        # Return error for unsupported features.
        if request.suffix is not None:
            return self.create_error_response(
                "suffix is not currently supported")

        model_name = self.served_model_names[0]
        request_id = f"cmpl-{random_uuid()}"
        created_time = int(time.time())

        # Schedule the request and get the result generator.
        generators: List[AsyncIterator[RequestOutput]] = []
        try:
            sampling_params = request.to_sampling_params()
            prompt_is_tokens, prompts = parse_prompt_format(request.prompt)

            for i, prompt in enumerate(prompts):
                if prompt_is_tokens:
                    raise ValueError(
                        "array of tokens, or array of token arrays not supported")

                generator = self.engine.generate(
                    f"{request_id}-{i}",
                    prompt,
                    sampling_params,
                )

                generators.append(generator)
        except ValueError as e:
            # TODO: Use a sarathi-specific Validation Error
            return self.create_error_response(str(e))

        result_generator: AsyncIterator[Tuple[
            int, RequestOutput]] = merge_async_iterators(*generators)

        # Streaming response
        if request.stream:
            return self.completion_stream_generator(request,
                                                    raw_request,
                                                    result_generator,
                                                    request_id,
                                                    created_time,
                                                    model_name,
                                                    num_prompts=len(prompts))

        # Non-streaming response
        final_res_batch: List[Optional[RequestOutput]] = [None] * len(prompts)
        try:
            async for i, res in result_generator:
                if await raw_request.is_disconnected():
                    # Abort the request if the client disconnects.
                    await self.engine.abort(f"{request_id}-{i}")
                    return self.create_error_response("Client disconnected")
                final_res_batch[i] = res
            response = self.request_output_to_completion_response(
                final_res_batch, request, request_id, created_time, model_name)
        except ValueError as e:
            # TODO: Use a sarathi-specific Validation Error
            return self.create_error_response(str(e))

        # When user requests streaming but we don't stream, we still need to
        # return a streaming response with a single event.
        if request.stream:
            response_json = response.model_dump_json()

            async def fake_stream_generator() -> AsyncGenerator[str, None]:
                yield f"data: {response_json}\n\n"
                yield "data: [DONE]\n\n"

            return fake_stream_generator()

        return response

    async def completion_stream_generator(
        self,
        request: CompletionRequest,
        raw_request: Request,
        result_generator: AsyncIterator[Tuple[int, RequestOutput]],
        request_id: str,
        created_time: int,
        model_name: str,
        num_prompts: int,
    ) -> AsyncGenerator[str, None]:
        previous_texts = [""] * num_prompts
        previous_num_tokens = [0] * num_prompts
        has_echoed = [False] * num_prompts

        try:
            async for i, res in result_generator:

                # Abort the request if the client disconnects.
                if await raw_request.is_disconnected():
                    await self.engine.abort(f"{request_id}-{i}")
                    raise StopAsyncIteration()

                # TODO(simon): optimize the performance by avoiding full
                # text O(n^2) sending.
                assert request.max_tokens is not None
                if request.echo and request.max_tokens == 0:
                    # only return the prompt
                    delta_text = res.prompt
                    delta_token_ids = res.prompt_token_ids
                    has_echoed[i] = True
                elif (request.echo and request.max_tokens > 0
                        and not has_echoed[i]):
                    # echo the prompt and first token
                    delta_text = res.prompt + res.text
                    has_echoed[i] = True
                else:
                    # return just the delta
                    delta_text = res.text[len(previous_texts[i]):]

                previous_texts[i] = res.text
                previous_num_tokens[i] = len(res.token_ids)
                finish_reason = res.finish_reason

                if res.finish_reason is not None:  # return final usage
                    prompt_tokens = len(res.prompt_token_ids)
                    completion_tokens = len(res.token_ids)
                    final_usage = UsageInfo(
                        prompt_tokens=prompt_tokens,
                        completion_tokens=completion_tokens,
                        total_tokens=prompt_tokens + completion_tokens,
                    )
                else:
                    final_usage = None

                chunk = CompletionStreamResponse(
                    id=request_id,
                    created=created_time,
                    model=model_name,
                    choices=[
                        CompletionResponseStreamChoice(
                            index=i,
                            text=delta_text,
                            finish_reason=finish_reason,
                        )
                    ])
                if (request.stream_options
                        and request.stream_options.include_usage):
                    chunk.usage = None

                response_json = chunk.model_dump_json(exclude_unset=True)
                yield f"data: {response_json}\n\n"

            if (request.stream_options
                    and request.stream_options.include_usage):
                final_usage_chunk = CompletionStreamResponse(
                    id=request_id,
                    created=created_time,
                    model=model_name,
                    choices=[],
                    usage=final_usage,
                )
                final_usage_data = (final_usage_chunk.model_dump_json(
                    exclude_unset=True, exclude_none=True))
                yield f"data: {final_usage_data}\n\n"

        except ValueError as e:
            # TODO: Use a sarathi-specific Validation Error
            data = self.create_streaming_error_response(str(e))
            yield f"data: {data}\n\n"
        yield "data: [DONE]\n\n"

    def request_output_to_completion_response(
        self,
        final_res_batch: List[RequestOutput],
        request: CompletionRequest,
        request_id: str,
        created_time: int,
        model_name: str,
    ) -> CompletionResponse:
        choices: List[CompletionResponseChoice] = []
        num_prompt_tokens = 0
        num_generated_tokens = 0
        for final_res in final_res_batch:
            assert final_res is not None
            prompt_token_ids = final_res.prompt_token_ids
            prompt_text = final_res.prompt

            assert request.max_tokens is not None
            if request.echo and request.max_tokens == 0:
                output_text = prompt_text
            elif request.echo and request.max_tokens > 0:
                output_text = prompt_text + final_res.text
            else:
                output_text = final_res.text

            choice_data = CompletionResponseChoice(
                index=len(choices),
                text=output_text,
                finish_reason=final_res.finish_reason,
            )
            choices.append(choice_data)

            num_prompt_tokens += len(prompt_token_ids)
            num_generated_tokens += len(final_res.token_ids)

        usage = UsageInfo(
            prompt_tokens=num_prompt_tokens,
            completion_tokens=num_generated_tokens,
            total_tokens=num_prompt_tokens + num_generated_tokens,
        )

        return CompletionResponse(
            id=request_id,
            created=created_time,
            model=model_name,
            choices=choices,
            usage=usage,
        )
