import asyncio
from functools import partial
from typing import (
    AsyncIterator,
    Callable,
    Dict,
    Iterable,
    List,
    Optional,
    Set,
    Tuple,
    Union,
)

from sarathi.config import ModelConfig, SystemConfig
from sarathi.core.datatypes.request_output import RequestOutput
from sarathi.core.datatypes.sampling_params import SamplingParams
from sarathi.engine.llm_engine import LLMEngine
from sarathi.logger import init_logger

logger = init_logger(__name__)

ENGINE_ITERATION_TIMEOUT_S = 60
MAX_PROMPT_LOG_LEN = 100


class AsyncStream:
    """A stream of RequestOutputs or EmbeddingRequestOutputs for a request
    that can be iterated over asynchronously."""

    def __init__(self, request_id: str) -> None:
        self.request_id = request_id
        self._queue: asyncio.Queue = asyncio.Queue()
        self._finished = False

    def put(self, item: Union[RequestOutput, Exception]) -> None:
        if self._finished:
            return

        self._queue.put_nowait(item)

    def finish(self) -> None:
        self._queue.put_nowait(StopAsyncIteration())
        self._finished = True

    @property
    def finished(self) -> bool:
        return self._finished

    def __aiter__(self):
        return self

    async def __anext__(self) -> Union[RequestOutput, Exception]:
        result = await self._queue.get()
        if isinstance(result, Exception):
            raise result
        return result


class RequestTracker:
    """Synchronous abstraction for tracking requests."""

    def __init__(self) -> None:
        self._request_streams: Dict[str, AsyncStream] = {}
        self._finished_requests: asyncio.Queue[str] = asyncio.Queue()
        self._new_requests: asyncio.Queue[Tuple[AsyncStream, dict]] = asyncio.Queue()
        self.new_requests_event = asyncio.Event()

    def __contains__(self, item):
        return item in self._request_streams

    def __len__(self) -> int:
        return len(self._request_streams)

    def propagate_exception(
        self, exc: Exception, request_id: Optional[str] = None
    ) -> None:
        """Propagate an exception to request streams
        (all if request_id is None)."""
        if request_id is not None:
            self._request_streams[request_id].put(exc)
            self.abort_request(request_id)
        else:
            for rid, stream in self._request_streams.items():
                stream.put(exc)
                self.abort_request(rid)

    def process_request_output(
        self, request_output: RequestOutput, *, verbose: bool = False
    ) -> None:
        """Process a request output from the engine."""
        request_id = request_output.seq_id

        if request_id not in self._request_streams:
            # aborted request
            return

        self._request_streams[request_id].put(request_output)
        if request_output.finished:
            if verbose:
                logger.info(f"Finished request {request_id}.")
            self.abort_request(request_id)

    def process_exception(
        self, request_id: str, exception: Exception, *, verbose: bool = False
    ) -> None:
        """Propagate an exception from the engine."""
        self._request_streams[request_id].put(exception)
        if verbose:
            logger.info(f"Finished request {request_id}.")
        self.abort_request(request_id)

    def add_request(self, request_id: str, **engine_add_request_kwargs) -> AsyncStream:
        """Add a request to be sent to the engine on the next background
        loop iteration."""
        if request_id in self._request_streams:
            raise KeyError(f"Request {request_id} already exists.")

        stream = AsyncStream(request_id)
        self._new_requests.put_nowait(
            (stream, {"seq_id": request_id, **engine_add_request_kwargs})
        )

        self.new_requests_event.set()

        return stream

    def abort_request(self, request_id: str, *, verbose: bool = False) -> None:
        """Abort a request during next background loop iteration."""
        if verbose:
            logger.info(f"Aborted request {request_id}.")

        self._finished_requests.put_nowait(request_id)

        if (
            request_id not in self._request_streams
            or self._request_streams[request_id].finished
        ):
            # The request has already finished or been aborted.
            return

        self._request_streams[request_id].finish()

    def get_new_and_finished_requests(self) -> Tuple[List[Dict], Set[str]]:
        """Get the new requests and finished requests to be
        sent to the engine."""
        new_requests: List[Dict] = []
        finished_requests: Set[str] = set()

        while not self._finished_requests.empty():
            request_id = self._finished_requests.get_nowait()
            finished_requests.add(request_id)
            self._request_streams.pop(request_id, None)

        while not self._new_requests.empty():
            stream, new_request = self._new_requests.get_nowait()
            if stream.request_id in finished_requests:
                # The request has already been aborted.
                stream.finish()
                continue
            self._request_streams[stream.request_id] = stream
            new_requests.append(new_request)

        return new_requests, finished_requests

    async def wait_for_new_requests(self):
        if not self.has_new_requests():
            await self.new_requests_event.wait()
        self.new_requests_event.clear()

    def has_new_requests(self):
        return not self._new_requests.empty()


def _log_task_completion(
    task: asyncio.Task, error_callback: Callable[[Exception], None]
) -> None:
    """This function is only intended for the `engine.run_engine_loop()` task.

    In particular, that task runs a `while True` loop that can only exit if
    there is an exception.
    """

    exception = None
    try:
        return_value = task.result()
        raise AssertionError(
            f"The engine background task should never finish without an "
            f"exception. {return_value}"
        )
    except asyncio.exceptions.CancelledError:
        # We assume that if the task is cancelled, we are gracefully shutting
        # down. This should only happen on program exit.
        logger.info("Engine is gracefully shutting down.")
    except Exception as e:
        exception = e
        logger.error("Engine background task failed", exc_info=e)
        error_callback(exception)
        raise RuntimeError(
            "Task finished unexpectedly. This should never happen! "
            "Please open an issue on Github. See stack trace above for the"
            "actual cause."
        ) from e


class _AsyncLLMEngine(LLMEngine):
    """Extension of LLMEngine to add async methods."""

    def __init__(self, engine: LLMEngine) -> None:
        super().__init__()

        self.engine = engine

    def get_model_config(self) -> ModelConfig:
        return self.engine.get_model_config()

    def add_request(
        self,
        seq_id: str,
        prompt: Optional[str],
        sampling_params: SamplingParams,
    ) -> None:
        self.engine.add_request(
            prompt=prompt,
            sampling_params=sampling_params,
            seq_id=seq_id,
        )

    async def step_async(self) -> List[RequestOutput]:
        """
        Simple wrapper around the synchronous `step` method to make it
        """
        return await asyncio.get_event_loop().run_in_executor(
            None, self.engine.step, False
        )


class AsyncLLMEngine(LLMEngine):
    """An asynchronous wrapper for :class:`LLMEngine`.

    This class is used to wrap the :class:`LLMEngine` class to make it
    asynchronous. It uses asyncio to create a background loop that keeps
    processing incoming requests. The :class:`LLMEngine` is kicked by the
    generate method when there are requests in the waiting queue. The generate
    method yields the outputs from the :class:`LLMEngine` to the caller.
    """

    def __init__(self, engine: _AsyncLLMEngine, verbose: bool = False) -> None:
        self.engine = engine
        self.verbose = verbose

        self.background_loop: Optional[asyncio.Future] = None
        # We need to keep a reference to unshielded
        # task as well to prevent it from being garbage
        # collected
        self._background_loop_unshielded: Optional[asyncio.Task] = None
        self._errored_with: Optional[BaseException] = None

        # Lazy initialized fields
        self._request_tracker: RequestTracker

    @classmethod
    def from_system_config(
        cls, config: SystemConfig, verbose: bool = False
    ) -> "LLMEngine":
        """Creates an LLM engine from the engine arguments."""
        engine = super().from_system_config(config)
        return cls(_AsyncLLMEngine(engine), verbose=verbose)

    @property
    def is_running(self) -> bool:
        return (
            self.background_loop is not None
            and self._background_loop_unshielded is not None
            and not self._background_loop_unshielded.done()
        )

    @property
    def is_stopped(self) -> bool:
        return self.errored or (
            self.background_loop is not None
            and self._background_loop_unshielded is not None
            and self._background_loop_unshielded.done()
        )

    @property
    def errored(self) -> bool:
        return self._errored_with is not None

    def set_errored(self, exc: Exception) -> None:
        self._errored_with = exc

    def _error_callback(self, exc: Exception) -> None:
        self.set_errored(exc)
        self._request_tracker.propagate_exception(exc)

    def start_background_loop(self) -> None:
        """Start the background loop."""
        if self.errored:
            raise RuntimeError(
                "Background loop has errored already."
            ) from self._errored_with
        if self.is_running:
            raise RuntimeError("Background loop is already running.")
        # Initialize the RequestTracker here so it uses the right event loop.
        self._request_tracker = RequestTracker()

        self._background_loop_unshielded = asyncio.get_event_loop().create_task(
            self.run_engine_loop()
        )
        self._background_loop_unshielded.add_done_callback(
            partial(_log_task_completion, error_callback=self._error_callback)
        )
        self.background_loop = asyncio.shield(self._background_loop_unshielded)

    async def engine_step(self) -> bool:
        """Kick the engine to process the waiting requests.

        Returns True if there are in-progress requests."""

        new_requests, finished_requests = (
            self._request_tracker.get_new_and_finished_requests()
        )

        for new_request in new_requests:
            # Add the request into the vLLM engine's waiting queue.
            # TODO: Maybe add add_request_batch to reduce Ray overhead
            try:
                self.engine.add_request(**new_request)
            except ValueError as e:
                # TODO: use a vLLM specific error for failed validation
                self._request_tracker.process_exception(
                    new_request["request_id"],
                    e,
                    verbose=self.verbose,
                )

        if finished_requests:
            await self._engine_abort(finished_requests)

        request_outputs = await self.engine.step_async()

        # Put the outputs into the corresponding streams.
        for request_output in request_outputs:
            self._request_tracker.process_request_output(
                request_output, verbose=self.verbose
            )

        return len(request_outputs) > 0

    async def _engine_abort(self, request_ids: Iterable[str]):
        # TODO(amey): Add support for aborting request in scheduler
        pass

    async def run_engine_loop(self):
        while True:
            # Abort if iteration takes too long due to unrecoverable errors
            # (eg. NCCL timeouts).
            try:
                await asyncio.wait_for(self.engine_step(), ENGINE_ITERATION_TIMEOUT_S)
            except asyncio.TimeoutError as exc:
                logger.error("Engine iteration timed out. This should never happen!")
                self.set_errored(exc)
                raise
            await asyncio.sleep(0)

    async def get_model_config(self) -> ModelConfig:
        return self.engine.get_model_config()

    async def add_request(
        self,
        request_id: str,
        prompt: str,
        sampling_params: SamplingParams,
    ) -> AsyncStream:
        if self.verbose:
            logger.info(
                f"Received request {request_id}: prompt: {prompt[:MAX_PROMPT_LOG_LEN]}, sampling_params: {sampling_params}"
            )

        if not self.is_running:
            self.start_background_loop()

        stream = self._request_tracker.add_request(
            request_id,
            prompt=prompt,
            sampling_params=sampling_params,
        )

        return stream

    async def generate(
        self,
        request_id: str,
        prompt: str,
        sampling_params: SamplingParams,
    ) -> AsyncIterator[RequestOutput]:
        """Generate outputs for a request.

        Generate outputs for a request. This method is a coroutine. It adds the
        request into the waiting queue of the LLMEngine and streams the outputs
        from the LLMEngine to the caller.

        Args:
            prompt: Input prompt to LLM.
            sampling_params: The sampling parameters of the request.
            request_id: The unique id of the request.

        Yields:
            The output `RequestOutput` objects from the LLMEngine
            for the request.

        Details:
            - If the engine is not running, start the background loop,
              which iteratively invokes
              :meth:`~sarathi.engine.async_llm_engine.AsyncLLMEngine.engine_step`
              to process the waiting requests.
            - Add the request to the engine's `RequestTracker`.
              On the next background loop, this request will be sent to
              the underlying engine.
              Also, a corresponding `AsyncStream` will be created.
            - Wait for the request outputs from `AsyncStream` and yield them.

        Example:
            >>> # Please refer to entrypoints/api_server.py for
            >>> # the complete example.
            >>>
            >>> # initialize the engine and the example input
            >>> engine = AsyncLLMEngine.from_engine_args(engine_args)
            >>> example_input = {
            >>>     "prompt": "What is LLM?",
            >>>     "stream": False, # assume the non-streaming case
            >>>     "temperature": 0.0,
            >>>     "request_id": 0,
            >>> }
            >>>
            >>> # start the generation
            >>> results_generator = engine.generate(
            >>>    example_input["prompt"],
            >>>    SamplingParams(temperature=example_input["temperature"]),
            >>>    example_input["request_id"])
            >>>
            >>> # get the results
            >>> final_output = None
            >>> async for request_output in results_generator:
            >>>     if await request.is_disconnected():
            >>>         # Abort the request if the client disconnects.
            >>>         await engine.abort(request_id)
            >>>         # Return or raise an error
            >>>         ...
            >>>     final_output = request_output
            >>>
            >>> # Process and return the final output
            >>> ...
        """
        async for output in self._process_request(
            request_id,
            prompt,
            sampling_params,
        ):
            yield output

    async def _process_request(
        self,
        request_id: str,
        prompt: str,
        sampling_params: SamplingParams,
    ) -> AsyncIterator[RequestOutput]:
        """Common logic to process requests with SamplingParams or
        PoolingParams."""
        stream = await self.add_request(
            request_id,
            prompt,
            sampling_params,
        )

        try:
            async for request_output in stream:
                yield request_output
        except (Exception, asyncio.CancelledError) as e:
            self._abort(request_id)
            raise e

    async def abort(self, request_id: str) -> None:
        """Abort a request.

        Abort a submitted request. If the request is finished or not found,
        this method will be a no-op.

        Args:
            request_id: The unique id of the request.
        """
        if not self.is_running:
            raise RuntimeError(
                "Background loop is not running. If it was running, "
                "inspect the output to find the stacktrace of the "
                "error that caused the background loop to stop."
            )

        return self._abort(request_id)

    def _abort(self, request_id: str) -> None:
        """Abort a request.

        Abort a submitted request. If the request is finished or not found,
        this method will be a no-op.

        Args:
            request_id: The unique id of the request.
        """
        self._request_tracker.abort_request(request_id, verbose=self.verbose)
