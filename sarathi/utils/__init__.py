import asyncio
import os
import socket
import sys
import uuid
from platform import uname
from typing import AsyncIterator, List, Tuple, TypeVar, Union

import psutil
import torch

T = TypeVar("T")


class Counter:

    def __init__(self, start: int = 0) -> None:
        self.counter = start

    def __next__(self) -> int:
        i = self.counter
        self.counter += 1
        return i

    def reset(self) -> None:
        self.counter = 0


def get_gpu_memory(gpu: int = 0) -> int:
    """Returns the total memory of the GPU in bytes."""
    return torch.cuda.get_device_properties(gpu).total_memory


def get_cpu_memory() -> int:
    """Returns the total CPU memory of the node in bytes."""
    return psutil.virtual_memory().total


def random_uuid() -> str:
    return str(uuid.uuid4().hex)


def get_ip() -> str:
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.settimeout(0)
    try:
        s.connect(("10.254.254.254", 1))
        ip = s.getsockname()[0]
    except Exception:
        ip = "127.0.0.1"
    finally:
        s.close()
    return ip


def get_open_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]


def set_cuda_visible_devices(device_ids: List[int]) -> None:
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, device_ids))


def unset_cuda_visible_devices() -> None:
    os.environ.pop("CUDA_VISIBLE_DEVICES", None)


def is_port_in_use(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(("localhost", port)) == 0


def get_random_port() -> int:
    port = None
    while not port or is_port_in_use(port):
        port = int(random_uuid(), 16) % (65536 - 8000) + 8000

    return port


def merge_async_iterators(*iterators: AsyncIterator[T]) -> AsyncIterator[Tuple[int, T]]:
    """Merge multiple asynchronous iterators into a single iterator.

    This method handle the case where some iterators finish before others.
    When it yields, it yields a tuple (i, item) where i is the index of the
    iterator that yields the item.
    """
    queue: asyncio.Queue[Union[Tuple[int, T], Exception]] = asyncio.Queue()

    finished = [False] * len(iterators)

    async def producer(i: int, iterator: AsyncIterator[T]):
        try:
            async for item in iterator:
                await queue.put((i, item))
        except Exception as e:
            await queue.put(e)
        finished[i] = True

    _tasks = [
        asyncio.create_task(producer(i, iterator))
        for i, iterator in enumerate(iterators)
    ]

    async def consumer():
        try:
            while not all(finished) or not queue.empty():
                item = await queue.get()
                if isinstance(item, Exception):
                    raise item
                yield item
        except (Exception, asyncio.CancelledError) as e:
            for task in _tasks:
                if sys.version_info >= (3, 9):
                    # msg parameter only supported in Python 3.9+
                    task.cancel(e)
                else:
                    task.cancel()
            raise e
        await asyncio.gather(*_tasks)

    return consumer()
