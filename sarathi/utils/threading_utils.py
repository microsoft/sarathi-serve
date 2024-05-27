import os
import traceback
from functools import wraps
from threading import Lock


def synchronized(method):
    """Synchronization decorator at the instance level."""

    @wraps(method)
    def synced_method(self, *args, **kwargs):
        # pylint: disable=protected-access
        if not hasattr(self, "_lock"):
            self._lock = Lock()

        with self._lock:
            return method(self, *args, **kwargs)

    return synced_method


def exit_on_error(func):

    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception:  # pylint: disable=broad-except
            traceback.print_exc()
            os._exit(1)  # pylint: disable=protected-access

    return wrapper
