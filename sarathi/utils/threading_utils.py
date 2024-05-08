import os
import traceback
from threading import Lock
from functools import wraps


def synchronized(method):
    """ Synchronization decorator at the instance level. """

    @wraps(method)
    def synced_method(self, *args, **kwargs):
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
        except Exception:
            traceback.print_exc()
            os._exit(1)

    return wrapper
