from abc import ABC, abstractmethod
from enum import Enum
from typing import Any


class BaseRegistry(ABC):
    _key_class = Enum

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        cls._registry = {}

    @classmethod
    def register(cls, key: Enum, implementation_class: Any) -> None:
        if key in cls._registry:
            return

        cls._registry[key] = implementation_class

    @classmethod
    def unregister(cls, key: Enum) -> None:
        if key not in cls._registry:
            raise ValueError(f"{key} is not registered")

        del cls._registry[key]

    @classmethod
    def get(cls, key: Enum, *args, **kwargs) -> Any:
        if key not in cls._registry:
            raise ValueError(f"{key} is not registered")

        return cls._registry[key](*args, **kwargs)

    @classmethod
    def get_class(cls, key: Enum) -> Any:
        if key not in cls._registry:
            raise ValueError(f"{key} is not registered")

        return cls._registry[key]

    @classmethod
    @abstractmethod
    def get_key_from_str(cls, key_str: str) -> Enum:
        pass

    @classmethod
    def get_from_str(cls, key_str: str, *args, **kwargs) -> Any:
        return cls.get(cls.get_key_from_str(key_str), *args, **kwargs)
