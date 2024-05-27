# RWConfig is for the original tiiuae/falcon-40b(-instruct) and
# tiiuae/falcon-7b(-instruct) models. Newer Falcon models will use the
# `FalconConfig` class from the official HuggingFace transformers library.
from sarathi.transformers_utils.configs.falcon import RWConfig
from sarathi.transformers_utils.configs.qwen import QWenConfig
from sarathi.transformers_utils.configs.yi import YiConfig

__all__ = [
    "QWenConfig",
    "RWConfig",
    "YiConfig",
]
