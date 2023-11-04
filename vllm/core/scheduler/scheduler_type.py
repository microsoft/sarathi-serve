from vllm.utils.base_int_enum import BaseIntEnum


class SchedulerType(BaseIntEnum):
    VLLM = 1
    ORCA = 2
    FASTER_TRANSFORMER = 3
    SARATHI = 4
    DSARATHI = 5
