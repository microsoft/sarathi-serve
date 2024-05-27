from typing import List

from sarathi.core.datatypes.sequence import Sequence


class Policy:

    def get_priority(
        self,
        now: float,
        seq: Sequence,
    ) -> float:
        raise NotImplementedError

    def sort_by_priority(
        self,
        now: float,
        seqs: List[Sequence],
    ) -> List[Sequence]:
        return sorted(
            seqs,
            key=lambda seq: self.get_priority(now, seq),
            reverse=True,
        )


class FCFS(Policy):

    def get_priority(
        self,
        now: float,
        seq: Sequence,
    ) -> float:
        return now - seq.arrival_time


class PolicyFactory:

    _POLICY_REGISTRY = {
        "fcfs": FCFS,
    }

    @classmethod
    def get_policy(cls, policy_name: str, **kwargs) -> Policy:
        return cls._POLICY_REGISTRY[policy_name](**kwargs)
