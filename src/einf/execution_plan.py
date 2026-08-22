"""Immutable execution-plan values shared by control and model planes.

The control plane owns lifecycle state and constructs :class:`BatchPlan`
values. The Torch execution plane consumes them without modifying request or
block state. A future Rust control plane can produce the same contract.
"""

from dataclasses import dataclass
from enum import Enum, auto


class WorkType(Enum):
    PREFILL = auto()
    DECODE = auto()


@dataclass(frozen=True, slots=True)
class ScheduledRequest:
    """One request's immutable work description for one execution step."""

    request_id: str
    input_token_ids: tuple[int, ...]
    work_type: WorkType
    start_position: int
    block_table: tuple[int, ...]
    need_sample: bool


@dataclass(frozen=True, slots=True)
class BatchPlan:
    """A complete immutable execution plan for one scheduler step."""

    step_id: int
    requests: tuple[ScheduledRequest, ...]


# Compatibility name used by the existing Python control-plane tests/callers.
ScheduledBatch = BatchPlan


__all__ = [
    "BatchPlan",
    "ScheduledBatch",
    "ScheduledRequest",
    "WorkType",
]
