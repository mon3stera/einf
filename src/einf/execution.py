"""Execution contracts and deterministic control-plane executor."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from einf.scheduler import ScheduledBatch


@dataclass(frozen=True, slots=True)
class RequestExecutionResult:
    request_id: str
    generated_token_ids: tuple[int, ...]
    cached_len_delta: int
    is_eos: bool = False


@dataclass(frozen=True, slots=True)
class ExecutionResult:
    step_id: int
    request_results: tuple[RequestExecutionResult, ...]


class Executor(ABC):
    @abstractmethod
    def execute(self, batch: ScheduledBatch) -> ExecutionResult:
        """Execute one scheduled batch."""


class FakeExecutor(Executor):
    """Produce deterministic cache progress and optional sampled tokens."""

    def execute(self, batch: ScheduledBatch) -> ExecutionResult:
        return ExecutionResult(
            step_id=batch.step_id,
            request_results=tuple(
                RequestExecutionResult(
                    request_id=request.request_id,
                    generated_token_ids=(
                        (request.input_token_ids[-1] + 1,)
                        if request.need_sample
                        else ()
                    ),
                    cached_len_delta=len(
                        request.input_token_ids
                    ),
                    is_eos=False,
                )
                for request in batch.requests
            ),
        )
