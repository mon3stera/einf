from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from einf.execution_plan import BatchPlan


@dataclass(frozen=True, slots=True)
class RequestExecutionResult:
    request_id: str
    generated_token_ids: tuple[int, ...]
    cached_len_delta: int
    is_eos: bool


@dataclass(frozen=True, slots=True)
class ExecutionResult:
    step_id: int
    request_results: tuple[RequestExecutionResult, ...]


class Executor(ABC):
    @abstractmethod
    def execute(self, batch: "BatchPlan") -> ExecutionResult:
        pass
