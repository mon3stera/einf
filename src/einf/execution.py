"""Execution-plane protocol plus native Rust/Python result values."""

from abc import ABC, abstractmethod

from einf._control import ExecutionResult, RequestExecutionResult
from einf.execution_plan import BatchPlan


class Executor(ABC):
    @abstractmethod
    def execute(self, batch: BatchPlan) -> ExecutionResult:
        pass


__all__ = ["ExecutionResult", "Executor", "RequestExecutionResult"]
