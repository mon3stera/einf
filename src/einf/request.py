"""Request lifecycle domain model for Gate 2.1."""

from dataclasses import dataclass, field
from enum import StrEnum, auto


class RequestException(Exception):
    """Base exception for Request lifecycle errors."""


class InvalidStateTransition(RequestException):
    """Raised when an operation is not legal from the current state."""


class RequestInvariantViolation(RequestException):
    """Raised when Request fields do not agree with its lifecycle state."""


@dataclass(frozen=True, slots=True)
class RequestSpec:
    request_id: str
    prompt_token_ids: tuple[int, ...]
    max_new_len: int


class RequestState(StrEnum):
    WAITING = auto()
    RUNNING = auto()
    FINISHED = auto()
    CANCELLED = auto()
    FAILED = auto()


class CompletionReason(StrEnum):
    EOS = auto()
    LENGTH = auto()


@dataclass(frozen=True, slots=True)
class AdvanceResult:
    generated_token_ids: list[int]
    cached_len_delta: int
    completion_reason: CompletionReason | None


@dataclass(slots=True)
class Request:
    request_id: str
    prompt_token_ids: tuple[int, ...]
    arrival_index: int
    max_new_len: int
    generated_token_ids: list[int] = field(default_factory=list)
    state: RequestState = RequestState.WAITING
    cached_len: int = 0
    completion_reason: CompletionReason | None = None
    error: str | None = None

    @classmethod
    def create(cls, spec: RequestSpec, arrival_index: int) -> "Request":
        request = cls(
            request_id=spec.request_id,
            prompt_token_ids=spec.prompt_token_ids,
            max_new_len=spec.max_new_len,
            arrival_index=arrival_index,
        )
        request.assert_invariants()
        return request

    def pending_len(self) -> int:
        if self.cached_len >= len(self.prompt_token_ids):
            return 1
        else:
            return len(self.prompt_token_ids) - self.cached_len

    def reach_token_limit(self) -> bool:
        return len(self.generated_token_ids) >= self.max_new_len

    def add_token(self, token_id: int) -> None:
        self.generated_token_ids.append(token_id)

    def assert_state(self, *allowed_states: RequestState) -> None:
        if self.state not in allowed_states:
            allowed = ", ".join(state.value for state in allowed_states)
            raise InvalidStateTransition(
                f"request {self.request_id}: cannot transition from "
                f"{self.state.value}; expected one of [{allowed}]"
            )

    def assert_invariants(self) -> None:
        if len(self.generated_token_ids) > self.max_new_len:
            raise RequestInvariantViolation(
                f"request {self.request_id}: generated token count "
                f"{len(self.generated_token_ids)} exceeds max_new_len "
                f"{self.max_new_len}"
            )

        if self.cached_len < 0:
            raise RequestInvariantViolation(
                f"request {self.request_id}: cached_len must be non-negative"
            )

        if self.state in (RequestState.WAITING, RequestState.RUNNING):
            if self.completion_reason is not None or self.error is not None:
                raise RequestInvariantViolation(
                    f"request {self.request_id}: {self.state.value} request cannot "
                    "have completion_reason or error"
                )
            return

        if self.state is RequestState.FINISHED:
            if self.completion_reason not in (
                CompletionReason.EOS,
                CompletionReason.LENGTH,
            ) or self.error is not None:
                raise RequestInvariantViolation(
                    f"request {self.request_id}: finished request requires a valid "
                    "completion_reason and no error"
                )
            return

        if self.state is RequestState.CANCELLED:
            if self.completion_reason is not None or self.error is not None:
                raise RequestInvariantViolation(
                    f"request {self.request_id}: cancelled request cannot have "
                    "completion_reason or error"
                )
            return

        if self.state is RequestState.FAILED:
            if self.completion_reason is not None or self.error is None:
                raise RequestInvariantViolation(
                    f"request {self.request_id}: failed request requires an error "
                    "and no completion_reason"
                )
            return

        raise RequestInvariantViolation(
            f"request {self.request_id}: unknown state {self.state!r}"
        )

    def admit(self) -> None:
        self.assert_invariants()
        self.assert_state(RequestState.WAITING)
        self.state = RequestState.RUNNING
        self.assert_invariants()

    def preempt(self) -> None:
        self.assert_invariants()
        self.assert_state(RequestState.RUNNING)
        self.state = RequestState.WAITING
        self.cached_len = 0
        self.assert_invariants()

    def advance(self, result: AdvanceResult) -> None:
        self.assert_invariants()
        self.assert_state(RequestState.RUNNING)

        if result.completion_reason is not None and result.completion_reason not in (
            CompletionReason.EOS,
            CompletionReason.LENGTH,
        ):
            raise RequestInvariantViolation(
                f"request {self.request_id}: invalid completion reason "
                f"{result.completion_reason!r}"
            )

        generated_token_ids = [
            *self.generated_token_ids,
            *result.generated_token_ids,
        ]
        cached_len = (
            self.cached_len + result.cached_len_delta
        )

        if len(generated_token_ids) > self.max_new_len:
            raise RequestInvariantViolation(
                f"request {self.request_id}: advance would exceed max_new_len"
            )
        if cached_len < 0:
            raise RequestInvariantViolation(
                f"request {self.request_id}: advance would make "
                "cached_len negative"
            )

        self.generated_token_ids = generated_token_ids
        self.cached_len = cached_len

        if result.completion_reason is not None:
            self.finish(result.completion_reason)
        else:
            self.assert_invariants()

    def finish(self, reason: CompletionReason) -> None:
        self.assert_invariants()
        self.assert_state(RequestState.RUNNING)
        if reason not in (CompletionReason.EOS, CompletionReason.LENGTH):
            raise RequestInvariantViolation(
                f"request {self.request_id}: invalid completion reason {reason!r}"
            )

        self.state = RequestState.FINISHED
        self.completion_reason = reason
        self.assert_invariants()

    def fail(self, error: str) -> None:
        self.assert_invariants()
        self.assert_state(RequestState.RUNNING)
        if error is None:
            raise RequestInvariantViolation(
                f"request {self.request_id}: failure requires an error"
            )

        self.state = RequestState.FAILED
        self.error = error
        self.assert_invariants()

    def cancel(self) -> None:
        self.assert_invariants()
        self.assert_state(
            RequestState.WAITING,
            RequestState.RUNNING,
            RequestState.CANCELLED,
        )

        if self.state is RequestState.CANCELLED:
            return

        self.state = RequestState.CANCELLED
        self.assert_invariants()
