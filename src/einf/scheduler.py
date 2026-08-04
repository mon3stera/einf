"""Scheduler policy and Request lifecycle ownership.

The Scheduler is the single writer of Request lifecycle state.
"""

from abc import ABC, abstractmethod
from collections import deque, OrderedDict
from dataclasses import dataclass
from enum import Enum, auto
from typing import Iterator

from einf.cache.manager import KVCacheManager
from einf.execution import ExecutionResult
from einf.request import AdvanceResult, CompletionReason, Request, RequestSpec, RequestState


@dataclass
class RequestBundle:
    request_id: str
    work_type: "WorkType"

class WorkType(Enum):
    PREFILL = auto()
    DECODE = auto()

@dataclass(frozen=True, slots=True)
class ScheduledRequest:
    request_id: str
    input_token_ids: tuple[int, ...]
    work_type: WorkType
    start_position: int
    block_table: tuple[int, ...]
    need_sample: bool

@dataclass(frozen=True, slots=True)
class ScheduledBatch:
    step_id: int
    requests: tuple[ScheduledRequest, ...]

class Policy(ABC):
    @abstractmethod
    def add(self, bundle: RequestBundle) -> None:
        pass

    @abstractmethod
    def update(self, bundle: RequestBundle) -> bool:
        pass

    @abstractmethod
    def remove(self, request_id: str) -> RequestBundle | None:
        pass

    @abstractmethod
    def pop_victim(self) -> RequestBundle | None:
        pass

    @abstractmethod
    def __len__(self) -> int:
        pass

    @abstractmethod
    def candidates(self) -> tuple[RequestBundle, ...]:
        pass


class FCFSPolicy(Policy):
    def __init__(self) -> None:
        self._data: OrderedDict[str, RequestBundle] = OrderedDict()

    def add(self, bundle: RequestBundle) -> None:
        self._data[bundle.request_id] = bundle

    def update(self, bundle: RequestBundle) -> bool:
        if bundle.request_id in self._data:
            self._data[bundle.request_id] = bundle
            return True
        return False

    def remove(self, request_id: str) -> RequestBundle | None:
        return self._data.pop(request_id, None)

    def candidates(self) -> tuple[RequestBundle, ...]:
        return tuple(self._data.values())

    def pop_victim(self) -> RequestBundle | None:
        if self._data:
            _, element = self._data.popitem(last=True)
            return element
        return None

    def __len__(self) -> int:
        return len(self._data)

class DecodeFirstPolicy(FCFSPolicy):
    def candidates(self) -> tuple[RequestBundle, ...]:
        bundles = super().candidates()
        decode = tuple(
            bundle
            for bundle in bundles
            if bundle.work_type is WorkType.DECODE
        )
        prefill = tuple(
            bundle
            for bundle in bundles
            if bundle.work_type is WorkType.PREFILL
        )
        return decode + prefill

    def pop_victim(self) -> RequestBundle | None:
        for bundle in reversed(self._data.values()):
            if bundle.work_type is WorkType.PREFILL:
                return self._data.pop(bundle.request_id)

        return super().pop_victim()


class Scheduler:
    def __init__(
        self,
        policy: Policy,
        cache_manager: KVCacheManager,
        *,
        max_batch_len: int,
        max_prefill_chunk_len: int
    ) -> None:
        self._requests: dict[str, Request] = {}
        self._next_arrival_index = 0
        self._next_step_id = 0
        self._policy = policy
        self._waiting: deque[RequestBundle] = deque()
        self._cache_manager = cache_manager
        self._max_prefill_chunk_len = max_prefill_chunk_len
        self._max_batch_len = max_batch_len

    def next_arrival_index(self) -> int:
        index = self._next_arrival_index
        self._next_arrival_index += 1
        return index

    def _get_request(self, request_id: str) -> Request:
        return self._requests[request_id]

    def iter_running_bundles(self) -> Iterator[RequestBundle]:
        for bundle in self._policy.candidates():
            request = self._get_request(bundle.request_id)

            if request.state is RequestState.RUNNING:
                yield bundle

    def submit(self, spec: RequestSpec) -> str:
        if spec.request_id in self._requests:
            raise ValueError(f"Request {spec.request_id} has been registered")

        request = Request.create(spec, self.next_arrival_index())
        self._requests[spec.request_id] = request
        self._waiting.append(self.build_bundle(request))
        return spec.request_id

    def admit(self, request_id: str) -> None:
        request = self._get_request(request_id)
        request.admit()
        self._policy.add(self.build_bundle(request))

    def next_work_type(self, request: Request) -> WorkType:
        if request.cached_len >= len(request.prompt_token_ids):
            return WorkType.DECODE
        else:
            return WorkType.PREFILL

    def chunk_prefill_len(self, pending_len: int, remaining_len: int) -> int:
        return min(
            self._max_prefill_chunk_len,
            pending_len,
            remaining_len,
        )

    def preempt(self) -> RequestBundle | None:
        bundle = self._policy.pop_victim()

        if bundle is None:
            return None

        request = self._requests[bundle.request_id]

        request.preempt()
        self._waiting.appendleft(self.build_bundle(request))
        self._cache_manager.release(bundle.request_id)

        return bundle

    def _try_schedule_request(
        self,
        request: Request,
        remaining_len: int,
        allow_preemption: bool,
    ) -> tuple[ScheduledRequest, int] | None:
        work_type = self.next_work_type(request)

        context_token_ids = (request.prompt_token_ids + tuple(request.generated_token_ids))
        start_position = request.cached_len
        pending_token_ids = context_token_ids[start_position:]

        if work_type is WorkType.PREFILL:
            scheduled_len = self.chunk_prefill_len(
                len(pending_token_ids),
                remaining_len,
            )
        else:
            scheduled_len = 1

        input_token_ids = pending_token_ids[:scheduled_len]

        required_cache_len = start_position + scheduled_len

        need_sample = scheduled_len == len(pending_token_ids)

        if scheduled_len > remaining_len:
            return None

        if allow_preemption:
            while not self._cache_manager.allocate(
                request.request_id,
                required_cache_len,
            ):
                if len(self._policy) == 1:
                    self.fail(request.request_id, f"Insufficient memory to fulfill request {request.request_id}")
                    return None

                bundle = self.preempt()

                if bundle is None:
                    return None
                if bundle.request_id == request.request_id:
                    return None

        else:
            if not self._cache_manager.allocate(request.request_id, required_cache_len):
                return None

        return (
            ScheduledRequest(
                request_id=request.request_id,
                input_token_ids=input_token_ids,
                work_type=self.next_work_type(request),
                start_position=request.cached_len,
                block_table=self._cache_manager.block_table(
                    request.request_id
                ),
                need_sample=need_sample,
            ),
            scheduled_len,
        )

    def schedule(self) -> ScheduledBatch | None:
        remaining_len = self._max_batch_len
        scheduled_requests: list[ScheduledRequest] = []

        for bundle in self.iter_running_bundles():
            request = self._get_request(bundle.request_id)
            scheduled = self._try_schedule_request(
                request,
                remaining_len,
                allow_preemption=True,
            )

            if scheduled is None:
                continue

            scheduled_request, scheduled_len = scheduled
            scheduled_requests.append(scheduled_request)
            remaining_len -= scheduled_len

            if remaining_len == 0:
                break

        while remaining_len > 0 and self._waiting:
            bundle = self._waiting[0]
            request = self._get_request(bundle.request_id)

            if request.state is not RequestState.WAITING:
                self._waiting.popleft()
                continue

            scheduled = self._try_schedule_request(
                request,
                remaining_len,
                allow_preemption=False,
            )

            if scheduled is None:
                break

            self._waiting.popleft()
            self.admit(request.request_id)
            scheduled_request, scheduled_len = scheduled
            scheduled_requests.append(scheduled_request)
            remaining_len -= scheduled_len

        if not scheduled_requests:
            return None

        batch = ScheduledBatch(
            step_id=self._next_step_id,
            requests=tuple(scheduled_requests),
        )
        self._next_step_id += 1
        return batch

    def build_bundle(self, request: Request) -> RequestBundle:
        if request.cached_len >= len(request.prompt_token_ids):
            work_type = WorkType.DECODE
        else:
            work_type = WorkType.PREFILL
        return RequestBundle(request_id=request.request_id, work_type=work_type)

    def advance(self, request_id: str, result: AdvanceResult) -> None:
        request = self._get_request(request_id)
        request.advance(result)

    def cancel(self, request_id: str) -> None:
        request = self._get_request(request_id)
        request.cancel()
        self._policy.remove(request_id)
        self._cache_manager.release(request_id)

    def fail(self, request_id: str, error: str) -> None:
        request = self._get_request(request_id)
        request.fail(error)
        self._policy.remove(request_id)
        self._cache_manager.release(request_id)

    def fail_batch(self, batch: ScheduledBatch, error: str) -> None:
        for r in batch.requests:
            self.fail(r.request_id, error)

    def apply_result(self, result: ExecutionResult) -> None:
        for request_result in result.request_results:
            request = self._get_request(request_result.request_id)
            reason = None

            if request_result.is_eos:
                reason = CompletionReason.EOS
            elif (
                len(request.generated_token_ids)
                + len(request_result.generated_token_ids)
                == request.max_new_len
            ):
                reason = CompletionReason.LENGTH

            self.advance(
                request.request_id,
                AdvanceResult(
                    generated_token_ids=list(
                        request_result.generated_token_ids
                    ),
                    cached_len_delta=(
                        request_result.cached_len_delta
                    ),
                    completion_reason=reason,
                ),
            )

            if request.state is RequestState.RUNNING:
                self._policy.update(self.build_bundle(request))
            else:
                self._policy.remove(request.request_id)
                self._cache_manager.release(request.request_id)
