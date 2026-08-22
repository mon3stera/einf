from einf.execution import ExecutionResult, Executor, RequestExecutionResult
from einf.execution_plan import BatchPlan


class FakeExecutor(Executor):
    def execute(self, batch: BatchPlan) -> ExecutionResult:
        results = tuple(
            RequestExecutionResult(
                request_id=request.request_id,
                generated_token_ids=(request.input_token_ids[-1] + 1,)
                if request.need_sample
                else (),
                cached_len_delta=len(request.input_token_ids),
                is_eos=False,
            )
            for request in batch.requests
        )
        return ExecutionResult(step_id=batch.step_id, request_results=results)
