from einf.execution import Executor
from einf.scheduler import Scheduler


class LLMServer:
    def __init__(self, scheduler: Scheduler, executor: Executor) -> None:
        self.scheduler = scheduler
        self.executor = executor

    def run_once(self) -> bool:
        batch = self.scheduler.schedule()

        if batch is None:
            return False

        try:
            result = self.executor.execute(batch)
        except Exception as error:
            self.scheduler.fail_batch(batch, f"{type(error).__name__}: {error}")
            return True

        self.scheduler.apply_result(result)
        return True

    def run_until_idle(self) -> None:
        while self.run_once():
            pass
