import torch

from einf.executors.torch.input import ModelInput
from einf.executors.torch.output import ModelOutput


class DeterministicModelRunner:
    def __init__(self, *, vocab_size: int) -> None:
        if vocab_size <= 0:
            raise ValueError("vocab_size must be positive")
        self._vocab_size = vocab_size

    def forward(self, model_input: ModelInput) -> ModelOutput:
        target_token_ids = (
            model_input.input_token_ids + 1
        ) % self._vocab_size
        logits = torch.full(
            (
                model_input.input_token_ids.numel(),
                self._vocab_size,
            ),
            float("-inf"),
            dtype=torch.float32,
            device=model_input.input_token_ids.device,
        )
        logits.scatter_(
            1,
            target_token_ids.unsqueeze(1),
            0.0,
        )
        return ModelOutput(logits=logits)
