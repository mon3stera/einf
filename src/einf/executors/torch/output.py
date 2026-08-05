from dataclasses import dataclass

from torch import Tensor


@dataclass(frozen=True, slots=True)
class ModelOutput:
    logits: Tensor
