
from dataclasses import dataclass

from einf.scheduler import ScheduledBatch
import torch
from torch import Tensor


@dataclass(frozen=True, slots=True)
class ModelInput:
    input_token_ids: Tensor
    position: Tensor
    slot_mapping: Tensor
    query_start_loc: Tensor
    block_tables: Tensor
    context_lens: Tensor

    @classmethod
    def from_batch(cls, batch: ScheduledBatch, *, block_len: int, device: torch.device) -> "ModelInput":
        input_token_ids = []
        position = []
        slot_mapping = []
        query_start_loc = [0]
        context_lens = []
        block_tables = []

        for request in batch.requests:
            input_token_ids.extend(request.input_token_ids)

            position.extend([request.start_position + i for i in range(len(request.input_token_ids))])

            for t in range(len(request.input_token_ids)):
                absolute_position = t + request.start_position

                logical_block_idx = absolute_position // block_len
                slot_idx = absolute_position % block_len

                physical_flatten_slot_idx = request.block_table[logical_block_idx] * block_len + slot_idx

                slot_mapping.append(physical_flatten_slot_idx)

            query_start_loc.append(query_start_loc[-1] + len(request.input_token_ids))
            context_lens.append(request.start_position + len(request.input_token_ids))
            block_tables.append(
                torch.tensor(
                    request.block_table,
                    device=device,
                    dtype=torch.long,
                )
            )

        return cls(
            input_token_ids=torch.tensor(input_token_ids, device=device, dtype=torch.long),
            position=torch.tensor(position, device=device, dtype=torch.long),
            slot_mapping=torch.tensor(slot_mapping, device=device, dtype=torch.long),
            query_start_loc=torch.tensor(query_start_loc, device=device, dtype=torch.long),
            block_tables=torch.nn.utils.rnn.pad_sequence(block_tables, batch_first=True, padding_value=-1),
            context_lens=torch.tensor(context_lens, device=device, dtype=torch.long),
        )
