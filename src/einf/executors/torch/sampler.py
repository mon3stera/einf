from dataclasses import dataclass

import torch
from torch import Tensor

@dataclass(frozen=True, slots=True)
class SamplingBatch:
    is_all_greedy: bool = False
    temperatures: Tensor | None = None
    top_ks: Tensor | None = None
    top_ps: Tensor | None = None
    min_ps: Tensor | None = None
    seeds: Tensor | None = None
    offsets: Tensor | None = None

@dataclass(frozen=True, slots=True)
class SamplingOutput:
    token_ids: Tensor
    logprobs: Tensor | None = None

class Sampler:
    def sample(self, logits: Tensor, batch: SamplingBatch) -> SamplingOutput:
        batch_size, vocab_size = logits.shape

        if batch_size == 0:
            return SamplingOutput(
                token_ids=torch.empty(0, dtype=torch.long, device=logits.device),
                logprobs=torch.empty(0, dtype=torch.float32, device=logits.device),
            )

        if batch.is_all_greedy:
            return SamplingOutput(token_ids=logits.argmax(dim=-1), logprobs=None)
        if (
            batch.temperatures is None
            or batch.top_ks is None
            or batch.top_ps is None
            or batch.min_ps is None
            or batch.seeds is None
            or batch.offsets is None
        ):
            raise ValueError("non-greedy SamplingBatch requires parameter tensors")

        logits = logits.to(torch.float32)
        is_greedy = batch.temperatures == 0.0

        temperatures = torch.where(
            is_greedy,
            torch.ones_like(batch.temperatures),
            batch.temperatures
        ).unsqueeze(-1)

        logits = logits / temperatures

        max_top_k = int(batch.top_ks.max().item())
        if max_top_k > 0:
            topk_vals, _ = torch.topk(logits, min(max_top_k, vocab_size), dim=-1)

            k_indices = torch.clamp(batch.top_ks - 1, min=0, max=topk_vals.shape[-1] - 1)
            k_cutoffs = topk_vals.gather(dim=-1, index=k_indices.unsqueeze(-1))

            top_k_mask = (batch.top_ks.unsqueeze(-1) > 0) & (logits < k_cutoffs)
            logits.masked_fill_(top_k_mask, float("-inf"))

        if (batch.top_ps < 1.0).any():
            sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
            sorted_probs = torch.softmax(sorted_logits, dim=-1)
            cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

            sorted_mask = cumulative_probs > batch.top_ps.unsqueeze(-1)
            sorted_mask[..., 1:] = sorted_mask[..., :-1].clone()
            sorted_mask[..., 0] = False

            sorted_mask = sorted_mask & (batch.top_ps.unsqueeze(-1) < 1.0)
            sorted_logits.masked_fill_(sorted_mask, float("-inf"))
            logits = torch.scatter(logits, dim=-1, index=sorted_indices, src=sorted_logits)
            
        if (batch.min_ps > 0.0).any():
            probs = torch.softmax(logits, dim=-1)
            max_probs = probs.max(dim=-1, keepdim=True).values
            min_p_thresholds = max_probs * batch.min_ps.unsqueeze(-1)
            min_p_mask = (batch.min_ps.unsqueeze(-1) > 0.0) & (probs < min_p_thresholds)
            logits.masked_fill_(min_p_mask, float("-inf"))

        final_probs = torch.softmax(logits, dim=-1)

        greedy_tokens = torch.argmax(logits, dim=-1)
        sampled_tokens = torch.empty(batch_size, dtype=torch.long, device=logits.device)
        
        for i in range(batch_size):
            if is_greedy[i]:
                continue

            gen = torch.Generator(device=logits.device)
            gen.manual_seed(int(batch.seeds[i].item() + batch.offsets[i].item()))
            sampled_tokens[i] = torch.multinomial(final_probs[i], num_samples=1, generator=gen)

        selected_tokens = torch.where(is_greedy, greedy_tokens, sampled_tokens)

        logprobs_matrix = torch.log_softmax(logits, dim=-1)
        selected_logprobs = logprobs_matrix.gather(dim=-1, index=selected_tokens.unsqueeze(-1)).squeeze(-1)

        return SamplingOutput(token_ids=selected_tokens, logprobs=selected_logprobs)
        
            
            
