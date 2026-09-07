"""Chain speculative decoding engine over two model runners.

Phase-2 scope: single request, eager forward path, own KV cache per
model. The verify forward processes ``[pending, d_0 .. d_{K-1}]`` —
K+1 query positions whose logits line up exactly with the
``accept_reject_*`` contract in ``speculative.py``:

    verify logits row i   scores draft slot i      (i in 0..K-1)
    verify logits row K   is the bonus distribution

KV rollback is bookkeeping-only: entries written beyond the accepted
prefix stay in the physical cache but are dead (context_lens excludes
them) and are overwritten as generation continues — the same retract
semantics the Rust control plane's ``rollback_advance`` applies at the
scheduler level.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch

from einf.executors.torch.input import ModelInput
from einf.executors.torch.output import ModelOutput
from einf.executors.torch.speculative import (
    accept_reject_greedy,
    accept_reject_sampling,
)


@dataclass
class SpecStats:
    """Acceptance bookkeeping across a generate call."""

    steps: int = 0
    proposed: int = 0
    accepted: int = 0
    committed: int = 0

    @property
    def accept_rate(self) -> float:
        return self.accepted / self.proposed if self.proposed else 0.0


@dataclass
class _Track:
    """Per-model speculative state."""

    forward: object                     # ModelInput -> ModelOutput
    block_table: list[int]
    context_len: int                    # tokens with live KV in this cache
    pending_token: int                  # committed token not yet in cache
    proposal_logits: list[torch.Tensor] = field(default_factory=list)


class SpeculativeEngine:
    """Single-request chain speculative decoding (greedy or sampling)."""

    def __init__(
        self,
        target: object,
        draft: object,
        *,
        device: torch.device,
        block_len: int = 16,
        num_blocks: int = 64,
        num_spec_tokens: int = 4,
    ) -> None:
        if num_spec_tokens < 1:
            raise ValueError("num_spec_tokens must be >= 1")
        self._device = device
        self._block_len = block_len
        self._num_spec_tokens = num_spec_tokens
        # Separate identity page tables: each runner owns its whole cache,
        # so physical block i is logical block i for both.
        self._target = _Track(
            forward=target.forward,
            block_table=list(range(num_blocks)),
            context_len=0,
            pending_token=-1,
        )
        self._draft = _Track(
            forward=draft.forward,
            # Reserve the last page: the runner-level decode graph captures
            # with it as the dummy block, and try_replay refuses any block
            # table that touches it. Harmless when the draft runs eager.
            block_table=list(range(num_blocks - 1)),
            context_len=0,
            pending_token=-1,
        )
        self._stats = SpecStats()

    # ------------------------------------------------------------------
    # ModelInput construction (single request, from engine bookkeeping)
    # ------------------------------------------------------------------

    def _make_input(self, track: _Track, q_token_ids: list[int]) -> ModelInput:
        n = len(q_token_ids)
        start = track.context_len
        needed_blocks = (start + n + self._block_len - 1) // self._block_len
        if needed_blocks > len(track.block_table):
            raise RuntimeError(
                f"speculative cache capacity exceeded: need {needed_blocks} "
                f"blocks, have {len(track.block_table)}"
            )
        positions = list(range(start, start + n))
        slots = [
            track.block_table[pos // self._block_len] * self._block_len
            + pos % self._block_len
            for pos in positions
        ]
        return ModelInput(
            input_token_ids=torch.tensor(q_token_ids, device=self._device, dtype=torch.long),
            position=torch.tensor(positions, device=self._device, dtype=torch.long),
            slot_mapping=torch.tensor(slots, device=self._device, dtype=torch.long),
            query_start_loc=torch.tensor([0, n], device=self._device, dtype=torch.long),
            block_tables=torch.tensor(
                [track.block_table], device=self._device, dtype=torch.long
            ),
            context_lens=torch.tensor([start + n], device=self._device, dtype=torch.long),
            query_start_loc_host=(0, n),
            context_lens_host=(start + n,),
        )

    # ------------------------------------------------------------------
    # phases
    # ------------------------------------------------------------------

    def prefill(
        self,
        prompt_ids: list[int],
        *,
        greedy: bool = True,
        generator: torch.Generator | None = None,
    ) -> int:
        """Run the prompt through both caches; return the first token."""
        if not prompt_ids:
            raise ValueError("prompt must be non-empty")

        out = self._target.forward(self._make_input(self._target, list(prompt_ids)))
        first = _pick(out.logits[-1], greedy=greedy, generator=generator)

        self._target.context_len = len(prompt_ids)
        self._target.pending_token = first

        # The draft primes its cache with the prompt too; its own argmax is
        # discarded because the committed sequence comes from the target.
        draft_out = self._draft.forward(self._make_input(self._draft, list(prompt_ids)))
        del draft_out
        self._draft.context_len = len(prompt_ids)
        self._draft.pending_token = first
        return first

    def _draft_decode_forward(self, draft, token: int):
        """One draft decode pass, preferring the runner's captured CUDA graph.

        The draft loop is K+1 identical q=1 forwards — exactly the shape the
        runner-level decode graph captures (embed through lm_head, batch
        bucket 1). Graph outputs are views into the static replay buffer and
        every replay overwrites them, so a graph hit must clone the logits
        before the proposal loop stores them for the accept/reject step.
        """
        model_input = self._make_input(draft, [token])
        try_graph = getattr(draft, "try_decode_cuda_graph", None)
        if try_graph is not None:
            out = try_graph(model_input)
            if out is not None:
                return ModelOutput(logits=out.logits.clone())
        return draft.forward(model_input)

    def step(
        self,
        *,
        greedy: bool = True,
        generator: torch.Generator | None = None,
    ) -> list[int]:
        """One draft-then-verify cycle; returns the committed tokens."""
        draft = self._draft

        # Draft decode passes: process pending, then every proposal. The
        # extra (K+1)-th pass writes the last proposal's KV so the draft
        # cache holds every committed draft, keeping the pending semantics
        # identical to the target's (only the final committed token is ever
        # missing from a cache).
        draft.proposal_logits.clear()
        proposals: list[int] = []
        token = draft.pending_token
        for i in range(self._num_spec_tokens + 1):
            out = self._draft_decode_forward(draft, token)
            draft.context_len += 1
            if i < self._num_spec_tokens:
                logits = out.logits[-1]
                draft.proposal_logits.append(logits)
                token = _pick(logits, greedy=greedy, generator=generator)
                proposals.append(token)

        # Verify: [pending, d_0 .. d_{K-1}] -> K+1 slot distributions.
        verify_q = [self._target.pending_token, *proposals]
        out = self._target.forward(self._make_input(self._target, verify_q))
        target_logits = out.logits[-(self._num_spec_tokens + 1):].unsqueeze(0)

        if greedy:
            accepted_tokens, accepted_len = accept_reject_greedy(
                torch.tensor([proposals], device=self._device, dtype=torch.long),
                target_logits,
            )
        else:
            draft_probs = torch.stack(
                [torch.softmax(lg.detach().float(), dim=-1) for lg in draft.proposal_logits]
            ).unsqueeze(0)
            accepted_tokens, accepted_len = accept_reject_sampling(
                torch.tensor([proposals], device=self._device, dtype=torch.long),
                draft_probs.to(self._device),
                target_logits,
                generator=generator,
            )

        n = int(accepted_len[0].item())
        committed = accepted_tokens[0, :n].tolist()

        # Bookkeeping: the verify forward wrote the old pending plus every
        # proposal, so the old pending and the accepted drafts are live on
        # the target; the last committed token (correction/bonus) becomes
        # the new pending everywhere. The draft wrote its pending and all K
        # proposals; the proposals beyond the accepted prefix are dead and
        # retract here.
        num_accepted_drafts = n - 1
        self._target.context_len += 1 + num_accepted_drafts
        self._draft.context_len -= self._num_spec_tokens - num_accepted_drafts
        self._target.pending_token = committed[-1]
        self._draft.pending_token = committed[-1]

        self._stats.steps += 1
        self._stats.proposed += self._num_spec_tokens
        self._stats.accepted += num_accepted_drafts
        self._stats.committed += n
        return committed

    def generate(
        self,
        prompt_ids: list[int],
        *,
        max_new_len: int,
        eos_token_id: int | None = None,
        greedy: bool = True,
        generator: torch.Generator | None = None,
    ) -> tuple[list[int], SpecStats]:
        if max_new_len < 1:
            raise ValueError("max_new_len must be positive")

        with torch.inference_mode():
            first = self.prefill(prompt_ids, greedy=greedy, generator=generator)
            generated = [first]
            while len(generated) < max_new_len:
                committed = self.step(greedy=greedy, generator=generator)
                for token in committed:
                    generated.append(token)
                    if eos_token_id is not None and token == eos_token_id:
                        return generated[:max_new_len], self._stats
            return generated[:max_new_len], self._stats


def _pick(logits: torch.Tensor, *, greedy: bool, generator: torch.Generator | None) -> int:
    if greedy:
        return int(torch.argmax(logits).item())
    probs = torch.softmax(logits.detach().float(), dim=-1)
    return int(torch.multinomial(probs, 1, generator=generator).item())
