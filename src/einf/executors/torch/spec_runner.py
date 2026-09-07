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
    runner: object                      # the QwenModelRunner itself
    block_table: list[int]
    context_len: int                    # tokens with live KV in this cache
    pending_token: int                  # committed token not yet in cache
    proposal_logits: list[torch.Tensor] = field(default_factory=list)
    # Device-resident copy of the block table (constant for the engine's
    # lifetime; the page assignment never changes) and per-query-length
    # persistent input buffers, both created lazily on first use.
    block_tables_dev: torch.Tensor | None = None
    buffers: dict[int, _StepInputBuffers] = field(default_factory=dict)
    csr: _CsrStaging | None = None


@dataclass
class _CsrStaging:
    """Host-pinned FlashInfer CSR staging for one track.

    plan() reads context lengths on the host to compute its launch schedule,
    then copies indptr/last_page_len H2D from the pinned buffers — the same
    layout the ModelInputPool proved out. The engine's identity page table
    makes kv_indices a constant device arange written once at init. The
    indptr/last_page_len staging is fenced: host writes for forward i+1 wait
    on an event recorded after forward i, because plan()'s H2D of forward i
    may still be in flight (the hazard the pool fences as
    _wait_for_previous_copies).
    """

    qo_indptr: torch.Tensor             # pinned int32 [2]
    kv_indptr: torch.Tensor             # pinned int32 [2]
    last_page_len: torch.Tensor         # pinned int32 [1]
    kv_indices_dev: torch.Tensor        # device int32 [pages capacity]
    qo_np: object                       # numpy views of the pinned rows
    kv_np: object
    lpl_np: object


@dataclass
class _StepInputBuffers:
    """Persistent device buffers backing one track's ModelInputs for a fixed
    packed-token count.

    ModelInputs built on these are stream-ordered views: the next forward's
    fill_/copy_ ops execute after the previous forward's kernels on the same
    stream, so in-place reuse is safe — the same contract the ModelInputPool
    documents. Constant fields (query_start_loc) are written once at
    creation; per-step fields are single device-side scalar writes on the
    q=1 hot path, with no allocation and no H2D.
    """

    token: torch.Tensor
    position: torch.Tensor
    slot: torch.Tensor
    query_start_loc: torch.Tensor
    context_lens: torch.Tensor
    is_decode_only: bool


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
            runner=target,
            block_table=list(range(num_blocks)),
            context_len=0,
            pending_token=-1,
        )
        self._draft = _Track(
            forward=draft.forward,
            runner=draft,
            # Reserve the last page: the runner-level decode graph captures
            # with it as the dummy block, and try_replay refuses any block
            # table that touches it. Harmless when the draft runs eager.
            block_table=list(range(num_blocks - 1)),
            context_len=0,
            pending_token=-1,
        )
        self._stats = SpecStats()
        # Device-resident block tables: the page assignment is fixed for the
        # engine's lifetime, so the H2D happens once here instead of once
        # per forward. Built with arange on the device — no host transfer.
        for track in (self._target, self._draft):
            track.block_tables_dev = torch.arange(
                len(track.block_table), dtype=torch.long, device=device
            ).unsqueeze(0)
            # Host-pinned CSR staging: plan() takes indptr/last_page_len from
            # pinned CPU (its H2D is sync-free) and the device arange as
            # kv_indices — the identity page table makes it a true constant.
            pages_cap = len(track.block_table)
            pin = {"pin_memory": device.type == "cuda"}
            qo = torch.zeros(2, dtype=torch.int32, **pin)
            kv = torch.zeros(2, dtype=torch.int32, **pin)
            lpl = torch.zeros(1, dtype=torch.int32, **pin)
            track.csr = _CsrStaging(
                qo_indptr=qo,
                kv_indptr=kv,
                last_page_len=lpl,
                kv_indices_dev=torch.arange(
                    pages_cap, dtype=torch.int32, device=device
                ),
                qo_np=qo.numpy(),
                kv_np=kv.numpy(),
                lpl_np=lpl.numpy(),
            )
        # Fence guarding the pinned CSR staging against in-flight plan() H2D.
        self._csr_fence = torch.cuda.Event() if device.type == "cuda" else None

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
        buf = track.buffers.get(n)
        if buf is None:
            buf = _StepInputBuffers(
                token=torch.empty(n, dtype=torch.long, device=self._device),
                position=torch.empty(n, dtype=torch.long, device=self._device),
                slot=torch.empty(n, dtype=torch.long, device=self._device),
                # Constant per shape: [0, n] never changes for this buffer.
                query_start_loc=torch.tensor(
                    [0, n], dtype=torch.long, device=self._device
                ),
                context_lens=torch.empty(1, dtype=torch.long, device=self._device),
                is_decode_only=n == 1,
            )
            track.buffers[n] = buf

        end = start + n
        # Pinned CSR for plan(): only two host scalars change per forward —
        # the context length and its last-page remainder. The fence first
        # drains any plan() H2D still reading this staging from the previous
        # forward (it almost never fires: plan's copies execute in microseconds).
        csr = track.csr
        if self._csr_fence is not None and not self._csr_fence.query():
            self._csr_fence.synchronize()
        pages = (end + self._block_len - 1) // self._block_len
        csr.qo_np[1] = n
        csr.kv_np[1] = pages
        csr.lpl_np[0] = end - (pages - 1) * self._block_len
        buf.context_lens.fill_(end)
        if n == 1:
            # Hot path (the draft loop): four device-side scalar writes —
            # no allocation, no H2D, no synchronization.
            buf.token.fill_(q_token_ids[0])
            buf.position.fill_(start)
            buf.slot.fill_(
                track.block_table[start // self._block_len] * self._block_len
                + start % self._block_len
            )
        else:
            # Verify shape (one call per step): positions come from a device
            # arange; tokens and slots are one small H2D each.
            buf.position.copy_(torch.arange(start, end, device=self._device))
            buf.slot.copy_(
                torch.tensor(
                    [
                        track.block_table[pos // self._block_len]
                        * self._block_len
                        + pos % self._block_len
                        for pos in range(start, end)
                    ],
                    dtype=torch.long,
                )
            )
            buf.token.copy_(torch.tensor(q_token_ids, dtype=torch.long))
        return ModelInput(
            input_token_ids=buf.token,
            position=buf.position,
            slot_mapping=buf.slot,
            query_start_loc=buf.query_start_loc,
            block_tables=track.block_tables_dev,
            context_lens=buf.context_lens,
            query_start_loc_host=(0, n),
            context_lens_host=(end,),
            is_decode_only=buf.is_decode_only,
            flashinfer_csr=(
                csr.qo_indptr,
                csr.kv_indptr,
                csr.kv_indices_dev[:pages],
                csr.last_page_len,
            ),
        )

    def _fence_csr(self) -> None:
        """Cap the stream after a forward so later CSR staging writes are safe."""
        if self._csr_fence is not None:
            self._csr_fence.record()

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
        self._fence_csr()
        first = _pick(out.logits[-1], greedy=greedy, generator=generator)

        self._target.context_len = len(prompt_ids)
        self._target.pending_token = first

        # The draft primes its cache with the prompt too; its own argmax is
        # discarded because the committed sequence comes from the target.
        draft_out = self._draft.forward(self._make_input(self._draft, list(prompt_ids)))
        self._fence_csr()
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
        try_graph = getattr(draft.runner, "try_decode_cuda_graph", None)
        if try_graph is not None:
            out = try_graph(model_input)
            if out is not None:
                self._fence_csr()
                return ModelOutput(logits=out.logits.clone())
        out = draft.forward(model_input)
        self._fence_csr()
        return out

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
        self._fence_csr()
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
