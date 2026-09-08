"""Tree speculative decoding (phase 3): scratch-region tree, masked verify,
gather commit.

Architecture:

- One paged KV cache per model. Tree nodes are appended sequentially into a
  scratch region right after the live context; accepted tokens never move.
- A tree node's LOGICAL position (the block-table addressing key) is its
  append index: ``live_tail + 1 + j`` for node ``j``, with the pending token
  occupying ``live_tail``. Its ROPE position is ``live_tail + depth`` —
  siblings share depth, the tree mask isolates branches, so RoPE by depth is
  correct even though two nodes share a position.
- Visibility during expansion/verify forwards comes entirely from the packed
  tree mask (one attention kernel, one KV source).
- Commit gathers the accepted path's KV to the contiguous positions
  ``live_tail + 1 .. live_tail + L``. This is pure data movement: RoPE was
  applied at write time by depth, and a depth-d path node's content is
  exactly what belongs at ``live_tail + d``. The gather runs in ascending
  depth order, which is collision-free (every source slot is strictly
  greater than every earlier destination slot). Dead branches are never
  cleaned; the next step's scratch overwrites them.
- The pending token is re-processed at the head of every forward (its KV is
  not in the cache), identical to the chain engine's pending semantics.

Confidence (EAGLE-2 style): ``log_conf(child) = log_conf(parent) +
log_softmax(logits_parent)[child]`` — full-vocabulary probabilities, never
renormalized across siblings.

Skeleton scope: this module is CPU-complete (expansion, mask reference,
greedy path finding, gather commit, bookkeeping). GPU-only wiring is
explicitly deferred and marked TODO: FlashInfer ``packed_custom_mask``
planning (the phase-0 probe validated the kernel API) and per-level masks
for the expansion forwards.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch

from einf.executors.torch.input import ModelInput
from einf.executors.torch.spec_runner import SpeculativeEngine, _Track


@dataclass
class TreeSpec:
    """Flat tree built during one speculative step.

    Nodes are stored in append (scratch) order. ``parents[j] == -1`` marks a
    root child (its parent is the pending token). The pending token itself is
    NOT a node — it is re-processed at the head of every forward.

    Each node carries ``node_masks[j]``: a uint64-style visibility bitmask,
    bit 0 = the pending token, bits 1..T = tree nodes (bit j+1 = node j).
    ``append`` maintains it in O(1) — ``mask_child = mask_parent | 1 << (j+1)``
    — so the optimized tree-mask builder is a pure unpack of these words: a
    node's full tree visibility (pending + ancestors + self, never siblings)
    is one integer. Budgets above 63 nodes are rejected at engine
    construction to keep the int64 tensor conversion sign-safe.
    """

    PENDING_BIT = 1

    num_history: int
    parents: list[int] = field(default_factory=list)
    tokens: list[int] = field(default_factory=list)
    depths: list[int] = field(default_factory=list)
    node_log_confs: list[float] = field(default_factory=list)
    node_masks: list[int] = field(default_factory=list)

    def append(self, parent: int | None, token: int, log_conf: float) -> int:
        depth = 1 if parent is None else self.depths[parent] + 1
        j = len(self.tokens)

        if parent is None:
            mask = self.PENDING_BIT | (1 << (j + 1))
        else:
            mask = self.node_masks[parent] | (1 << (j + 1))

        self.parents.append(-1 if parent is None else parent)
        self.tokens.append(token)
        self.depths.append(depth)
        self.node_log_confs.append(log_conf)
        self.node_masks.append(mask)
        return j

    def children(self, j: int | None) -> list[int]:
        if j is None:
            return [i for i, p in enumerate(self.parents) if p == -1]

        return [i for i, p in enumerate(self.parents) if p == j]

    def ancestors(self, j: int) -> list[int]:
        """Chain from the top down to (excluding) node j; empty for root children."""
        chain = []
        p = self.parents[j]

        while p != -1:
            chain.append(p)
            p = self.parents[p]

        return list(reversed(chain))

    def logical_pos(self, j: int) -> int:
        """Block-table addressing position of node j's KV."""
        return self.num_history + 1 + j

    def rope_pos(self, j: int) -> int:
        """RoPE position of node j (by depth; siblings share it)."""
        return self.num_history + self.depths[j]

    @property
    def total_q(self) -> int:
        return 1 + len(self.tokens)

    @property
    def total_kv(self) -> int:
        return self.num_history + self.total_q


def build_tree_mask_reference(spec: TreeSpec) -> torch.Tensor:
    """Correctness oracle for the tree mask; O(T^2), host-side.

    Row 0 is the pending token: it sees all history and itself. Row 1+j is
    node j: history, the pending token, its ancestor chain, and itself —
    never its siblings. Returns a bool tensor ``[total_q, total_kv]``; the
    FlashInfer bit-packing (``np.packbits(flat, bitorder="little")``) happens
    at the attention boundary.

    TODO(3a): replace with the user's optimized builder; this reference stays
    as the oracle the optimized version must match.
    """
    total_q, total_kv = spec.total_q, spec.total_kv
    mask = torch.zeros((total_q, total_kv), dtype=torch.bool)
    mask[:, : spec.num_history] = True
    mask[0, spec.num_history] = True

    for j in range(len(spec.tokens)):
        row = 1 + j
        mask[row, spec.num_history] = True

        for a in spec.ancestors(j):
            mask[row, spec.num_history + 1 + a] = True

        mask[row, spec.num_history + 1 + j] = True

    return mask


def build_tree_mask(spec: TreeSpec) -> torch.Tensor:
    """Vectorized tree-mask builder: unpack each node's O(1) visibility
    bitmask instead of walking ancestor chains. Must stay bit-identical to
    ``build_tree_mask_reference`` (the oracle tests enforce this).

    Row 0 is the pending token (history + itself); row 1+j is node j, whose
    ``node_masks[j]`` encodes pending + ancestors + self as bits 0..T.
    """
    t = len(spec.tokens)
    total_q, total_kv = 1 + t, spec.num_history + 1 + t
    mask = torch.zeros((total_q, total_kv), dtype=torch.bool)

    mask[:, : spec.num_history] = True
    mask[0, spec.num_history] = True

    bits = torch.tensor(spec.node_masks, dtype=torch.int64)
    cols = torch.arange(1, t + 1)
    node_block = (bits.unsqueeze(-1) >> cols) & 1
    mask[1:, spec.num_history + 1 :] = node_block.bool()

    # bit 0 of every node mask is the pending token, already written to the
    # cache before the verify forward, so every node row sees it
    mask[1:, spec.num_history] = True
    return mask


def tree_verify_sampling(
    target_logits: torch.Tensor,
    spec: TreeSpec,
    generator: torch.Generator | None = None,
) -> tuple[list[int], list[int]]:
    """Naive-sampling tree verification (SpecInfer's acknowledged-exact NS).

    At each reached row, sample the next token from the target's own
    distribution; if it names a child, descend (the child's KV is already
    computed); otherwise the sample IS the correction token. A leaf has no
    children, so its row's sample always terminates the walk as the bonus.
    Exactly follows the target's autoregressive distribution by construction;
    each reached child is emitted at its information-theoretic cap p(c).

    Returns ``(path, committed)`` — same contract as ``find_greedy_path``,
    whose argmax walk is the deterministic special case.
    """
    u, path, committed = None, [], []

    while True:
        row = 0 if u is None else u + 1
        s = int(torch.multinomial(torch.softmax(target_logits[row], dim=-1), 1, generator=generator))
        kids = spec.children(u)
        match = next((c for c in kids if spec.tokens[c] == s), None)

        if match is None:
            committed.append(s)
            return path, committed

        path.append(match)
        committed.append(s)
        u = match


def pack_mask_flashinfer(mask: torch.Tensor) -> torch.Tensor:
    """Pack a bool ``[total_q, total_kv]`` mask for FlashInfer's
    ``packed_custom_mask`` (bit-per-entry, little bit order — the phase-0
    probe's validated convention). Returns a uint8 tensor ready for the
    attention boundary."""
    import numpy as np

    bits = np.packbits(mask.detach().cpu().numpy().flatten(), bitorder="little")

    return torch.from_numpy(bits)


def find_greedy_path(
    logits: torch.Tensor, spec: TreeSpec
) -> tuple[list[int], list[int]]:
    """Greedy token-tree walk (reference for the chain accept/reject analog).

    ``logits`` has ``total_q`` rows: row 0 is the pending token's next-token
    distribution (it scores the root children), row ``1 + j`` is node j's
    (it scores j's children). The walk follows the target's argmax through
    the tree; at the first node whose children do not contain it, that
    argmax becomes the correction token. Reaching a leaf makes the leaf's
    argmax the bonus token.

    Returns ``(path, committed)`` — path as node indices in root-to-leaf
    order, committed as token ids including the trailing correction/bonus.

    TODO(3a): the user's tree-verify (sampling / max-expected-path) extends
    this; the greedy walk stays as the pipeline's plumbing and oracle.
    """
    path: list[int] = []
    committed: list[int] = []
    row = logits[0]
    siblings = spec.children(None)

    while True:
        predicted = int(torch.argmax(row).item())
        match = next((c for c in siblings if spec.tokens[c] == predicted), None)

        if match is None:
            committed.append(predicted)
            return path, committed

        path.append(match)
        committed.append(spec.tokens[match])
        kids = spec.children(match)

        if not kids:
            committed.append(int(torch.argmax(logits[1 + match]).item()))
            return path, committed

        row = logits[1 + match]
        siblings = kids


def gather_commit_kv(storage: object, src_slots: list[int], dst_slots: list[int]) -> None:
    """Copy accepted-path KV from scratch slots to their final contiguous slots.

    ``src_slots[d]``/``dst_slots[d]`` are flattened ``block * block_len + off``
    slots for the depth-ordered path. Per layer, one advanced-indexing copy
    per cache; both reads and writes are in ascending depth order, which is
    collision-free (see module docstring). Pure data movement — RoPE content
    is already correct for the destination.

    TODO(3b-GPU): replace with one fused copy kernel (or extend write_slots)
    to avoid 2 * layers small D2D launches.
    """
    if not src_slots:
        return

    src = torch.tensor(src_slots, dtype=torch.long, device=storage.K.device)
    dst = torch.tensor(dst_slots, dtype=torch.long, device=storage.K.device)

    for layer in range(storage.K.shape[0]):
        for cache in (storage.K, storage.V):
            # cache[layer] is [slots, kv_heads, head_dim] — flatten slots only
            flat = cache[layer].view(-1, *cache[layer].shape[1:])
            flat[dst] = flat[src]


class TreeSpeculativeEngine(SpeculativeEngine):
    """Static/dynamic tree speculative decoding on the chain engine's tracks.

    Expansion runs level-wise on the draft (one forward per depth). The
    verify forward processes ``[pending, *tree_nodes]`` in one masked call.
    ``rank_by_confidence=False`` gives the static-tree first milestone; the
    confidence ranking is the EAGLE-2 dynamic selection.
    """

    def __init__(
        self,
        target: object,
        draft: object,
        *,
        device: torch.device,
        block_len: int = 16,
        num_blocks: int = 64,
        tree_budget: int = 8,
        branch_factor: int = 2,
        max_depth: int = 4,
        rank_by_confidence: bool = True,
    ) -> None:
        super().__init__(
            target,
            draft,
            device=device,
            block_len=block_len,
            num_blocks=num_blocks,
            num_spec_tokens=1,
        )
        if tree_budget > 63:
            raise ValueError(
                "tree_budget must be <= 63: node visibility bitmasks pack "
                "the pending token plus every node into one int64 word"
            )
        if tree_budget < 1:
            raise ValueError("tree_budget must be >= 1")
        if branch_factor < 1:
            raise ValueError("branch_factor must be >= 1")
        if max_depth < 1:
            raise ValueError("max_depth must be >= 1")

        self._tree_budget = tree_budget
        self._branch_factor = branch_factor
        self._max_depth = max_depth
        self._rank_by_confidence = rank_by_confidence
        # Last step's (spec, root_row, node_rows) — inspectable by tests and
        # the future sampling verification.
        self._last_tree: tuple[TreeSpec, torch.Tensor, torch.Tensor] | None = None

    # ------------------------------------------------------------------
    # ModelInput construction for tree shapes
    # ------------------------------------------------------------------

    def _make_tree_input(
        self,
        track: _Track,
        token_ids: list[int],
        positions: list[int],
        slots: list[int],
        *,
        context_len: int,
        packed_mask: torch.Tensor | None,
    ) -> ModelInput:
        n = len(token_ids)
        device = self._device

        return ModelInput(
            input_token_ids=torch.tensor(token_ids, dtype=torch.long, device=device),
            position=torch.tensor(positions, dtype=torch.long, device=device),
            slot_mapping=torch.tensor(slots, dtype=torch.long, device=device),
            query_start_loc=torch.tensor([0, n], dtype=torch.long, device=device),
            block_tables=track.block_tables_dev
            if track.block_tables_dev is not None
            else torch.zeros((1, 1), dtype=torch.long),
            context_lens=torch.tensor([context_len], dtype=torch.long, device=device),
            query_start_loc_host=(0, n),
            context_lens_host=(context_len,),
            packed_mask=packed_mask,
        )

    # ------------------------------------------------------------------
    # draft expansion (level-wise)
    # ------------------------------------------------------------------

    def _expand_draft(
        self,
        *,
        greedy: bool,
        generator: torch.Generator | None,
    ) -> tuple[TreeSpec, torch.Tensor, torch.Tensor]:
        """Build the tree on the draft track.

        Returns ``(spec, root_row, draft_rows)`` where ``draft_rows`` is
        ``[1 + T, V]`` aligned with the verify forward's row indexing: row 0
        is the draft distribution at the pending context (it proposed the
        root children), row ``1 + j`` is the draft distribution at node j's
        context (it proposed j's children). Nodes appended at the final
        depth never get forwarded (nothing beyond them can be proposed), so
        their rows are ``-inf`` placeholders — a correct verifier never
        reads them, because a leaf ends the walk with a bonus drawn from
        the target row and no residual.

        Level 1 candidates come from the pending token's row (one q=1
        forward); each later level is one batched forward over that level's
        nodes, whose rows propose the next level's children. Scratch slots
        and RoPE-by-depth positions follow the module docstring.
        """
        draft = self._draft
        live = draft.context_len

        # Both caches take the same scratch footprint: the verify forward
        # writes pending + T nodes into the target cache too.
        for track in (self._draft, self._target):
            capacity = len(track.block_table) * self._block_len

            if live + 1 + self._tree_budget >= capacity:
                raise RuntimeError(
                    f"tree scratch exceeds cache capacity "
                    f"({track is self._draft and 'draft' or 'target'})"
                )

        spec = TreeSpec(num_history=live)
        root_row = None
        level: list[int] = []
        node_rows: dict[int, torch.Tensor] = {}

        for depth in range(1, self._max_depth + 1):
            if depth == 1:
                out = self._draft_forward_q1(draft)
                root_row = out.logits[-1].detach().float()
                parents: list[int | None] = [None]
                rows = [root_row]
            else:
                if not level:
                    break

                tokens = [spec.tokens[j] for j in level]
                positions = [spec.rope_pos(j) for j in level]
                slots = [spec.logical_pos(j) for j in level]
                context = live + 1 + len(spec.tokens)
                model_input = self._make_tree_input(
                    draft,
                    tokens,
                    positions,
                    slots,
                    context_len=context,
                    # TODO(3b-GPU): per-level ancestor masks for the real
                    # attention path; the deterministic CPU runner ignores
                    # attention entirely.
                    packed_mask=None,
                )
                out = draft.forward(model_input)
                parents = level
                rows = [out.logits[i].detach().float() for i in range(len(level))]

                for j, row in zip(level, rows):
                    node_rows[j] = row

            # Collect candidates: top-k children per node in this level.
            candidates: list[tuple[float, int, int | None]] = []

            for parent, row in zip(parents, rows):
                log_probs = torch.log_softmax(row, dim=-1)
                k = min(self._branch_factor, log_probs.numel())
                top = torch.topk(log_probs, k)

                for token, logp in zip(top.indices.tolist(), top.values.tolist()):
                    parent_conf = 0.0 if parent is None else spec.node_log_confs[parent]
                    candidates.append((parent_conf + logp, token, parent))

            if self._rank_by_confidence:
                candidates.sort(key=lambda c: -c[0])

            # Deduplicate tokens within a parent (topk indices are distinct,
            # but keep the guard for future sampling-based proposal paths).
            seen: dict[int | None, set[int]] = {}
            new_level: list[int] = []

            for log_conf, token, parent in candidates:
                if len(spec.tokens) + len(new_level) >= self._tree_budget:
                    break

                used = seen.setdefault(parent, set())

                if token in used:
                    continue

                used.add(token)
                j = spec.append(parent, token, log_conf)
                new_level.append(j)

            level = new_level

        vocab = root_row.numel()
        draft_rows = torch.full((spec.total_q, vocab), float("-inf"))
        draft_rows[0] = root_row

        for j, row in node_rows.items():
            draft_rows[1 + j] = row

        return spec, root_row, draft_rows

    def _draft_forward_q1(self, draft: _Track):
        """q=1 forward processing the draft's pending token at live_tail."""
        model_input = self._make_input(draft, [draft.pending_token])
        return draft.forward(model_input)

    # ------------------------------------------------------------------
    # step
    # ------------------------------------------------------------------

    def step(
        self,
        *,
        greedy: bool = True,
        generator: torch.Generator | None = None,
        **kwargs,
    ) -> list[int]:
        if kwargs:
            raise TypeError(f"unexpected step arguments: {sorted(kwargs)}")

        spec, root_row, draft_rows = self._expand_draft(
            greedy=greedy, generator=generator
        )

        # Verify: [pending, *tree_nodes] in one masked forward.
        target = self._target
        live = target.context_len
        verify_tokens = [target.pending_token, *spec.tokens]
        verify_positions = [live] + [spec.rope_pos(j) for j in range(len(spec.tokens))]
        verify_slots = [live] + [spec.logical_pos(j) for j in range(len(spec.tokens))]
        mask = build_tree_mask_reference(spec)
        out = target.forward(
            self._make_tree_input(
                target,
                verify_tokens,
                verify_positions,
                verify_slots,
                context_len=spec.total_kv,
                # TODO(3b-GPU): pack and hand to FlashInfer plan(); the
                # deterministic CPU runner ignores it.
                packed_mask=mask,
            )
        )
        logits = out.logits.detach().float()

        if greedy:
            path, committed = find_greedy_path(logits, spec)
        else:
            path, committed = tree_verify_sampling(
                logits, spec, generator=generator
            )

        # Gather commit on both caches (skip when the runner owns no storage).
        live_slots = [live + 1 + j for j in path]
        final_slots = [live + d for d in range(1, len(path) + 1)]
        moves = [
            (s, d)
            for s, d in zip(live_slots, final_slots)
            if s != d
        ]

        for track in (self._target, self._draft):
            storage = getattr(track.runner, "cache", None)

            if storage is not None:
                gather_commit_kv(
                    storage,
                    [s for s, _ in moves],
                    [d for _, d in moves],
                )

        num_accepted = len(path)
        self._target.context_len = live + 1 + num_accepted
        self._draft.context_len = live + 1 + num_accepted
        self._target.pending_token = committed[-1]
        self._draft.pending_token = committed[-1]

        self._stats.steps += 1
        self._stats.proposed += len(spec.tokens)
        self._stats.accepted += num_accepted
        self._stats.committed += len(committed)
        self._last_tree = (spec, draft_rows, logits)
        return committed
