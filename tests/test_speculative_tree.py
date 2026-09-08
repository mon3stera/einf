"""CPU tests for the tree-speculation skeleton (phase 3b).

The deterministic runner maps each token to a one-hot next-token logit, so
the closed form (t+1, t+2, ...) again anchors the end-to-end bookkeeping.
The tree mask builder, greedy path walk, and gather commit each carry their
own focused invariants; the reference implementations here are the oracles
the optimized 3a versions must match.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import torch

from einf.executors.torch.model_runner import DeterministicModelRunner
from einf.executors.torch.spec_runner import SpeculativeEngine
from einf.executors.torch.spec_tree import (
    TreeSpec,
    TreeSpeculativeEngine,
    build_tree_mask_reference,
    find_greedy_path,
    gather_commit_kv,
    pack_mask_flashinfer,
)

VOCAB = 64


def _tree_spec() -> TreeSpec:
    """Hand-built tree: two root children, two depth-2 nodes under node 0.

    append order: a(0), b(1), c(parent 0), d(parent 0) — depths 1,1,2,2.
    """
    spec = TreeSpec(num_history=10)
    spec.append(None, 40, 0.0)
    spec.append(None, 50, -1.0)
    spec.append(0, 41, -0.5)
    spec.append(0, 42, -2.0)

    return spec


def test_tree_spec_structure():
    spec = _tree_spec()

    assert spec.tokens == [40, 50, 41, 42]
    assert spec.depths == [1, 1, 2, 2]
    assert spec.children(None) == [0, 1]
    assert spec.children(0) == [2, 3]
    assert spec.children(1) == []
    assert spec.ancestors(3) == [0]
    assert spec.ancestors(0) == []
    # logical position = append slot; rope position = depth (siblings share)
    assert [spec.logical_pos(j) for j in range(4)] == [11, 12, 13, 14]
    assert [spec.rope_pos(j) for j in range(4)] == [11, 11, 12, 12]


def test_node_masks_match_ancestor_walk():
    """The O(1) append-time bitmask must agree with the parent-chain walk:
    bit 0 = pending, bit j+1 = node j; a node's word is exactly pending +
    ancestors + self."""
    spec = _tree_spec()
    # node 3's parent is node 0: word = pending | node0 | node3
    assert spec.node_masks[3] == (1 << 0) | (1 << 1) | (1 << 4)
    # root children: pending | self
    assert spec.node_masks[0] == (1 << 0) | (1 << 1)
    assert spec.node_masks[1] == (1 << 0) | (1 << 2)

    for j in range(len(spec.tokens)):
        expected = 1  # pending bit

        for a in spec.ancestors(j):
            expected |= 1 << (a + 1)

        expected |= 1 << (j + 1)
        assert spec.node_masks[j] == expected


def test_tree_budget_cap():
    runner = DeterministicModelRunner(vocab_size=VOCAB)

    try:
        TreeSpeculativeEngine(
            runner, runner, device=torch.device("cpu"), tree_budget=64
        )
    except ValueError as exc:
        assert "63" in str(exc)
    else:
        raise AssertionError("tree_budget 64 must be rejected (int64 sign bit)")


def test_reference_mask_invariants():
    spec = _tree_spec()
    mask = build_tree_mask_reference(spec)
    total_q, total_kv = 5, 15

    assert mask.shape == (total_q, total_kv)
    # everyone sees all history
    assert mask[:, :10].all()
    # pending row: history + self only — never the tree nodes
    assert mask[0, 10] and not mask[0, 11:].any()
    # node rows see pending (col 10), themselves, and ancestors — not siblings
    assert mask[1:, 10].all()
    for j in range(4):
        row = 1 + j

        for other in range(4):
            visible = mask[row, 11 + other]

            if other == j:
                assert visible
            elif other in spec.ancestors(j):
                assert visible
            else:
                assert not visible

    # node 2 and 3 are siblings (both children of node 0): mutually invisible
    assert not mask[1 + 3, 11 + 2]
    assert not mask[1 + 2, 11 + 3]


def test_pack_mask_roundtrip():
    spec = _tree_spec()
    mask = build_tree_mask_reference(spec)
    packed = pack_mask_flashinfer(mask)
    unpacked = np.unpackbits(packed.numpy(), bitorder="little")
    assert torch.from_numpy(unpacked[: mask.numel()].reshape(mask.shape)).equal(mask)


def _row_logit(vocab: int, best: int) -> torch.Tensor:
    row = torch.full((vocab,), float("-inf"))
    row[best] = 0.0

    return row


def test_find_greedy_path_diverges_to_correction():
    """Target argmax leaves the tree at node 0: the walk commits the root
    child, then the correction from node 0's own row."""
    spec = _tree_spec()
    rows = [_row_logit(VOCAB, 40), _row_logit(VOCAB, 63), _row_logit(VOCAB, 888 % VOCAB),
            _row_logit(VOCAB, 777 % VOCAB), _row_logit(VOCAB, 666 % VOCAB)]
    logits = torch.stack(rows)

    path, committed = find_greedy_path(logits, spec)

    assert path == [0]
    # root child 40 accepted; node 0's row predicts 63, not in {41, 42}
    assert committed == [40, 63]


def test_find_greedy_path_full_accept_to_bonus():
    """target agrees at every level: walk root child -> node 2 (leaf) and
    take the leaf's argmax as the bonus."""
    spec = _tree_spec()
    rows = [_row_logit(VOCAB, 40), _row_logit(VOCAB, 41), _row_logit(VOCAB, 55 % VOCAB),
            _row_logit(VOCAB, 56), _row_logit(VOCAB, 57)]
    logits = torch.stack(rows)

    path, committed = find_greedy_path(logits, spec)

    assert path == [0, 2]
    # node 2's row is logits[3]; its argmax 56 is the bonus
    assert committed == [40, 41, 56]


def test_find_greedy_path_root_divergence():
    """No root child matches: correction only, from the pending's row."""
    spec = _tree_spec()
    logits = torch.stack([_row_logit(VOCAB, 42)] + [_row_logit(VOCAB, i) for i in range(4)])

    path, committed = find_greedy_path(logits, spec)

    assert path == []
    assert committed == [42]


def _fake_storage(layers: int = 2, blocks: int = 4, block_len: int = 4) -> SimpleNamespace:
    shape = (layers, blocks * block_len, 2, 3)
    return SimpleNamespace(
        K=torch.zeros(shape),
        V=torch.zeros(shape),
    )


def test_gather_commit_kv_moves_and_is_collision_free():
    """live=4: path node at depth 1 is append index 1 (slot 6 -> 5) and the
    depth-2 node is append index 3 (slot 8 -> 6). The second write targets
    slot 6 only after its content was consumed, and slot 8 is read after
    every earlier write."""
    storage = _fake_storage()
    for layer in range(storage.K.shape[0]):
        for slot in range(16):
            storage.K[layer, slot] = float(slot)
            storage.V[layer, slot] = -float(slot)

    gather_commit_kv(storage, src_slots=[6, 8], dst_slots=[5, 6])

    for layer in range(storage.K.shape[0]):
        assert storage.K[layer, 5].equal(torch.full((2, 3), 6.0))
        assert storage.K[layer, 6].equal(torch.full((2, 3), 8.0))
        assert storage.V[layer, 5].equal(torch.full((2, 3), -6.0))
        assert storage.V[layer, 6].equal(torch.full((2, 3), -8.0))
        # untouched slots stay put
        assert storage.K[layer, 7].equal(torch.full((2, 3), 7.0))


def test_gather_commit_kv_empty_and_identity():
    storage = _fake_storage()
    before = storage.K.clone()

    gather_commit_kv(storage, src_slots=[], dst_slots=[])
    gather_commit_kv(storage, src_slots=[3], dst_slots=[3])

    assert storage.K.equal(before)


def test_tree_engine_matches_plain_loop():
    runner = DeterministicModelRunner(vocab_size=VOCAB)
    prompt = [5, 9, 2]
    max_new = 12

    engine = TreeSpeculativeEngine(
        runner,
        runner,
        device=torch.device("cpu"),
        tree_budget=8,
        branch_factor=2,
        max_depth=4,
    )
    ids, stats = engine.generate(prompt, max_new_len=max_new, greedy=True)

    expected = [(prompt[-1] + 1 + i) % VOCAB for i in range(max_new)]
    assert ids == expected
    # self-drafting one-hot accepts every full path: committed = accepted + steps
    assert stats.committed == stats.accepted + stats.steps
    # the tree explores dead branches, so proposed exceeds accepted
    assert stats.proposed > stats.accepted


def test_tree_engine_long_run_across_pages():
    """block_len=4: the scratch region crosses page boundaries every step and
    the gather moves entries both within and across pages."""
    runner = DeterministicModelRunner(vocab_size=VOCAB)
    engine = TreeSpeculativeEngine(
        runner,
        runner,
        device=torch.device("cpu"),
        block_len=4,
        num_blocks=32,
        tree_budget=6,
        branch_factor=2,
        max_depth=3,
        rank_by_confidence=False,
    )
    ids, _ = engine.generate([0, 1], max_new_len=40, greedy=True)

    assert ids == [(2 + i) % VOCAB for i in range(40)]


def test_tree_engine_static_and_dynamic_agree():
    runner = DeterministicModelRunner(vocab_size=VOCAB)
    prompt = [7, 3]

    static = TreeSpeculativeEngine(
        runner, runner, device=torch.device("cpu"),
        tree_budget=6, branch_factor=2, max_depth=3, rank_by_confidence=False,
    )
    dynamic = TreeSpeculativeEngine(
        runner, runner, device=torch.device("cpu"),
        tree_budget=6, branch_factor=2, max_depth=3, rank_by_confidence=True,
    )
    static_ids, _ = static.generate(list(prompt), max_new_len=10, greedy=True)
    dynamic_ids, _ = dynamic.generate(list(prompt), max_new_len=10, greedy=True)

    assert static_ids == dynamic_ids


def test_tree_engine_records_last_tree():
    runner = DeterministicModelRunner(vocab_size=VOCAB)
    engine = TreeSpeculativeEngine(
        runner, runner, device=torch.device("cpu"),
        tree_budget=4, branch_factor=2, max_depth=2,
    )
    engine.generate([5], max_new_len=2, greedy=True)

    spec, root_row, verify_logits = engine._last_tree

    assert spec is not None and root_row is not None
    # verify rows: pending + every tree node, one distribution each
    assert verify_logits.shape[0] == spec.total_q
    # the tree is non-trivial: at least one depth-2 node exists
    assert max(spec.depths) == 2


def test_tree_engine_sampling_deferred():
    runner = DeterministicModelRunner(vocab_size=VOCAB)
    engine = TreeSpeculativeEngine(
        runner, runner, device=torch.device("cpu"), tree_budget=2, max_depth=1,
    )
    engine.prefill([5], greedy=True)

    try:
        engine.step(greedy=False)
    except NotImplementedError as exc:
        assert "3a" in str(exc)
    else:
        raise AssertionError("sampling verify must be deferred to 3a")


def test_chain_engine_unaffected_by_tree_module():
    """Importing the tree module must not disturb the chain engine."""
    runner = DeterministicModelRunner(vocab_size=VOCAB)
    engine = SpeculativeEngine(runner, runner, device=torch.device("cpu"), num_spec_tokens=3)
    ids, stats = engine.generate([5, 6], max_new_len=8, greedy=True)

    assert ids == [(7 + i) % VOCAB for i in range(8)]
    assert stats.proposed == stats.accepted
