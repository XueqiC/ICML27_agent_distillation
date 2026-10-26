import pytest

from repair.align import (Alignment, AlignmentStep, align_actions, canonical_action, levenshtein_distance,
                          normalized_distance, supervised_positions)
from repair.patch import ExtendedRepair, InvalidPatch, Patch, PatchBlock, compile_patch


def block(name, op, **kwargs):
    return PatchBlock(name, op, **kwargs)


def test_noncontiguous_original_indices_do_not_drift():
    patch = Patch("original", [
        block("late", "replace", start=4, end_exclusive=5, actions=["D", "E"]),
        block("insert", "insert_before", index=1, actions=["X", "Y"]),
        block("early", "replace", start=1, end_exclusive=3, actions=["B"]),
        block("append", "insert_before", index=5, actions=["Z"]),
    ])
    original = ["a", "b", "c", "d", "e"]
    assert compile_patch(original, patch) == ["a", "X", "Y", "B", "d", "D", "E", "Z"]
    assert original == ["a", "b", "c", "d", "e"]
    assert Patch.from_json(patch.to_json()) == patch


def test_adjacent_ranges_left_boundary_and_delete_insert_reorder():
    patch = Patch("o", [block("d", "delete", start=0, end_exclusive=1),
                        block("r", "replace", start=1, end_exclusive=2, actions=["B"]),
                        block("i", "insert_before", index=2, actions=["a"])])
    assert compile_patch(["a", "b", "c"], patch) == ["B", "a", "c"]
    patch = Patch("o", [block("d", "delete", start=0, end_exclusive=2),
                        block("i", "insert_before", index=0, actions=["first"])])
    assert compile_patch(["a", "b", "c"], patch) == ["first", "c"]
    assert compile_patch([], Patch("o", [block("i", "insert_before", index=0, actions=["a"])])) == ["a"]
    assert compile_patch(["a"], Patch("o", [block("d", "delete", start=0, end_exclusive=1)])) == []


@pytest.mark.parametrize("edit,reason", [
    (block("i", "insert_before", index=0), "insert_without_actions"),
    (block("r", "replace", start=0, end_exclusive=1), "replace_without_actions"),
])
def test_insert_and_replace_require_actions(edit, reason):
    with pytest.raises(InvalidPatch, match=f"^{reason}$") as error:
        compile_patch(["a"], Patch("o", [edit]))
    assert error.value.reason == reason


@pytest.mark.parametrize("blocks,reason", [
    ([block("a", "replace", start=0, end_exclusive=2, actions=["x"]),
      block("b", "delete", start=1, end_exclusive=3)], "overlapping"),
    ([block("a", "replace", start=0, end_exclusive=2, actions=["x"]),
      block("b", "insert_before", index=1, actions=["x"])], "inside_range"),
    ([block("a", "delete", start=0, end_exclusive=2),
      block("b", "insert_before", index=1, actions=["x"])], "inside_range"),
    ([block("a", "insert_before", index=0, actions=["x"]),
      block("b", "insert_before", index=0, actions=["y"])], "duplicate_insert"),
    ([block("a", "insert_before", index=4, actions=["x"])], "out_of_bounds"),
    ([block("a", "insert_before", index=-1, actions=["x"])], "out_of_bounds"),
    ([block("a", "insert_before", index=True, actions=["x"])], "out_of_bounds"),
    ([block("a", "delete", start=1, end_exclusive=1)], "empty"),
    ([block("a", "delete", start=0, end_exclusive=4)], "out_of_bounds"),
    ([block("a", "delete", start=0, end_exclusive=1, actions=["x"])], "delete_has_actions"),
    ([block("a", "replace", start=0, end_exclusive=1, index=0, actions=["x"])], "range_has"),
    ([block("a", "reorder", actions=["b", "a"])], "unsupported"),
    ([block("a", ["not", "an", "operation"])], "unsupported"),
    ([block("a", "insert_before", index=0, actions=["x"]),
      block("a", "delete", start=0, end_exclusive=1)], "block_id"),
])
def test_rejects_invalid_patches_without_guessing(blocks, reason):
    with pytest.raises(InvalidPatch, match=reason) as error:
        compile_patch(["a", "b", "c"], Patch("o", blocks))
    assert error.value.reason


def test_extended_repair_and_unknown_fields_are_not_default_candidates():
    with pytest.raises(ExtendedRepair):
        compile_patch(["a"] * 6, Patch("o", [block(str(i), "delete", start=i, end_exclusive=i + 1)
                                                 for i in range(5)]))
    with pytest.raises(InvalidPatch):
        Patch.from_json({"original_trajectory_id": "o", "blocks": [], "observation": "fabricated"})
    with pytest.raises(InvalidPatch):
        Patch.from_json({"original_trajectory_id": "o", "blocks": [{"block_id": "p", "op": "delete",
                                                                      "hidden_state": "forbidden"}]})


@pytest.mark.parametrize("a,b,d", [([], [], 0), ([], ["a", "b"], 2), (["a"], [], 1),
    (["a", "b", "c"], ["a", "x", "c"], 1), (["a", "b"], ["b", "a"], 2),
    (["a", "b", "c", "d"], ["b", "c", "e"], 2),
    (["a", "b", "c"], ["x", "a", "b", "y", "c"], 2)])
def test_hand_computed_distances(a, b, d):
    assert levenshtein_distance(a, b) == d
    assert normalized_distance(a, b) == d / max(1, len(a), len(b))
    assert levenshtein_distance(b, a) == d


def test_canonicalization_retains_instance_identity_and_uses_shared_parser():
    assert canonical_action("Thought: public plan\nAction: take  apple 1") == "take apple 1"
    assert levenshtein_distance(["take apple 1"], ["take apple 2"]) == 1
    assert levenshtein_distance(["Action: take  apple 1"], ["take apple 1"]) == 0


def test_alignment_tie_order_is_forward_keep_replace_delete_insert():
    # Equal repeated actions keep the earliest match.
    assert [s.op for s in align_actions(["a", "a"], ["a"]).path] == ["keep", "delete"]
    assert [s.op for s in align_actions(["a", "b"], ["b", "a"]).path] == ["replace", "replace"]
    # Delete and insert starts both cost two; replacing first costs three.
    assert [s.op for s in align_actions(["a", "b", "a"], ["b", "a", "b"]).path] == [
        "delete", "keep", "keep", "insert"]


def test_delete_supervises_next_keep_and_tail_creates_no_target():
    alignment = align_actions(["take key", "bad", "open door", "finish", "tail"],
                              ["take key", "open door", "finish"])
    assert supervised_positions(alignment) == [1]
    alignment = align_actions(["a", "b", "c"], ["x", "a", "y", "c"])
    assert supervised_positions(alignment) == [0, 2]
    assert supervised_positions(align_actions(["a", "b"], [])) == []


@pytest.mark.parametrize("first_op,original_index", [("keep", 1), ("insert", None), ("replace", 1)])
def test_delete_supervises_only_the_first_emitted_action(first_op, original_index):
    # Explicit paths isolate masking from the DP's preference for a replacement
    # over an adjacent delete/insert pair.
    alignment = Alignment(0, 0.0, (
        AlignmentStep("delete", 0, None),
        AlignmentStep(first_op, original_index, 0),
        AlignmentStep("keep", 1 if first_op == "insert" else 2, 1),
    ))
    assert supervised_positions(alignment) == [0]


def test_trailing_delete_after_success_adds_no_supervised_action():
    successful_actions = ["take key", "open door", "finish"]
    alignment = align_actions([*successful_actions, "inspect"], successful_actions)
    assert [step.op for step in alignment.path] == ["keep", "keep", "keep", "delete"]
    assert supervised_positions(alignment) == []
