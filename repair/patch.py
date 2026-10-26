"""Edits refer exclusively to boundaries in the original action sequence."""

from dataclasses import asdict, dataclass

from sed.storage import digest


class InvalidPatch(ValueError):
    def __init__(self, reason):
        self.reason = reason
        super().__init__(reason)


class ExtendedRepair(InvalidPatch):
    """More than four blocks belongs to a separately reported variant."""


@dataclass(frozen=True)
class PatchBlock:
    block_id: str
    op: str
    actions: tuple[str, ...] = ()
    index: int | None = None
    start: int | None = None
    end_exclusive: int | None = None

    def __post_init__(self):
        if not isinstance(self.actions, (tuple, list)):
            raise InvalidPatch("actions_must_be_a_sequence")
        object.__setattr__(self, "actions", tuple(self.actions))


@dataclass(frozen=True)
class Patch:
    original_trajectory_id: str
    blocks: tuple[PatchBlock, ...] = ()

    def __post_init__(self):
        if not isinstance(self.blocks, (tuple, list)):
            raise InvalidPatch("blocks_must_be_a_sequence")
        object.__setattr__(self, "blocks", tuple(self.blocks))

    @property
    def patch_id(self):
        return "patch_" + digest(asdict(self))

    def to_json(self):
        return {"patch_id": self.patch_id, "original_trajectory_id": self.original_trajectory_id,
                "blocks": [dict(asdict(b), actions=list(b.actions)) for b in self.blocks]}

    @classmethod
    def from_json(cls, value):
        try:
            if set(value) - {"patch_id", "original_trajectory_id", "blocks"}:
                raise InvalidPatch("unknown_patch_fields")
            patch = cls(value["original_trajectory_id"],
                        tuple(PatchBlock(**b) for b in value["blocks"]))
            if "patch_id" in value and value["patch_id"] != patch.patch_id:
                raise InvalidPatch("patch_id_mismatch")
            return patch
        except (KeyError, TypeError):
            raise InvalidPatch("invalid_patch_schema") from None


def compile_patch(original_actions, patch):
    """Validate without repairing indices, then compile in original-index order."""
    if not isinstance(patch, Patch):
        raise InvalidPatch("expected_patch")
    if len(patch.blocks) > 4:
        raise ExtendedRepair("extended_repair_more_than_four_blocks")
    if not isinstance(original_actions, (list, tuple)) or any(
            type(a) is not str for a in original_actions):
        raise InvalidPatch("original_actions_must_be_strings")
    if type(patch.original_trajectory_id) is not str or not patch.original_trajectory_id:
        raise InvalidPatch("missing_original_trajectory_id")
    n, inserts, spans, ids = len(original_actions), {}, [], set()
    for block in patch.blocks:
        if not isinstance(block, PatchBlock):
            raise InvalidPatch("expected_patch_block")
        if type(block.block_id) is not str or not block.block_id or block.block_id in ids:
            raise InvalidPatch("missing_or_duplicate_block_id")
        ids.add(block.block_id)
        if type(block.op) is not str:
            raise InvalidPatch("unsupported_operation")
        if any(type(a) is not str or not a.strip() for a in block.actions):
            raise InvalidPatch("actions_must_be_nonempty_strings")
        if block.op == "insert_before":
            if not block.actions:
                raise InvalidPatch("insert_without_actions")
            if block.start is not None or block.end_exclusive is not None:
                raise InvalidPatch("insert_has_range")
            if type(block.index) is not int or not 0 <= block.index <= n:
                raise InvalidPatch("insert_index_out_of_bounds")
            if block.index in inserts:
                raise InvalidPatch("duplicate_insert_boundary")
            inserts[block.index] = block
        elif block.op in {"replace", "delete"}:
            if block.index is not None:
                raise InvalidPatch("range_has_insert_index")
            if (type(block.start) is not int or type(block.end_exclusive) is not int
                    or not 0 <= block.start < block.end_exclusive <= n):
                raise InvalidPatch("range_out_of_bounds_or_empty")
            if block.op == "delete" and block.actions:
                raise InvalidPatch("delete_has_actions")
            if block.op == "replace" and not block.actions:
                raise InvalidPatch("replace_without_actions")
            spans.append(block)
        else:
            raise InvalidPatch("unsupported_operation")
    spans.sort(key=lambda b: b.start)
    if any(left.end_exclusive > right.start for left, right in zip(spans, spans[1:])):
        raise InvalidPatch("overlapping_ranges")
    if any(b.start < i < b.end_exclusive for b in spans for i in inserts):
        raise InvalidPatch("insert_inside_range")
    # A left boundary is outside the open interior, including for a delete.
    by_start = {b.start: b for b in spans}
    result, index = [], 0
    while index <= n:
        if index in inserts:
            result.extend(inserts[index].actions)
        if index == n:
            break
        block = by_start.get(index)
        if block is not None:
            result.extend(block.actions)
            index = block.end_exclusive
        else:
            result.append(original_actions[index])
            index += 1
    return result
