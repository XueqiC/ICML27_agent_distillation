"""Unit-cost action alignment with deterministic forward-path tie breaking."""

from dataclasses import asdict, dataclass

from sed.actions import ActionParseError, parse_action


def action_reply(action):
    """Accept commands or recorded replies through the existing ReAct parser."""
    if type(action) is not str:
        raise TypeError("Action must be text")
    try:
        parse_action(action)
        return action
    except ActionParseError:
        # Same command-to-reply convention as ScriptedPolicy; multiline replies
        # remain intact, including malformed replies and their parser feedback.
        return "Action: " + action if "\n" not in action else action


def canonical_action(action):
    """Normalize whitespace only; preserve case and concrete instance numbers."""
    try:
        value = parse_action(action_reply(action))
    except ActionParseError:
        value = action
    return " ".join(value.split())


@dataclass(frozen=True)
class AlignmentStep:
    op: str
    original_index: int | None
    corrected_index: int | None


@dataclass(frozen=True)
class Alignment:
    distance: int
    normalized_distance: float
    path: tuple[AlignmentStep, ...]

    def to_json(self):
        return {"distance": self.distance, "normalized_distance": self.normalized_distance,
                "path": [asdict(step) for step in self.path]}

    @classmethod
    def from_json(cls, value):
        return cls(value["distance"], value["normalized_distance"],
                   tuple(AlignmentStep(**step) for step in value["path"]))


def align_actions(original_actions, corrected_actions):
    """Tie order at each forward decision: keep, replace, delete, insert."""
    a, b = [canonical_action(x) for x in original_actions], [
        canonical_action(x) for x in corrected_actions]
    n, m = len(a), len(b)
    # Suffix DP makes the tie order apply to the saved path from left to right,
    # rather than inadvertently reversing it during traceback.
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n, -1, -1):
        for j in range(m, -1, -1):
            if i == n:
                dp[i][j] = m - j
            elif j == m:
                dp[i][j] = n - i
            else:
                dp[i][j] = min(int(a[i] != b[j]) + dp[i + 1][j + 1],
                               1 + dp[i + 1][j], 1 + dp[i][j + 1])
    path, i, j = [], 0, 0
    while i < n or j < m:
        if i < n and j < m and a[i] == b[j] and dp[i][j] == dp[i + 1][j + 1]:
            path.append(AlignmentStep("keep", i, j))
            i, j = i + 1, j + 1
        elif i < n and j < m and dp[i][j] == 1 + dp[i + 1][j + 1]:
            path.append(AlignmentStep("replace", i, j))
            i, j = i + 1, j + 1
        elif i < n and dp[i][j] == 1 + dp[i + 1][j]:
            path.append(AlignmentStep("delete", i, None))
            i += 1
        else:
            path.append(AlignmentStep("insert", None, j))
            j += 1
    return Alignment(dp[0][0], dp[0][0] / max(1, n, m), tuple(path))


def levenshtein_distance(original_actions, corrected_actions):
    return align_actions(original_actions, corrected_actions).distance


def normalized_distance(original_actions, corrected_actions):
    return align_actions(original_actions, corrected_actions).normalized_distance


def supervised_positions(alignment):
    """Corrected decision indices, including the first action after any deletion."""
    positions, pending_delete = set(), False
    for step in alignment.path:
        if step.op == "delete":
            pending_delete = True
        elif step.op in {"insert", "replace"}:
            positions.add(step.corrected_index)
            pending_delete = False
        elif step.op == "keep" and pending_delete:
            positions.add(step.corrected_index)
            pending_delete = False
    # A trailing deletion has no next action; no terminal label is invented.
    return sorted(positions)


def supervised_decisions(original_actions, trajectory):
    """Map alignment positions to actual turns and freshly executed public prefixes.

    Parser feedback turns are context, not environment actions. Explicit indices
    prevent those zero-step turns from shifting a later action's target prefix.
    """
    step_indices = [i for i, s in enumerate(trajectory.steps) if s.get("action_parsed") is not None
                    and s["visible"] is not None and s.get("error_type") is None]
    alignment = align_actions(original_actions, trajectory.executed_actions)
    return [{"corrected_action_index": i, "decision_index": step_indices[i],
             **trajectory.decision_context(step_indices[i]),
             "action_target": trajectory.steps[step_indices[i]]["action_parsed"]}
            for i in supervised_positions(alignment)]
