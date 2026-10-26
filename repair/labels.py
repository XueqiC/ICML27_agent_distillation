"""Sequential public-prefix labels and an atomic full-trajectory learning view."""

from dataclasses import dataclass, field
import csv
from io import StringIO
import re

from sed.storage import canonical_bytes, digest
from sed.thinking import split_thinking
from sed.visibility import visible_observation
from repair.align import align_actions, supervised_positions
from repair.prompts import observation, rationale_prompt, task_text, VERSIONS
from repair.records import JsonRecord
from repair.runtime import CaseStopped
from repair.schemas import parse_label, token_count

INSTANCE = re.compile(r"\b([A-Za-z][A-Za-z_\-]*\s+[0-9]+)\b")


def instance_ids(text):
    # ALFWorld public instance syntax, not a lookup in hidden world state.
    return {m.group(1).lower() for m in INSTANCE.finditer(text)
            if m.group(1).split()[0].lower() not in {"step", "steps", "action", "turn"}}


def prefix_checks(task, initial, prefix, action, label, current_index):
    public = "\n".join([task_text(task), initial, *[
        "\n".join((p["reply"], p["action"], p["observation"])) for p in prefix]])
    available = instance_ids(public)
    action_missing = sorted(instance_ids(action) - available)
    label_missing = sorted(instance_ids(label.reasoning) - available)
    bad_steps = [i for i in label.evidence_steps if not -1 <= i < current_index]
    return {"cited_steps_precede_action": not bad_steps, "invalid_cited_steps": bad_steps,
            "action_instances_public": not action_missing, "unknown_action_instances": action_missing,
            "label_instances_public": not label_missing, "unknown_label_instances": label_missing,
            "teacher_unsupported": label.unsupported,
            "structural_pass": not (action_missing or label_missing or bad_steps or label.unsupported)}


@dataclass(frozen=True)
class LearningEpisode(JsonRecord):
    id_field = "episode_id"
    task_id: str
    repair_id: str
    trajectory_id: str
    task_text: str
    initial_observation: str
    history: list[dict]
    supervised_decision_ids: list[int]  # 8.4 repair-mask, stored alongside full view
    full_trajectory_decision_ids: list[int]
    token_mask: list[dict]
    supervised_target_tokens: int
    source_bucket: str
    checkpoint_hash: str | None
    label_protocol: dict
    prefix_checks: list[dict]
    review_status: str = "pending_human_review"
    view: str = "full_trajectory"
    student_history: list[dict] = field(default_factory=list)
    native_token_mask: list[int] | None = None
    native_input_ids: list[int] | None = None


@dataclass
class LabelOutcome:
    episode: LearningEpisode | None
    status: str
    review_tsv: str
    checks: list[dict] = field(default_factory=list)


def review_sheet(rows):
    stream = StringIO(newline="")
    fields = ["decision_id", "source", "action", "reasoning", "evidence_steps", "prefix_sha256",
              "cited_steps_precede_action", "action_instances_public", "label_instances_public",
              "teacher_unsupported", "structural_pass", "human_supported", "human_notes"]
    writer = csv.DictWriter(stream, fieldnames=fields, delimiter="\t", lineterminator="\n",
                            extrasaction="ignore")
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue()


def reconstruct_labels(repair, *, session, tokenizer=None, tokenizer_id=None,
                       reasoning=True, review_path=None):
    """Never return a partial LearningEpisode. No tokenization loads a model.

    Without a chat tokenizer, ``token_mask`` indexes text segments and token
    counts are provisional. With an injected student chat tokenizer, reuse the
    existing trainer's native episode encoder, including turn-end labels, and
    save the exact packed mask. No tokenizer or model is loaded here.
    """
    rows, checks, history = [], [], []
    original, target = repair.original_trajectory, repair.executed_trajectory
    if repair.repair_status != "VERIFIED" or target is None or target.success is not True:
        raise ValueError("Labels require a confirmed successful repair")
    if target.task_id != session.task["task_id"] or target.max_steps != session.max_steps:
        raise ValueError("Label task/horizon mismatch")
    if any(s["action_parsed"] is None for s in target.steps):
        raise ValueError("Parser feedback needs a separately declared training view")
    if tokenizer is not None and not tokenizer_id:
        raise ValueError("Supply the frozen student tokenizer identity")
    alignment = align_actions(original.executed_actions, target.executed_actions)
    supervised = supervised_positions(alignment)
    first = min(supervised, default=len(target.steps))
    initial = observation(target.initial["visible"])
    if initial != observation(original.initial["visible"]):
        raise ValueError("Initial public history differs")
    status, episode = "COMPLETE", None
    try:
        for index, step in enumerate(target.steps):
            action = step["action_parsed"]
            prompt = rationale_prompt(session.task, initial, history, action)
            prefix_hash = digest(prompt.payload)
            if index < first:
                old = original.steps[index]
                if (old["action_parsed"] != action or old["visible"] != step["visible"] or
                        split_thinking(old["action_raw"])[1]):
                    raise ValueError("Unmodified public prefix is not byte-identical")
                reply = old["action_raw"]
                # Preserved labels still require human review and ID grounding.
                from repair.schemas import ReasoningLabel
                label = ReasoningLabel(reply.partition("\nAction:")[0].removeprefix("Thought: "), [], False)
                source = "original_byte_identical"
            elif reasoning:
                label = session.request(prompt, lambda raw: parse_label(raw, tokenizer))
                reply = "Thought: " + label.reasoning + "\nAction: " + action
                source = "teacher_prefix_label"
            else:
                from repair.schemas import ReasoningLabel
                label = ReasoningLabel("", [], False)
                reply, source = "Action: " + action, "action_only"
            check = prefix_checks(session.task, initial, history, action, label, index)
            row = {"decision_id": index, "source": source, "action": action,
                   "reasoning": label.reasoning, "evidence_steps": label.evidence_steps,
                   "prefix_sha256": prefix_hash, **check, "human_supported": "", "human_notes": ""}
            rows.append(row)
            checks.append({"decision_id": index, "prefix_sha256": prefix_hash, **check})
            session.log("reasoning_label", details=row,
                supervision={"additional_teacher_labeled_decisions": int(source == "teacher_prefix_label")})
            if not check["structural_pass"]:
                status = "UNGROUNDED_TARGET"
                break
            history.append({"action_index": index, "reply": reply, "action": action,
                            "observation": observation(step["visible"])})
        if status == "COMPLETE":
            counts = [token_count(turn["reply"], tokenizer) for turn in history]
            # Teacher labeling omits command lists; student inputs retain the
            # harness's PUBLIC visible-field whitelist, shared with every policy.
            student_history, previous = [], target.initial["visible"]
            for turn, step in zip(history, target.steps):
                student_history.append({"visible": visible_observation(previous), "reply": turn["reply"]})
                previous = step["visible"]
            native = None
            if tokenizer is not None and hasattr(tokenizer, "apply_chat_template"):
                from train.dataset import episode_examples
                training_view = {"schema_version": 2, "task_id": target.task_id, "split": "train", "sealed": False,
                    "initial": {"visible": visible_observation(target.initial["visible"])},
                    "steps": [{"event_id": f"{target.trajectory_id}:{i}", "action_raw": turn["reply"],
                               "visible": visible_observation(step["visible"])}
                              for i, (turn, step) in enumerate(zip(history, target.steps))]}
                native, = episode_examples(training_view, tokenizer, packing="episode")
            episode = LearningEpisode(target.task_id, repair.repair_id, target.trajectory_id,
                task_text(session.task), initial, history, supervised, list(range(len(history))),
                [{"decision_id": i, "assistant": [1] * n,
                  "observation": [0] * token_count(history[i]["observation"], tokenizer)}
                 for i, n in enumerate(counts)], native["supervised_tokens"] if native else sum(counts), "successful_repair",
                original.student_checkpoint, {"template_version": VERSIONS["rationale"],
                    "reasoning": reasoning, "reasoning_cap": 64,
                    "tokenizer": tokenizer_id if tokenizer else "whitespace_fallback",
                    "token_scope": "native_assistant_and_turn_end" if native else "assistant_text_before_chat_template",
                    "chat_template_sha256": digest(getattr(tokenizer, "chat_template", None)) if native else None,
                    "original_prefix": "byte_identical"}, checks,
                student_history=student_history,
                native_token_mask=[int(v != -100) for v in native["labels"]] if native else None,
                native_input_ids=native["input_ids"] if native else None)
    except CaseStopped:
        status = "LABEL_BUDGET_EXHAUSTED" if session.stopped != "format_retry_exhausted" else "LABEL_SCHEMA_FAILED"
    sheet = review_sheet(rows)
    if review_path is not None:
        from pathlib import Path
        Path(review_path).write_text(sheet)
    session.log("label_result", reason=status, details={
        "episode_id": episode.stable_id if episode else None, "review_tsv": sheet,
        "human_review_required": True})
    return LabelOutcome(episode, status, sheet, checks)
