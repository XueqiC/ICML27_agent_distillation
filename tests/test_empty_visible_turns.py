"""Offline pinned-tokenizer regression for empty visible student completions."""
from copy import deepcopy
from contextlib import nullcontext
import json
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")
from test_stage2b_torch import tokenizer, trajectory
from baselines.agentgym.data import training_examples
from baselines.common.conversation import (conversation, encode_continuation, encode_messages,
                                          episode_messages, preference_conversation, sequence_rows)
from baselines.common.fsdp_train import (batch_logps, dpo_objective, sft_microsteps,
                                        sft_objective, train_loop)
from baselines.ipr.pairs import build_pairs, preference_conversation as ipr_conversation
from baselines.leap.data import encode_rows
from train.dataset import EMPTY_THOUGHT, encode_turn, episode_examples, mask_report


def episode(*replies):
    value = trajectory()
    value["steps"] = [dict(value["steps"][0], event_id=f"event-{i}", action_raw=reply)
                      for i, reply in enumerate(replies)]
    return value


def pair_with_empty_student():
    teacher = episode("Action: look", "Action: look")
    student = episode("", "Action: look", "", "Action: look", "")
    return {"task_id": teacher["task_id"], "prompt": episode_messages(teacher)[:2],
            "chosen": episode_messages(teacher)[2:], "rejected": episode_messages(student)[2:]}


@pytest.mark.parametrize("packing", ["turn", "episode"])
def test_teacher_paths_still_refuse_empty_targets(tokenizer, monkeypatch, packing):
    value = episode("Action: look", "")
    messages = episode_messages(value)
    with pytest.raises(ValueError, match="visible teacher reply"):
        episode_examples(value, tokenizer, packing=packing)  # Seed-SFT
    with pytest.raises(ValueError, match="visible teacher reply"):
        conversation(value, tokenizer, source="trusted_teacher_seed", packing=packing)  # Kang/BC
    with pytest.raises(ValueError, match="visible teacher reply"):
        encode_messages(messages, tokenizer, task_id=value["task_id"], packing=packing)
    with pytest.raises(ValueError, match="visible teacher reply"):
        encode_rows([{"messages": messages[:-1], "target": ""}], tokenizer)  # LEAP
    monkeypatch.setattr("baselines.agentgym.data.load_dataset", lambda *a: [
        {"messages": messages, "task_id": value["task_id"], "source": "trusted_teacher_seed"}])
    with pytest.raises(ValueError, match="visible teacher reply"):
        training_examples("fake", {}, tokenizer, packing=packing)
    pair = pair_with_empty_student()
    pair["chosen"] = messages[2:]
    with pytest.raises(ValueError, match="visible teacher reply"):
        preference_conversation(pair, tokenizer)


@pytest.mark.parametrize("packing", ["turn", "episode"])
def test_empty_student_history_preserved_without_any_target(tokenizer, packing):
    value = episode("", "Action: look", "", "Action: look", "")
    messages = episode_messages(value)
    template = tokenizer.chat_template
    example = conversation(value, tokenizer, source="student_exploration", packing=packing)
    assert example["messages"] == messages
    assert example["empty_visible_turns_masked"] == 3 and example["turn_count"] == 5
    assert tokenizer.chat_template == template
    if packing == "turn":
        assert len(sequence_rows(example)) == 2
        for index, row in enumerate(example["turns"]):
            if index in (0, 2, 4):
                assert row["labels"] == [-100] * len(row["input_ids"])
                assert row["assistant_spans"] == [] and row["supervised_tokens"] == 0
                assert row["empty_visible_turns_masked"] == 1
            else:
                prompt = tokenizer.apply_chat_template(messages[:2 + 2 * index], tokenize=False,
                    add_generation_prompt=True, enable_thinking=False)
                ids = tokenizer.encode(prompt, add_special_tokens=False)
                assert prompt.endswith(EMPTY_THOUGHT)
                assert row["input_ids"][:row["prompt_tokens"]] == ids
                assert row["labels"][:len(ids)] == [-100] * len(ids)
                assert "<|turn>model\n<turn|>\n" in prompt
        microsteps = sft_microsteps([example], [0])
        assert len(microsteps) == 2 and sum(w for _, w in microsteps) == pytest.approx(1)
        mask_report(example["turns"], tokenizer)
    else:
        assert [s["turn_index"] for s in example["assistant_spans"]] == [1, 3]
        native = tokenizer.apply_chat_template(messages, tokenize=False,
            add_generation_prompt=False, enable_thinking=False)
        assert example["input_ids"] == tokenizer.encode(native[:-1], add_special_tokens=False)
        assert example["input_ids"][-1] == 106 and example["labels"][-1] == -100
        report = mask_report([example], tokenizer)
        assert report["examples"][0]["prefix_mismatch_turn_indices"] == [1, 3]
    target = tokenizer.encode("Action: look<turn|>", add_special_tokens=False)
    assert example["supervised_tokens"] == 2 * len(target)
    for row in sequence_rows(example):
        assert [v for v in row["labels"] if v != -100] == target * (2 if packing == "episode" else 1)
    with pytest.raises(ValueError, match="no truncation"):
        conversation(value, tokenizer, source="student_exploration", packing=packing, max_length=1)


def test_preference_empty_turns_masked_and_later_prefixes_exact(tokenizer):
    pair = pair_with_empty_student()
    encoded = preference_conversation(pair, tokenizer)
    assert encoded["empty_visible_turns_masked"] == encoded["rejected"]["empty_visible_turns_masked"] == 3
    assert encoded["chosen"]["empty_visible_turns_masked"] == 0
    for index, row in enumerate(encoded["rejected"]["turns"]):
        n = len(encoded["prompt_input_ids"])
        assert row["input_ids"][:n] == encoded["prompt_input_ids"]
        assert row["labels"][:n] == [-100] * n
        assert encoded["prompt_input_ids"] + row["continuation_input_ids"] == row["input_ids"]
        if row["supervised_tokens"]:
            expected = encode_turn(tokenizer, pair["prompt"] + pair["rejected"][:2 * index],
                                   pair["rejected"][2 * index]["content"], max_length=32768)
            assert row["input_ids"] == expected["input_ids"] and row["labels"] == expected["labels"]
    tampered = deepcopy(pair)
    tampered["rejected"][1]["content"] = '{"observation":"room","admissible_commands":[],"facts":"SECRET"}'
    with pytest.raises(ValueError, match="allowlisted"):
        preference_conversation(tampered, tokenizer)


def test_episode_empty_history_before_nonempty_final_reply(tokenizer):
    value = episode("", "Action: look", "", "Action: look")
    packed = conversation(value, tokenizer, source="student_exploration", packing="episode")
    turns = conversation(value, tokenizer, source="student_exploration", packing="turn")
    assert packed["input_ids"] == turns["turns"][-1]["input_ids"]
    assert packed["supervised_tokens"] == turns["supervised_tokens"]
    assert packed["empty_visible_turns_masked"] == 2 and packed["labels"][-1] == 106
    assert mask_report([packed], tokenizer)["examples"][0]["prefix_mismatch_turn_indices"] == [1]


@pytest.mark.parametrize("packing", ["turn", "episode"])
def test_all_empty_student_trajectory_and_continuation_refused(tokenizer, packing):
    with pytest.raises(ValueError, match="Nothing to score"):
        conversation(episode("", ""), tokenizer, source="student_exploration", packing=packing)
    # A scoreable shared prefix must not hide an all-empty continuation.
    messages = episode_messages(episode("Action: look", "", ""))
    with pytest.raises(ValueError, match="Nothing to score"):
        encode_continuation(messages[:4], messages[4:], tokenizer, task_id="train/a/1",
                            source="student_exploration")


@pytest.mark.parametrize("packing", ["turn", "episode"])
def test_agentgym_student_success_encoding_and_provenance(tokenizer, monkeypatch, packing):
    value = episode("", "Action: look", "")
    source = {"task_id": value["task_id"], "episode_id": "student-success", "parent_seed_ids": ["seed"],
              "source": "student_exploration", "reward": 1., "messages": episode_messages(value)}
    monkeypatch.setattr("baselines.agentgym.data.load_dataset", lambda *a: [source])
    example, = training_examples("fake", {}, tokenizer, packing=packing)
    assert example["messages"] == source["messages"] and example["empty_visible_turns_masked"] == 2
    assert example["label_provenance"]["source_episode_id"] == "student-success"
    assert all(t["source"] == "student_exploration" for t in example["label_provenance"]["turns"])
    assert len(sft_microsteps([example], [0])) == 1


@pytest.mark.parametrize("student_chosen", [False, True])
@pytest.mark.parametrize("j", [0, 1])
@pytest.mark.parametrize("empty_content", ["", " \n\t"])
@pytest.mark.parametrize("empty_role", ["assistant", "user"])
def test_ipr_global_and_step_pairs_exclude_empty_messages(student_chosen, j, empty_content, empty_role):
    expert = episode(*(["Action: look"] * 5))
    agent = episode(*(["Action: look"] * (j + 3)))
    messages = episode_messages(agent)
    messages[2 + 2 * j + int(empty_role == "user")]["content"] = empty_content
    ef, af, em, am = (0., 1., .2, .8) if student_chosen else (1., 0., .8, .2)
    pairs, report = build_pairs([
        {"task_id": expert["task_id"], "messages": episode_messages(expert),
         "parent_seed_ids": ["seed"], "final_reward": ef, "mc": {j: em}}], [
        {"task_id": agent["task_id"], "messages": messages,
         "episode_id": "student", "final_reward": af, "mc": am, "j": j}])
    assert pairs == [] and report["P"] == 0 and report["empty"] == 1
    assert report["decisions"][0]["decision"] == "empty"
    assert report["P_by_kind"] == {"trajectory": 0, "entire": 0}
    assert report["agent_chosen"] == report["expert_chosen"] == 0


@pytest.mark.parametrize("student_chosen", [False, True])
@pytest.mark.parametrize("j", [0, 1])
def test_ipr_encoder_handles_empty_student_turns_in_hand_built_pair(tokenizer, student_chosen, j):
    # Encoder robustness only: build_pairs excludes these conversations upstream.
    expert = episode_messages(episode(*(["Action: look"] * 5)))
    agent = episode_messages(episode(*(["Action: look"] * j + ["", "Action: look", ""])))
    cut = 2 + 2 * j
    pair = {"task_id": "train/a/1", "j": j, "kind": "trajectory" if j == 0 else "entire",
            "prompt": expert[:cut], "chosen": agent[cut:] if student_chosen else expert[cut:],
            "rejected": expert[cut:] if student_chosen else agent[cut:],
            "chosen_source": "student_exploration" if student_chosen else "trusted_teacher_seed",
            "rejected_source": "trusted_teacher_seed" if student_chosen else "student_exploration",
            "parent_seed_ids": ["seed"], "agent_episode_id": "student"}
    encoded = ipr_conversation(pair, tokenizer)
    side = "chosen" if student_chosen else "rejected"
    other = "rejected" if student_chosen else "chosen"
    assert encoded["empty_visible_turns_masked"] == encoded[side]["empty_visible_turns_masked"] == 2
    assert encoded[other]["empty_visible_turns_masked"] == 0
    row, = sequence_rows(encoded[side])
    assert row["assistant_spans"][0]["turn_index"] == j + 1
    assert row["input_ids"][:len(encoded["prompt_input_ids"])] == encoded["prompt_input_ids"]
    assert encoded[side]["label_provenance"]["source"] == "student_exploration"
    pair[other][-1]["content"] = ""
    with pytest.raises(ValueError, match="visible teacher reply"):
        ipr_conversation(pair, tokenizer)


class SmallModel(torch.nn.Module):
    """Toy causal model; no pretrained weights or large vocabulary allocation."""
    def __init__(self):
        super().__init__()
        self.embedding = torch.nn.Embedding(128, 4)
        self.output = torch.nn.Linear(4, 128)

    def forward(self, input_ids, attention_mask, use_cache, logits_to_keep):
        hidden = self.embedding(input_ids).cumsum(dim=1).tanh()
        return SimpleNamespace(logits=self.output(hidden[:, logits_to_keep]))

    def save_pretrained(self, path, **kwargs):
        torch.save(self.state_dict(), path / "toy.pt")


def small_ids(group):
    group = deepcopy(group)
    for row in group.get("turns", [group]):
        row["input_ids"] = [v % 128 for v in row["input_ids"]]
        row["labels"] = [v % 128 if v != -100 else v for v in row["labels"]]
    return group


@pytest.mark.parametrize("objective", ["sft", "dpo", "ipr"])
def test_masked_turns_never_scored_and_stage_counts_not_epoch_multiplied(tokenizer, tmp_path, objective):
    torch.set_num_threads(1)
    torch.manual_seed(3)
    encoded = preference_conversation(pair_with_empty_student(), tokenizer)
    pair = {side: small_ids(encoded[side]) for side in ("chosen", "rejected")}
    student = pair["rejected"]
    model = SmallModel()
    reference = deepcopy(model).requires_grad_(False)
    expected_score = sum(batch_logps(model, [row], "cpu")[0] for row in sequence_rows(student))
    torch.testing.assert_close(batch_logps(model, [student], "cpu")[0], expected_score)
    if objective == "sft":
        loss, _ = sft_objective(model, student, "cpu")
        torch.testing.assert_close(loss, -expected_score / student["supervised_tokens"])
        examples = [student, pair["chosen"]]
    else:
        loss, metrics = dpo_objective(model, reference, pair, "cpu", beta=.1,
                                      nll_weight=float(objective == "ipr"))
        torch.testing.assert_close(metrics["logps/rejected"], expected_score.detach())
        assert torch.isfinite(loss)
        examples = [pair, {"chosen": pair["chosen"], "rejected": pair["chosen"]}]
    state = train_loop(model, examples, tmp_path, objective=objective, reference=reference,
        learning_rate=1e-5, scheduler="constant_with_warmup", warmup_ratio=0,
        original_batch=48, epochs=2, packing="turn")
    assert state["empty_visible_turns_masked"] == 3
    assert state["empty_visible_turn_examples"] == 1 and state["examples_seen"] == 4
    expected_tokens = sum(e["supervised_tokens"] for e in examples) if objective == "sft" else sum(
        e[s]["supervised_tokens"] for e in examples for s in ("chosen", "rejected"))
    assert state["supervised_tokens_seen"] == 2 * expected_tokens


def test_agentgym_dose_checkpoint_counts_and_mask_report(tokenizer, tmp_path):
    from train.sft import exposure_plan, train_loop as dose_train_loop
    torch.set_num_threads(1)
    example = small_ids(conversation(episode("", "Action: look", ""), tokenizer,
                                    source="student_exploration", packing="episode"))
    plan = exposure_plan([example["supervised_tokens"]], passes=2, packing="episode",
                         turn_counts=[example["turn_count"]])
    checkpoints = dose_train_loop(SmallModel(), [example], plan, tmp_path, identity_sha256="unit")
    state = json.loads((tmp_path / checkpoints[-1]["path"] / "training_state.json").read_text())
    assert state["empty_visible_turns_masked"] == 2 and state["empty_visible_turn_examples"] == 1
    assert state["examples_seen"] == 2


@pytest.mark.parametrize("packing", ["turn", "episode"])
@pytest.mark.parametrize("empty", [False, True])
def test_adftp_empty_teacher_reply_history_labels_and_trainer_provenance(tokenizer, tmp_path, monkeypatch, packing, empty):
    from baselines.adftp.pipeline import training_command
    from baselines.common import fsdp_train
    from sed.config import load_config
    from sed.storage import write_json

    torch.set_num_threads(1)
    value = episode("Thought: prefix\nAction: look", "" if empty else "Action: look",
                    "Malformed non-empty reply", "Action: look")
    value["steps"][1]["action_outcome"] = "parser_error" if empty else "legal"
    value["steps"][1]["visible"] = {"observation": "Observation after the empty reply", "admissible_commands": ["look"]}
    value["steps"][2]["action_outcome"] = "parser_error"
    config = load_config()
    collection = {"K": 1, "unique_instances": 1, "task_groups": 1, "verified": [{}]}
    monkeypatch.setenv("WORLD_SIZE", "1")
    monkeypatch.setenv("RANK", "0")
    monkeypatch.setenv("LOCAL_RANK", "0")
    monkeypatch.setattr(fsdp_train, "load_config", lambda *a: config)
    monkeypatch.setattr(fsdp_train, "load_frozen", lambda *a: ({}, {}))
    monkeypatch.setattr(fsdp_train, "load_verified_collection", lambda *a: (collection, {}, tmp_path))
    monkeypatch.setattr(fsdp_train, "checked_episodes", lambda *a: [value])
    monkeypatch.setattr(fsdp_train, "load_tokenizer", lambda *a: tokenizer)
    monkeypatch.setattr(fsdp_train, "snapshot_metadata", lambda *a: {"source": "fake"})
    monkeypatch.setattr(fsdp_train, "artifact_hashes", lambda *a: {})
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "set_device", lambda *a: None)
    monkeypatch.setattr(torch, "autocast", lambda *a, **kw: nullcontext())
    monkeypatch.setattr("transformers.set_seed", lambda *a: None)

    def model(*args, **kwargs):
        # Only the production GPU gate is stubbed; all updates run on a toy CPU model.
        monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
        result = SmallModel()
        result.peft_config = {"default": SimpleNamespace(r=64, lora_alpha=128, lora_dropout=.05,
                                                         target_modules=["fake"])}
        return result

    monkeypatch.setattr(fsdp_train, "prepare_lora_model", model)
    def provenance(output, **kwargs):
        record = {"config": kwargs["config"]}
        write_json(output / "provenance.json", record)
        return record
    monkeypatch.setattr(fsdp_train, "write_provenance", provenance)

    def cpu_train(model, examples, output, **kwargs):
        example, = examples
        assert example["messages"] == episode_messages(value)
        mask_report(example["turns"] if packing == "turn" else [example], tokenizer)
        if packing == "turn":
            assert len(sequence_rows(example)) == 4 - int(empty)
            if empty:
                row = example["turns"][1]
                assert row["labels"] == [-100] * len(row["input_ids"])
            rows = [example["turns"][2]]
        else:
            assert [s["turn_index"] for s in example["assistant_spans"]] == ([0, 2, 3] if empty else [0, 1, 2, 3])
            rows = [example]
        assert "Malformed non-empty reply" in tokenizer.decode([v for row in rows for v in row["labels"] if v != -100])
        final = sequence_rows(example)[-1]
        prompt = tokenizer.decode(final["input_ids"], skip_special_tokens=False)
        assert "Observation after the empty reply" in prompt
        if empty:
            assert "<|turn>model\n<turn|>\n" in prompt
        assert "SECRET" not in prompt and "PRIVATE_PDDL" not in prompt
        return train_loop(model, [small_ids(example)], output, **{**kwargs, "device": "cpu"})

    monkeypatch.setattr(fsdp_train, "train_loop", cpu_train)
    output = tmp_path / "run"
    command = training_command(SimpleNamespace(train_python="python", config="fake", snapshot="fake",
                                               verified="fake"), output)
    command[command.index("--packing") + 1] = packing
    fsdp_train.main(command[command.index("--config"):])
    training = json.loads((output / "provenance.json").read_text())["config"]["baseline_training"]
    result = json.loads((output / "result.json").read_text())
    for record in (training, result):
        assert record["empty_visible_turns_masked"] == int(empty)
        assert record["masked_empty_replies"] == ([{"task_id": value["task_id"], "step_index": 1}] if empty else [])
    assert training["allow_empty_visible_replies"] is True
    assert result["examples_seen"] == 2  # Mask count is per dataset, not multiplied by epochs.
