"""SAD span/order/dose contracts independent of any installed model or GPU."""
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from baselines.sad.data import encode_episode, curriculum, token_plan, ordered_batches, TOKEN_BUDGET
from baselines.sad.checkpoints import publisher
from baselines.sad.pipeline import training_command
from train.dataset import TURN_TOKEN, EMPTY_THOUGHT
from sed.storage import digest, write_json
from sed.seed_evidence import artifact_hashes


class CharTokenizer:
    all_special_ids = [106]
    chat_template = "test-only-Gemma-OFF-contract"

    def __call__(self, text, **kwargs):
        ids, offsets, i = [], [], 0
        while i < len(text):
            end = i + (len(TURN_TOKEN) if text.startswith(TURN_TOKEN, i) else 1)
            ids.append(106 if text.startswith(TURN_TOKEN, i) else ord(text[i]) + 200)
            offsets.append((i, end))
            i = end
        return {"input_ids": ids, "offset_mapping": offsets}

    def encode(self, text, **kwargs):
        return self(text)["input_ids"]

    def decode(self, ids, **kwargs):
        return ''.join(TURN_TOKEN if i == 106 else chr(i - 200) for i in ids)

    def apply_chat_template(self, messages, *, add_generation_prompt, **kwargs):
        text = ''.join(m['role'] + '\n' + m['content'].strip() + (TURN_TOKEN + '\n' if m['role'] == 'assistant' else '\n')
                       for m in messages)
        return text + ('assistant\n' + EMPTY_THOUGHT if add_generation_prompt else '')


def episode(reply):
    visible = {"observation": "OBSERVATION", "admissible_commands": ["look"], "facts": "HIDDEN"}
    return {"schema_version": 2, "task_id": "train/a", "split": "train", "sealed": False,
            "initial": {"visible": visible}, "steps": [{"visible": visible, "event_id": "e", "action_raw": reply}]}


def test_sad_masks_only_reason_action_turn_end_and_curriculum():
    tokenizer = CharTokenizer()
    reply = "Unlabelled introduction\nThought: café reason\ncontinued reasoning\nNote: MASK_THIS\nAction: look\ntrailing text"
    row = encode_episode(episode(reply), tokenizer)
    turn, = row["turns"]
    decoded = tokenizer.decode([x for x in turn["labels"] if x != -100])
    assert decoded == 'Thought: café reason\ncontinued reasoning\nAction: look<turn|>'
    assert all(v == -100 for v in turn["labels"][:turn["prompt_tokens"]])
    assert 'HIDDEN' not in tokenizer.decode(turn["input_ids"])
    assert row["complexity"] == row["reason_tokens"] + row["action_tokens"] == len(decoded) - len(TURN_TOKEN) + 1
    short = encode_episode(episode('Action: look'), tokenizer)
    assert short["reason_tokens"] == 0
    assert curriculum([row, short, dict(short, tag="tie")])[0:2] == [short, dict(short, tag="tie")]
    with pytest.raises(ValueError, match="Action line"):
        encode_episode(episode('Thought: no action'), tokenizer)
    with pytest.raises(ValueError, match="no truncation"):
        encode_episode(episode(reply), tokenizer, max_length=1)
    sealed = episode('Action: look'); sealed["split"] = "test"
    with pytest.raises(ValueError):
        encode_episode(sealed, tokenizer)


@pytest.mark.parametrize('n', [1, 5, 8, 20, 100, 103])
@pytest.mark.parametrize('budget', [1, 37, 399, 1000])
def test_token_budget_closes_whole_batches_and_repeats_order(n, budget):
    examples = [{"input_ids": list(range(i + 3)), "labels": [-100, 1]} for i in range(n)]
    plan = token_plan(examples, token_budget=budget)
    batches = list(ordered_batches(n, plan["b_prime"], plan["total_steps"]))
    costs = [sum(len(examples[i]["input_ids"]) for i in b["indices"]) for b in batches]
    assert sum(costs) == plan["planned_processed_tokens"] >= budget
    assert sum(costs[:-1]) < budget
    flat = [i for b in batches for i in b["indices"]]
    assert flat == (list(range(n)) * plan["epochs"])[:len(flat)]
    assert plan["b_prime"] == min(64, max(1, n // 8))
    assert plan["checkpoint_tokens"][-1] == budget
    assert TOKEN_BUDGET == 655_000_000


def test_all_crossings_publish_even_when_one_batch_crosses_multiple(tmp_path):
    output = tmp_path / 'run/stages/train-sad/attempt-000/output'
    plan = token_plan([{"input_ids": [0, 1], "labels": [-100, 1]}], token_budget=160)
    saves = []
    def export(model, path, base, rank):
        saves.append(path)
        write_json(path / 'weights.json', {"value": model})
        return {"artifact_sha256": artifact_hashes(path), "text_export_dtype": "bfloat16"}
    callback = publisher(output, 'base', {"source": "test"}, plan, export=export)
    callback(7, {"processed_token_positions": 21})
    assert len(saves) == 2
    callback(9, {"processed_token_positions": 161})
    models = json.loads((output / 'models.json').read_text())
    assert [m['fraction'] for m in models] == ['1/16', '1/8', '1/4', '1/2', '1/1']
    assert json.loads((tmp_path / 'run/models.json').read_text()) == models
    for m in models:
        state = json.loads((Path(m['checkpoint']) / 'training_state.json').read_text())
        assert state['final'] and state['provenance_sha256'] == digest({"source": "test"})
        assert state['budget_complete'] is (m['fraction'] == '1/1')
        assert state['threshold_overshoot'] >= 0
    args = SimpleNamespace(train_python='TRAIN', gpus=4, config='CONFIG', snapshot='BASE', verified='VERIFIED')
    command = training_command(args, output)
    assert command[command.index('--objective') + 1] == 'sad'
    assert command[command.index('--processed-token-budget') + 1] == str(TOKEN_BUDGET)
    assert '--reference' not in command


def test_full_checkpoint_equivalence_gate_rejects_numeric_mismatch():
    from tools.check_full_serving import compare
    assert compare([{"hf_logprobs": [-1.] * 100, "vllm_logprobs": [-1.01] * 100}])['passed']
    assert not compare([{"hf_logprobs": [-1.] * 100, "vllm_logprobs": [-1.1] * 100}])['passed']
    with pytest.raises(ValueError):
        compare([{"hf_logprobs": [-1.], "vllm_logprobs": [float('nan')]}])


def test_retry_keeps_every_previously_published_checkpoint(tmp_path):
    plan = token_plan([{"input_ids": [0, 1], "labels": [-100, 1]}], token_budget=16)
    def export(model, path, base, rank):
        write_json(path / 'weights.json', {})
        return {"artifact_sha256": artifact_hashes(path), "text_export_dtype": "bfloat16"}
    for attempt in range(2):
        output = tmp_path / f'run/stages/train-sad/attempt-{attempt:03d}/output'
        publish = publisher(output, 'base', {}, plan, export=export)
        publish(None, {'processed_token_positions': 16})
    models = json.loads((tmp_path / 'run/models.json').read_text())
    assert len(models) == len({m['name'] for m in models}) == 10
    assert all(Path(m['checkpoint']).is_dir() for m in models)


@pytest.mark.parametrize('method', ['sad', 'adftp'])
def test_training_disclosure_identifies_verified_input(tmp_path, method):
    from baselines.common.pipeline import update_disclosure
    update_disclosure(tmp_path, {'method':method, 'K':5, 'unique_instances':5, 'task_groups':5,
                                'verified':'collection', 'collection_sha256':'hash'})
    value = json.loads((tmp_path / 'disclosure.json').read_text())
    assert value['source'] == 'verified_support_collection' and value['expansion'] is False
    assert value['totals']['teacher_calls'] == 0 and value['collection_sha256'] == 'hash'
