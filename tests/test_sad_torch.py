"""Numerical one-hot span CE and optimizer exposure on a tiny CPU Gemma."""
import json

import pytest

torch = pytest.importorskip('torch')
pytest.importorskip('transformers')
from test_stage2b_torch import tiny_model, tokenizer, trajectory
from baselines.common.fsdp_train import sequence_logps, train_loop
from baselines.sad.data import encode_episode


def test_real_gemma_offsets_match_registered_reason_action_masks(tokenizer):
    example = trajectory()
    example['steps'][0]['action_raw'] = 'Intro ignored\nThought: café and reasoning\nNote: ignore this\nAction: look\nIgnored trailing text'
    result = encode_episode(example, tokenizer)
    row = result['turns'][0]
    target = tokenizer.decode([v for v in row['labels'] if v != -100])
    assert 'Thought:' in target and 'Action: look' in target and target.endswith('<turn|>')
    assert 'ignore this' not in target and 'Ignored trailing' not in target and 'Intro' not in target
    assert result['complexity'] == result['supervised_tokens']


def test_summed_span_ce_gradient_order_budget_and_crossings(tmp_path, monkeypatch):
    torch.set_num_threads(1)
    torch.manual_seed(21)
    model = tiny_model()
    rows = [{"input_ids": [1, 2, 3, 4, 5], "labels": [-100, -100, 3, -100, 5]},
            {"input_ids": [1, 6, 7, 8], "labels": [-100, 6, -100, 8]}]
    # Independent full-logit causal CE oracle, summed over both disjoint spans.
    model.eval()
    ids = torch.tensor([rows[0]['input_ids']])
    logits = model(input_ids=ids, use_cache=False).logits[:, :-1]
    ce = torch.nn.functional.cross_entropy(logits.reshape(-1, 128),
        torch.tensor(rows[0]['labels'][1:]), ignore_index=-100, reduction='sum')
    expected = torch.autograd.grad(ce, list(model.parameters()), allow_unused=True)
    loss = -sequence_logps(model, [rows[0]], 'cpu').sum()
    actual = torch.autograd.grad(loss, list(model.parameters()), allow_unused=True)
    assert torch.allclose(loss, ce, atol=1e-6)
    for a, e in zip(actual, expected):
        if a is not None:
            assert torch.allclose(a, e, atol=1e-6)
    examples = [{"packing": "turn", "turns": [row], "complexity": 2} for row in rows]
    callbacks = []
    before = [p.detach().clone() for p in model.parameters()]
    state = train_loop(model, examples, tmp_path / 'train', objective='sad', learning_rate=2e-5,
        scheduler='cosine', warmup_ratio=.03, original_batch=64, processed_token_budget=19,
        packing='turn', scoring='per_turn', checkpoint_callback=lambda model, state: callbacks.append(state.copy()))
    updates = [json.loads(line) for line in (tmp_path / 'train/updates.jsonl').read_text().splitlines()]
    assert [r['indices'] for r in updates] == [[0], [1], [0], [1], [0]]
    assert state['logical_forward_tokens'] == callbacks[-1]['processed_token_positions'] == 23
    assert state['supervised_tokens_seen'] == 10
    assert state['physical_reference_forward_tokens'] == 0
    assert updates[0]['lr'] == 0 and updates[1]['lr'] == 2e-5
    assert any(not torch.equal(a, b) for a, b in zip(before, model.parameters()))
