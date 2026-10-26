"""Golden-tree regression for the only shared-file change (opt-in hooks)."""
from copy import deepcopy
import json
from pathlib import Path
import subprocess
import types

import pytest


@pytest.mark.parametrize('objective,packing,scoring', [
    ('sft', 'episode', 'per_turn'), ('sft', 'turn', 'per_turn'), ('sft', 'turn', 'shared_history'),
    ('sad', 'turn', 'per_turn'), ('sad', 'turn', 'shared_history'),
    ('steca', 'episode', 'per_turn'), ('steca', 'turn', 'per_turn'),
    ('dpo', 'episode', 'per_turn'), ('dpo', 'turn', 'per_turn'), ('dpo', 'turn', 'shared_history'),
    ('ipr', 'episode', 'per_turn'), ('ipr', 'turn', 'per_turn'), ('ipr', 'turn', 'shared_history'),
    ('smartad', 'turn', 'per_turn'), ('smartad', 'turn', 'shared_history')])
def test_default_loop_bitwise_matches_2506fa9(tmp_path, monkeypatch, objective, packing, scoring):
    torch = pytest.importorskip('torch')
    import baselines.common.fsdp_train as current
    import baselines.common.shared_history as shared
    import baselines.smartad.data as smartad
    from baselines.common.conversation import conversation, sequence_rows
    from test_sad import CharTokenizer, episode
    old = types.ModuleType('theta0_trainer')
    source = subprocess.run(['git', 'show', '2506fa9:baselines/common/fsdp_train.py'],
                            capture_output=True, check=True).stdout
    exec(compile(source, '2506fa9:baselines/common/fsdp_train.py', 'exec'), old.__dict__)
    class Tiny(torch.nn.Module):
        def __init__(self, frozen=False):
            super().__init__()
            self.adapter = torch.nn.Parameter(torch.tensor(.3), requires_grad=not frozen)
    def scores(model, rows, device, **kw):
        # Deterministic model score, with real encoders, sampler, objectives,
        # backward, clipping, AdamW, schedules, accounting and checkpoint hooks.
        values = [-(model.adapter - .8).square() * sum(
            t['supervised_tokens'] for t in sequence_rows(r)) for r in rows]
        return torch.stack(values)
    def smart_objective(model, row, device, **kw):
        weights = sum(sum(t['token_weights']) for t in sequence_rows(row))
        return (model.adapter - .8).square() * weights, {}
    monkeypatch.setattr(shared, 'validate_model', lambda m: None)
    monkeypatch.setattr(smartad, 'objective', smart_objective)
    for module in (old, current):
        monkeypatch.setattr(module, 'batch_logps', scores)
        monkeypatch.setattr(module, 'sequence_logps', scores)
        monkeypatch.setattr(module, 'configure_flex_attention', lambda *a: {'kernel_options': {}})
    rows = []
    for i in range(3):
        ep = episode('Thought: ' + 'check ' * (i + 1) + '\nAction: look')
        if objective == 'smartad':
            row = smartad.encode_episode(ep, CharTokenizer())
        else:
            row = conversation(ep, CharTokenizer(), source='trusted_teacher_seed', packing=packing)
        if objective == 'steca': row.update(reward=.5, dataset_scale=2.)
        if objective in ('dpo', 'ipr'): row = {'chosen': row, 'rejected': deepcopy(row)}
        rows.append(row)
    results = []
    for name, module in [('old', old), ('current', current)]:
        model, reference = Tiny(), Tiny(frozen=True)
        callbacks = []
        opts = {'processed_token_budget': 1, 'checkpoint_callback': lambda m, s: callbacks.append(s)} if objective == 'sad' else {}
        state = module.train_loop(model, deepcopy(rows), tmp_path / name, objective=objective, learning_rate=5e-5,
            scheduler='linear', warmup_ratio=0, original_batch=8, epochs=2, packing=packing,
            scoring=scoring, reference=reference, **opts)
        state.pop('wall_time_seconds')
        ledger = [json.loads(line) for line in (tmp_path / name / 'updates.jsonl').read_text().splitlines()]
        results.append((model.adapter.detach(), state, ledger, callbacks))
    assert torch.equal(results[0][0], results[1][0])
    assert results[0][1:] == results[1][1:]


@pytest.mark.parametrize('options', [
    {'fixed_batches': []}, {'fixed_batches': [{'indices': [0]}]},
    {'fixed_batches': [{'indices': [0, 0, 0, 9]}]},
    {'fixed_batches': [{'indices': [0, 0, 0, True]}]},
    {'fixed_batches': [{'indices': [0] * 4}], 'objective': 'dpo'},
    {'fixed_batches': [{'indices': [0] * 4}], 'processed_token_budget': 1},
    {'optimizer_state': {}}, {'checkpoint_callback': lambda *a: None}])
def test_opt_in_hooks_fail_closed(tmp_path, options):
    torch = pytest.importorskip('torch')
    from baselines.common.fsdp_train import train_loop
    row = {'packing': 'episode', 'input_ids': [1, 2], 'labels': [-100, 2], 'supervised_tokens': 1}
    rows = [{'chosen': row, 'rejected': row}] if options.get('objective') == 'dpo' else [row]
    model = torch.nn.Linear(1, 1)
    args = dict(objective='sft', learning_rate=5e-5, scheduler='linear', warmup_ratio=0, original_batch=4)
    with pytest.raises(ValueError):
        train_loop(model, rows, tmp_path, **{**args, **options})
