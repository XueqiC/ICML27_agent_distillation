from copy import deepcopy
from types import SimpleNamespace

import pytest
import torch

from tests.p0_fakes import fake_pair, TinyTokenizer
from tests.test_p1_student import student, project, rank1, scale
from surgery.conditions import (random_edits, fixed_rule, automatic_finetune, ALPHAS, LRS, STEPS,
                               diagnosis_only, condition_set, freeze, load_condition, restore)
from surgery.edits import load_delta, save_delta
from surgery.common import seal


def test_random_controls_structure_and_frobenius(tmp_path):
    s = student()
    delta = save_delta(tmp_path / 'teacher', {'0:mlp.down_proj': torch.randn(16, 24),
                                          '1:mlp.down_proj': torch.zeros(16, 24)}, {})
    original = [project(s), rank1(s), scale(), dict(type='local_finetune', delta=delta, alpha=0.7)]
    before = deepcopy(original)
    variants = [random_edits(s, original, i, tmp_path / f'random-{i}') for i in range(5)]
    assert before == original
    assert len({v[0]['direction_ids'][0] for v in variants}) == 5
    for random in variants:
        for teacher_edit, edit in zip(original, random):
            assert teacher_edit['type'] == edit['type']
            for key in ('layers', 'targets', 'alpha', 'c'):
                assert teacher_edit.get(key) == edit.get(key)
        assert len(random[0]['direction_ids']) == len(original[0]['direction_ids'])
        assert len(random[2]['neurons']['0']) == len(original[2]['neurons']['0'])
        teacher_delta, random_delta = load_delta(delta), load_delta(random[3]['delta'])
        assert teacher_delta.keys() == random_delta.keys()
        for key in teacher_delta:
            assert torch.allclose(teacher_delta[key].norm(), random_delta[key].norm(), rtol=1e-6, atol=1e-6)


def test_fixed_grid_and_category_order_only_diagnosis():
    from repair.p0_pairs import CATEGORIES
    diagnosis = [fake_pair(i, category=c) for i, c in enumerate(CATEGORIES)]
    seen = []
    def hidden(state, action, layers, framing):
        assert framing == 'original_thought'
        assert layers == list(range(20, 42))
        return dict(layers={i: torch.arange(1, 17).float() * (2 if action == 'look' else 1) for i in layers})
    s = SimpleNamespace(layer_ids=lambda x: x, hidden_mean=hidden, tokenizer=TinyTokenizer(), directions={})
    def rollout(pair, edits):
        assert pair['split'] == 'diagnosis'
        seen.append(pair['pair_id'])
        return dict(success=edits[0]['alpha'] == 0.5)
    edits, selection = fixed_rule(s, diagnosis, rollout, setting='A')
    assert len(edits) == 5
    assert selection['chosen_alpha'] == 0.5
    assert [r['alpha'] for r in selection['grid']] == list(ALPHAS)
    assert len(seen) == 3 * len(diagnosis)
    with pytest.raises(ValueError, match='diagnosis'):
        fixed_rule(s, diagnosis + [fake_pair(999, split='test')], rollout, setting='A')


def test_automatic_snr_top_quarter_grid_rank16_and_diagnosis(tmp_path):
    diagnosis = [fake_pair()]
    trained = []
    def train(examples, modules, **kwargs):
        assert modules == ['7:mlp.down_proj', '6:mlp.down_proj']
        assert kwargs['rank'] == 16 and kwargs['pair_ids'] == ['pair-0']
        trained.append((kwargs['lr'], kwargs['steps']))
        return dict(edit=dict(type='fixture', lr=kwargs['lr'], steps=kwargs['steps']))
    s = SimpleNamespace(tokenizer=TinyTokenizer(), gradient_statistics=lambda *a, **k:
        [dict(tensor=f'{i}:mlp.down_proj', snr=i) for i in range(8)], local_finetune=train)
    seen = []
    def rollout(pair, edits):
        assert pair['split'] == 'diagnosis'
        seen.append(pair['pair_id'])
        return dict(success=(edits[0]['lr'], edits[0]['steps']) == (3e-5, 200))
    edits, selection = automatic_finetune(s, diagnosis, rollout, tmp_path)
    assert trained == [(lr, steps) for lr in LRS for steps in STEPS]
    assert selection['chosen'] == dict(lr=3e-5, steps=200)
    assert len(seen) == 6
    with pytest.raises(ValueError):
        automatic_finetune(s, [fake_pair(split='test')], rollout, tmp_path)


def test_freeze_reload_and_tamper(tmp_path):
    s = student()
    value = freeze(tmp_path / 'E1.json', s, 'pairs', 'A', 'E1', [project(s)])
    with s.edited(value['edits']) as receipt:
        assert value['edit_receipt'] == receipt
    assert value['edit_receipt']['rounding']['rounded_changed_parameters'] > 0
    loaded = load_condition(tmp_path / 'E1.json', 'pairs')
    s.directions.clear()
    restore(s, loaded)
    assert s.directions
    assert loaded == value
    with pytest.raises(ValueError):
        load_condition(tmp_path / 'E1.json', 'different_split')


def test_every_condition_required_before_test_side_access():
    pairs = dict(setting='A', sha256='pairs')
    def condition(name):
        return seal(dict(schema='p1-condition/1', setting='A', name=name, trial=False, pairs_sha256='pairs', student_identity={}))
    all_names = ['E0', 'E1', 'E3', 'E4', 'E5'] + [f'E2_{i}' for i in range(5)]
    values = [condition(n) for n in all_names]
    assert len(condition_set(values, pairs)) == 10
    with pytest.raises(ValueError, match='EVERY'):
        condition_set(values[:-1], pairs)


def test_condition_gate_precedes_manifest_and_test_access(tmp_path, monkeypatch):
    from surgery import evaluation
    monkeypatch.setattr(evaluation, 'Manifest', lambda *a, **k: pytest.fail('No test/probe access before freeze'))
    with pytest.raises(ValueError, match='EVERY'):
        evaluation.make_plan(dict(setting='A', sha256='pairs'), [], tmp_path, round_dir='/absent')


def test_frozen_condition_packages_delta_for_portable_reuse(tmp_path):
    from pathlib import Path
    s = student()
    s.layers[0].mlp.down_proj.weight.data.fill_(1)
    ref = save_delta(tmp_path / 'session', {'0:mlp.down_proj': torch.ones(16, 24) * 1e-5}, {})
    value = freeze(tmp_path / 'conditions/E1.json', s, 'pairs', 'A', 'E1',
                   [dict(type='local_finetune', delta=ref, alpha=1)])
    packaged = value['edits'][0]['delta']
    rounding = value['edit_receipt']['rounding']
    assert rounding['vanished'] and rounding['per_edit'][0]['effect_vanished']
    assert rounding['rounded_changed_parameters'] == rounding['rounded_relative_frobenius'] == 0
    assert load_condition(tmp_path / 'conditions/E1.json', 'pairs')['edit_receipt'] == value['edit_receipt']
    assert Path(packaged['path']).is_relative_to(tmp_path / 'conditions')
    Path(ref['path']).unlink()
    assert load_delta(packaged)['0:mlp.down_proj'].shape == (16, 24)
