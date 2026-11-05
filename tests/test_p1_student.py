from copy import deepcopy
from pathlib import Path

import pytest
import torch

from tests.p0_fakes import tiny_student, fake_pair, fake_greedy
from repair.p0_pairs import render_state
from repair.p0_introspect import tensor_hash
from repair.dry_fakes import FakeEnvironment
from surgery.student import Student
from surgery.session import Session
from surgery.edits import composed_weights, save_delta, load_delta, all_modules
from surgery.conditions import register


def student():
    p0 = tiny_student()
    return Student(p0.model, p0.tokenizer, identity=p0.identity, max_new_tokens=4, max_model_len=8192)


def directions(s, width=16, count=2):
    ids = []
    for i in range(count):
        v = torch.zeros(width); v[i] = 1
        ids.append(register(s, {0: v}))
    return ids


def project(s, alpha=0.5):
    return dict(type='project_out', direction_ids=directions(s), layers=[0], targets=['mlp.down_proj'], alpha=alpha)


def rank1(s):
    return dict(type='add_rank1', a=dict(direction_id=directions(s)[0]),
        b=dict(direction_id=directions(s, 24)[0]), layers=[0], targets=['mlp.down_proj'], alpha=0.003)


def scale():
    return dict(type='scale_neurons', layers=[0], neurons={'0': [0, 2]}, c=0.7)


def hashes(s):
    return {n: tensor_hash(p) for n, p in s.model.named_parameters()}


@pytest.mark.parametrize('kind', ['project', 'rank1', 'scale', 'delta'])
@pytest.mark.parametrize('raises', [False, True])
def test_every_edit_exact_rollback_and_forward(tmp_path, kind, raises):
    s = student()
    if kind == 'delta':
        ref = save_delta(tmp_path, {'0:mlp.down_proj': torch.full((16, 24), 1e-5)}, {})
        edit = dict(type='local_finetune', delta=ref, alpha=1)
    else:
        edit = dict(project=lambda: project(s), rank1=lambda: rank1(s), scale=scale)[kind]()
    before = hashes(s)
    original = s.layers[0].mlp.down_proj.weight
    try:
        with s.edited([edit]) as receipt:
            assert s.layers[0].mlp.down_proj.weight is not original
            assert s.layers[0].mlp.down_proj.weight.dtype == original.dtype == torch.bfloat16
            assert not s.layers[0].mlp.down_proj._forward_pre_hooks
            assert not s.layers[0].mlp.down_proj._forward_hooks
            assert receipt['edited_parameters'] == 16 * 24
            assert receipt['native_execution'] and receipt['fp32_composition']
            assert receipt['rounding']['rounded_changed_parameters'] > 0
            s._forward([130, 131, 132])
            if raises:
                raise RuntimeError('fixture')
    except RuntimeError:
        assert raises
    assert s.layers[0].mlp.down_proj.weight is original
    assert hashes(s) == before
    assert s.active_edit is None


def test_postnorm_rank_k_formula_and_order():
    s = student()
    s.layers[0].post_feedforward_layernorm.weight.data.copy_(torch.arange(1, 17))
    p, a, c = project(s), rank1(s), scale()
    raw = torch.randn(16, 2, generator=torch.Generator().manual_seed(7))
    p['direction_ids'] = [register(s, {0: raw[:, i] / raw[:, i].norm()}) for i in range(2)]
    weights = composed_weights(s, [p])['0:mlp.down_proj']
    w = s.layers[0].mlp.down_proj.weight.float()
    d = torch.stack([s.directions[i][0] for i in p['direction_ids']], 1)
    u = torch.linalg.qr(s.layers[0].post_feedforward_layernorm.weight.float()[:, None] * d).Q
    assert torch.allclose(weights, w - p['alpha'] * u @ u.T @ w, atol=1e-7)
    unscaled = torch.linalg.qr(d).Q
    assert not torch.allclose(weights, w - p['alpha'] * unscaled @ unscaled.T @ w, atol=1e-5)
    assert not torch.equal(composed_weights(s, [a, c])['0:mlp.down_proj'], composed_weights(s, [c, a])['0:mlp.down_proj'])


def test_sub_bf16_delta_vanishes_and_reuse_hash(tmp_path):
    s = student()
    original = s.layers[0].mlp.down_proj.weight
    original.data.fill_(1)
    before = hashes(s)
    logits = s._forward([130, 131]).logits.clone()
    delta = torch.full((16, 24), 1e-6)
    ref = save_delta(tmp_path, {'0:mlp.down_proj': delta}, {})
    e = dict(type='local_finetune', delta=ref, alpha=1)
    assert load_delta(ref)['0:mlp.down_proj'].dtype == torch.float32
    assert torch.equal(load_delta(ref)['0:mlp.down_proj'], delta)
    for _ in range(2):
        with s.edited([e]) as receipt:
            assert s.layers[0].mlp.down_proj.weight is original
            report = receipt['rounding']
            assert report['fp32_changed_parameters'] == 16 * 24
            assert report['rounded_changed_parameters'] == 0
            assert report['rounded_relative_frobenius'] == 0
            assert report['vanished'] and report['per_edit'][0]['effect_vanished']
            assert torch.equal(logits, s._forward([130, 131]).logits)
        assert hashes(s) == before
    Path(ref['path']).write_bytes(b'corrupt')
    with pytest.raises(ValueError, match='sha256'):
        load_delta(ref)


@pytest.mark.parametrize('kind', ['project', 'rank1', 'scale', 'delta', 'empty'])
def test_noop_keeps_parameter_object_bytes_and_logits(tmp_path, kind):
    s = student()
    original = s.layers[0].mlp.down_proj.weight
    original.data[0, 0] = -0.0
    if kind == 'delta':
        ref = save_delta(tmp_path, {'0:mlp.down_proj': torch.ones(16, 24)}, {})
        edits = [dict(type='local_finetune', delta=ref, alpha=0)]
    elif kind == 'empty':
        edits = []
    else:
        edit = dict(project=lambda: project(s), rank1=lambda: rank1(s), scale=scale)[kind]()
        edit['c' if kind == 'scale' else 'alpha'] = 1 if kind == 'scale' else 0
        edits = [edit]
    before = hashes(s)
    logits = s._forward([130, 131]).logits.clone()
    with s.edited(edits) as receipt:
        assert s.layers[0].mlp.down_proj.weight is original
        assert hashes(s) == before
        assert receipt['rounding']['rounded_changed_parameters'] == 0
        assert receipt['rounding']['rounded_relative_frobenius'] == 0
        assert not receipt['rounding']['vanished']
        assert torch.equal(logits, s._forward([130, 131]).logits)
    assert s.layers[0].mlp.down_proj.weight is original
    assert hashes(s) == before


def test_invalid_direction_does_not_change_weights():
    s = student()
    before = hashes(s)
    edit = project(s, 0)
    edit['direction_ids'] *= 5
    with pytest.raises(ValueError):
        with s.edited([edit]):
            pass
    assert hashes(s) == before


def test_rounded_delta_counts_and_relative_norm(tmp_path):
    s = student()
    original = s.layers[0].mlp.down_proj.weight
    original.data.fill_(1)
    delta = torch.zeros_like(original, dtype=torch.float32)
    delta[0, :3] = torch.tensor([0.01, 0.001, -0.01])
    ref = save_delta(tmp_path, {'0:mlp.down_proj': delta}, {})
    edits = [dict(type='local_finetune', delta=ref, alpha=1)]
    expected = (original.float() + delta).bfloat16()
    expected_norm = float((expected.double() - original.double()).norm() / original.double().norm())
    with s.edited(edits) as receipt:
        assert tensor_hash(s.layers[0].mlp.down_proj.weight) == tensor_hash(expected)
        report = receipt['rounding']
        for row in (report, report['tensors']['0:mlp.down_proj'], report['per_edit'][0]):
            assert row['fp32_changed_parameters'] == 3
            assert row['rounded_changed_parameters'] == 2
            assert row['rounded_relative_frobenius'] == pytest.approx(expected_norm)
            assert not row['vanished']
        assert report['tensors']['0:mlp.down_proj']['native_dtype'] == 'bfloat16'


def test_composition_rounds_only_after_all_edits_and_reports_each_effect(tmp_path):
    s = student()
    original = s.layers[0].mlp.down_proj.weight
    original.data.fill_(1)
    edits = [dict(type='local_finetune', delta=save_delta(tmp_path,
        {'0:mlp.down_proj': torch.full((16, 24), delta)}, {}), alpha=1) for delta in (0.002, 0.002, 0.0001)]
    expected = composed_weights(s, edits)['0:mlp.down_proj'].bfloat16()
    with s.edited(edits) as receipt:
        assert tensor_hash(s.layers[0].mlp.down_proj.weight) == tensor_hash(expected)
        assert not torch.equal(expected, original)
        report = receipt['rounding']
        assert report['rounded_changed_parameters'] == 16 * 24
        assert [r['rounded_changed_parameters'] for r in report['per_edit']] == [0, 384, 384]
        assert [r['rounded_effect_changed_parameters'] for r in report['per_edit']] == [0, 384, 0]
        assert [r['effect_vanished'] for r in report['per_edit']] == [True, False, True]
    assert s.layers[0].mlp.down_proj.weight is original


@pytest.mark.parametrize('cancel', [False, True])
def test_only_final_rounding_overflow_rejected_with_exact_rollback(cancel):
    s = student()
    before = hashes(s)
    original = s.layers[0].mlp.down_proj.weight
    edit = dict(rank1(s), alpha=torch.finfo(torch.float32).max)
    if cancel:
        edits = [edit, dict(edit, alpha=-edit['alpha'])]
        expected = composed_weights(s, edits)['0:mlp.down_proj'].bfloat16()
        with s.edited(edits) as receipt:
            assert torch.equal(s.layers[0].mlp.down_proj.weight, expected)
            assert receipt['rounding']['rounded_finite']
            assert not receipt['rounding']['per_edit'][0]['rounded_finite']
            assert receipt['rounding']['per_edit'][0]['rounded_relative_frobenius'] is None
    else:
        with pytest.raises(ValueError, match='Nonfinite native-dtype'):
            with s.edited([edit]):
                pytest.fail('Overflow must not reach evaluation')
    assert hashes(s) == before and s.layers[0].mlp.down_proj.weight is original
    assert s.active_edit is None


def test_gradient_and_neuron_tools_and_no_parameter_grad_leak():
    s = student()
    state = dict(text='room', spans=[], original_thought={'prefix': ''})
    examples = [(state, 'look', 'inventory')]
    before = hashes(s)
    result = s.gradient_attribution(examples)
    assert len(result['ranking']) == len(all_modules(s))
    assert result['ranking'][0]['gradient_norm'] > 0
    assert result['ranking'] == sorted(result['ranking'], key=lambda r: (-r['gradient_norm'], r['tensor']))
    result = s.neuron_attribution(examples, 0, 4)
    assert len(result['top_neurons']) == 4
    assert s.directions[result['direction_id']][0].shape == (24,)
    assert hashes(s) == before
    assert all(not p.requires_grad and p.grad is None for p in s.model.parameters())


def test_local_lora_training_loss_delta_and_reuse(tmp_path):
    s = student()
    examples = [(dict(text='room', spans=[]), 'look', 'inventory')]
    before = hashes(s)
    result = s.local_finetune(examples, ['0:mlp.down_proj'], rank=2, lr=1e-2, steps=3,
        directory=tmp_path, pair_ids=['p'], framing='no_thought')
    assert result['training_exposures'] == 3
    delta = load_delta(result['edit']['delta'])['0:mlp.down_proj']
    assert delta.norm() > 0 and delta.dtype == torch.float32
    assert torch.linalg.matrix_rank(delta, atol=1e-6) <= 2
    assert hashes(s) == before
    s.local_finetune = lambda *a, **k: pytest.fail('Must reuse delta, never retrain')
    with s.edited([result['edit']]):
        s._forward([130, 131])
    assert hashes(s) == before


@pytest.mark.parametrize('rank,steps,lr', [(0, 1, 1e-4), (17, 1, 1e-4), (2, 201, 1e-4), (2, 0, 1e-4), (2, 1, 0)])
def test_finetune_limits(tmp_path, rank, steps, lr):
    with pytest.raises(ValueError):
        student().local_finetune([(dict(text='x'), 'x', 'y')], ['0:mlp.down_proj'], rank=rank, steps=steps,
            lr=lr, directory=tmp_path, pair_ids=['p'])


def test_new_session_tools_cache_scope_and_rollout(tmp_path):
    s = student()
    pair = fake_pair()
    session = Session(s, [pair], tmp_path, env_factory=lambda p: FakeEnvironment())
    # Short state for costly derivative probes; still a diagnosis-only capability.
    session.state(pair['pair_id'])['text'] = 'room'
    before = hashes(s)
    for tool, args in [('gradient_attribution', dict(pair_ids=[pair['pair_id']])),
                       ('neuron_attribution', dict(pair_ids=[pair['pair_id']], layer=0, top_n=2)),
                       ('local_finetune', dict(pair_ids=[pair['pair_id']], modules=['0:mlp.down_proj'], rank=2, lr=1e-3, steps=1))]:
        first = session.call(tool, args)
        second = session.call(tool, args)
        assert not first['cache_hit'] and second['cache_hit']
        assert hashes(s) == before
    with pytest.raises(KeyError):
        session.call('gradient_attribution', dict(pair_ids=['test-pair']))
    with pytest.raises(ValueError):
        Session(s, [dict(pair, split='test')], tmp_path / 'bad')
    with pytest.raises(ValueError):
        session.resolve_edit([dict(type='local_finetune', delta={'path': '/arbitrary'}, alpha=1)])


def test_p0_tools_reused_with_p1_student(tmp_path):
    s = student()
    s.greedy = fake_greedy
    pair = fake_pair()
    session = Session(s, [pair], tmp_path, env_factory=lambda p: FakeEnvironment())
    ident = pair['pair_id']
    session.call('action_scores', dict(state=ident, candidates=['look', 'inventory']))
    session.call('logit_lens', dict(state=ident, a_f='look', a_r='inventory'))
    session.call('attention_summary', dict(state=ident, layers=[0]))
    session.call('patch', dict(target_state=ident, source_state=ident, layers=[0], positions=[dict(target=-1, source=-1)], candidates=['look']))
    direction = session.call('direction', dict(positive=[dict(state=ident, action='look')], negative=[dict(state=ident, action='inventory')], layers=[0]))['data']['direction_id']
    session.call('projection', dict(state=ident, action='look', direction_id=direction))
    span = session.state(ident)['spans'][0]['id']
    session.call('counterfactual_context', dict(state=ident, edit=dict(op='insert', span=span, side='after', text='test')))
    edit = [dict(type='project_out', direction_ids=[direction], layers=[0], targets=['mlp.down_proj'], alpha=0.1)]
    applied = session.call('apply_edit', dict(state=ident, candidates=['look'], edit=edit))
    result = session.call('rollout', dict(state=ident, edit=edit))
    assert applied['data']['receipt']['rounding'] == result['data']['edit_receipt']['rounding']
    assert result['data']['edit_receipt']['rounding']['rounded_changed_parameters'] > 0
    assert result['data']['edit_receipt']['rounding']['per_edit'][0]['index'] == 0
    assert result['data']['verification'] == 'real_environment'
    assert session.environment_steps > 0


def test_add_rank1_accepts_shape_matched_gate_projection():
    s = student()
    edit = dict(type='add_rank1', a=dict(direction_id=directions(s, 24)[0]),
        b=dict(direction_id=directions(s, 16)[0]), layers=[0], targets=['mlp.gate_proj'], alpha=0.1)
    before = hashes(s)
    with s.edited([edit]):
        s._forward([130, 131])
    assert hashes(s) == before


def test_gradient_failure_restores_flags_and_hooks(monkeypatch):
    s = student()
    def fail(*a, **k):
        raise RuntimeError('fixture forward failure')
    monkeypatch.setattr(s, 'log_probability', fail)
    with pytest.raises(RuntimeError):
        s.gradient_attribution([(dict(text='x'), 'look', 'inventory')])
    assert all(not p.requires_grad and p.grad is None for p in s.model.parameters())
    with pytest.raises(RuntimeError):
        s.neuron_attribution([(dict(text='x'), 'look', 'inventory')], 0, 2)
    assert not s.layers[0].mlp.down_proj._forward_pre_hooks
