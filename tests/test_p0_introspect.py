from copy import deepcopy
import json

import pytest

from repair.p0_introspect import ToolSession, counterfactual, tensor_hash
from repair.p0_pairs import render_state
from repair.dry_fakes import FakeEnvironment
from p0_fakes import tiny_student, fake_pair


@pytest.fixture
def student():
    return tiny_student()


@pytest.fixture
def session(student, tmp_path):
    return ToolSession(student, [fake_pair(0), fake_pair(1)], tmp_path,
                       env_factory=lambda pair: FakeEnvironment())


def direction(session):
    return session.call('direction', dict(positive=[dict(state='pair-0', action='look')],
        negative=[dict(state='pair-1', action='inventory')], layers=[0, 2]))['data']['direction_id']


def test_scores_full_sequence_matches_manual_teacher_forcing(session):
    import torch
    state = session.state('pair-0')
    result = session.call('action_scores', dict(state='pair-0', candidates=['look', 'inventory']))
    prefix = session.student.ids(session.student.decision_prefix(state))
    suffix, _ = session.student.action_tokens('look')
    logits = session.student._forward(prefix + suffix, keep=0).logits.float()[0]
    expected = sum(float(logits[len(prefix) + j - 1].log_softmax(-1)[tok]) for j, tok in enumerate(suffix))
    assert result['data']['scores'][0]['log_probability'] == pytest.approx(expected, abs=1e-5)
    assert result['data']['scores'][0]['tokens'] == len(suffix)
    assert suffix[-1] == 106
    assert isinstance(result['summary'], str)
    json.dumps(result, allow_nan=False)


def test_action_scores_cache(session, monkeypatch):
    kw = dict(state='pair-0', candidates=['look'])
    first = session.call('action_scores', kw)
    monkeypatch.setattr(session.student, 'action_scores', lambda *a, **k: pytest.fail('cache miss'))
    second = session.call('action_scores', kw)
    assert not first['cache_hit'] and second['cache_hit']
    assert first['data'] == second['data']


def test_lens_first_difference_and_identical_actions(session):
    r = session.call('logit_lens', dict(state='pair-0', a_f='look', a_r='inventory'))['data']
    assert len(r['layers']) == 4
    assert r['fail_token'] != r['repair_token']
    assert all(x['difference'] == pytest.approx(x['logp_repair'] - x['logp_fail']) for x in r['layers'])
    assert session.student.logit_lens(session.state('pair-0'), 'look', 'look')['status'] == 'identical_actions'


def test_single_query_attention_matches_native_eager_and_shared_kv(session):
    import torch
    from transformers.models.gemma4.modeling_gemma4 import eager_attention_forward
    state = session.state('pair-0')
    r = session.call('attention_summary', dict(state='pair-0', layers=[0, 1, 2, 3]))['data']
    assert [x['kv_shared'] for x in r['layers']] == [False, False, True, True]
    assert not hasattr(session.student.layers[2].self_attn, 'k_proj')
    for layer in r['layers']:
        assert layer['total_mass'] == pytest.approx(1.0, abs=1e-5)
        assert all(0 <= s['mean'] <= s['max'] <= 1.000001 for s in layer['spans'])
    # Independent reference captures native attention weights on a tiny input.
    # Only THIS TEST switches to eager; production always uses one-query hooks.
    short = dict(text='abcdefghijklmnopqrstuvwx', spans=[
        dict(id='early', label='early', start=0, end=5),
        dict(id='middle', label='middle', start=8, end=12),
        dict(id='last', label='last', start=23, end=24)])
    results = session.student.attention_summary(short, [0, 1, 2, 3])
    session.student.model.set_attn_implementation('eager')
    got, hooks = {}, []
    for i, layer in enumerate(session.student.layers):
        hooks.append(layer.self_attn.register_forward_hook(lambda m, a, out, i=i: got.update({i: out[1]})))
    try:
        session.student._forward(session.student.ids(short))
    finally:
        for h in hooks:
            h.remove()
        session.student.model.set_attn_implementation('sdpa')
    for row in results['layers']:
        for span, measured in zip(short['spans'], row['spans']):
            expected = got[row['layer']][0, :, -1, span['start']:span['end']].float().sum(-1)
            assert measured['mean'] == pytest.approx(float(expected.mean()), abs=.008)
            assert measured['max'] == pytest.approx(float(expected.max()), abs=.008)


def test_patch_identity_and_hook_cleanup(session):
    args = dict(target_state='pair-0', source_state='pair-0', layers=[0, 2],
                positions=[dict(target=-1, source=-1)], candidates=['look', 'inventory'])
    r = session.call('patch', args)['data']
    assert all(v['delta_log_probability'] == 0 for v in r['changes'])
    assert all(not l._forward_hooks for l in session.student.layers)
    with pytest.raises(ValueError):
        session.student.patch(session.state('pair-0'), session.state('pair-0'), [0],
                              [dict(target=100000, source=0)], ['look'])


def test_direction_uses_action_positions_and_is_normalized(session):
    import torch
    ident = direction(session)
    for v in session.student.directions[ident].values():
        assert float(v.norm()) == pytest.approx(1.0)
    a = session.student.hidden_mean(session.state('pair-0'), 'look', [0])['layers'][0]
    b = session.student.hidden_mean(session.state('pair-0'), 'inventory', [0])['layers'][0]
    assert not torch.equal(a, b)
    p = session.call('projection', dict(state='pair-0', action='look', direction_id=ident))
    assert all(-1.00001 <= x['cosine'] <= 1.00001 for x in p['data']['layers'])


def test_zero_direction_rejected(session):
    with pytest.raises(ValueError, match='Zero'):
        session.call('direction', dict(positive=[dict(state='pair-0', action='look')],
            negative=[dict(state='pair-0', action='look')], layers=[0]))


def test_direction_cache_restores_vectors_without_forward(session, monkeypatch):
    ident = direction(session)
    session.student.directions.clear()
    session.direction_specs.clear()
    monkeypatch.setattr(session.student, 'direction', lambda *a, **k: pytest.fail('recomputed cached direction'))
    assert direction(session) == ident
    assert ident in session.student.directions and ident in session.direction_specs


def test_counterfactual_cache_restores_state_capability(session):
    span = session.state('pair-0')['spans'][0]['id']
    args = dict(state='pair-0', edit=dict(op='delete', span=span))
    first = session.call('counterfactual_context', args)
    ident = first['data']['state_id']
    del session.states[ident]
    second = session.call('counterfactual_context', args)
    assert second['cache_hit'] and session.state(ident)['counterfactual']


@pytest.mark.parametrize('targets', [['mlp.down_proj'], ['self_attn.o_proj'], ['mlp.down_proj', 'self_attn.o_proj']])
def test_output_side_edit_formula_and_exact_rollback(session, targets):
    import torch
    ident = direction(session)
    spec = dict(direction_id=ident, layers=[0, 2], targets=targets, alpha=.8)
    originals = {n: p.detach().clone() for n, p in session.student.model.named_parameters()}
    with session.student.edited(spec) as receipt:
        count = 0
        for i in spec['layers']:
            for target in targets:
                w = getattr(getattr(session.student.layers[i], target.split('.')[0]), target.split('.')[1]).weight
                name = next(n for n, p in session.student.model.named_parameters() if p is w)
                d = session.student.directions[ident][i]
                old = originals[name].float()
                norm_name = 'post_attention_layernorm' if target == 'self_attn.o_proj' else 'post_feedforward_layernorm'
                u = getattr(session.student.layers[i], norm_name).weight.float() * d
                u = u / u.norm()
                assert torch.equal(w, (old - .8 * u[:, None] * (u @ old)[None, :]).to(w.dtype))
                count += w.numel()
        assert receipt['edited_parameters'] == count and receipt['changed_parameters'] > 0
    assert all(torch.equal(p, originals[n]) for n, p in session.student.model.named_parameters())
    with pytest.raises(RuntimeError):
        with session.student.edited(spec):
            raise RuntimeError('fixture')
    assert all(torch.equal(p, originals[n]) for n, p in session.student.model.named_parameters())


@pytest.mark.parametrize('bad', [dict(alpha=float('nan')), dict(targets=['q_proj']), dict(layers=[99]), dict(alpha='1')])
def test_invalid_edits_rejected(session, bad):
    ident = direction(session)
    spec = dict(direction_id=ident, layers=[0], targets=['mlp.down_proj'], alpha=1, **{})
    spec.update(bad)
    with pytest.raises(ValueError):
        with session.student.edited(spec):
            pytest.fail('invalid edit admitted')


def test_apply_edit_contract(session):
    ident = direction(session)
    r = session.call('apply_edit', dict(state='pair-0', candidates=['look'],
        edit=dict(direction_id=ident, layers=[0], targets=['mlp.down_proj'], alpha=1)))
    assert r['data']['reverted'] and session.student.active_edit is None
    json.dumps(r, allow_nan=False)


def test_counterfactual_delete_insert_reorder():
    state = dict(text='aa bb cc', spans=[dict(id=s, label=s, start=i, end=i + 2)
        for s, i in [('aa', 0), ('bb', 3), ('cc', 6)]], pair_id='p', origin_pair_id='p', sha256='x')
    assert counterfactual(state, dict(op='delete', span='bb'))['text'] == 'aa  cc'
    r = counterfactual(state, dict(op='insert', span='bb', side='after', text='X'))
    assert r['text'] == 'aa bbX cc'
    assert next(s for s in r['spans'] if s['id'] == 'cc')['start'] == 7
    assert counterfactual(state, dict(op='reorder', spans=['cc', 'aa']))['text'] == 'cc bb aa'


def test_counterfactual_is_usable_by_scoring(session):
    span = session.state('pair-0')['spans'][0]['id']
    c = session.call('counterfactual_context', dict(state='pair-0', edit=dict(op='delete', span=span)))
    r = session.call('action_scores', dict(state=c['data']['state_id'], candidates=['look']))
    assert r['data']['scores']


def test_diagnosis_boundary_denies_test_pairs_and_arbitrary_states(student, tmp_path, session):
    with pytest.raises(ValueError, match='diagnosis'):
        ToolSession(student, [fake_pair(split='test')], tmp_path / 'bad')
    with pytest.raises(KeyError):
        session.call('action_scores', dict(state='unseen-test-state', candidates=['look']))
    with pytest.raises(KeyError):
        session.call('direction', dict(positive=[dict(state='unseen', action='look')],
            negative=[dict(state='pair-0', action='look')], layers=[0]))


def test_rollout_real_prefix_checks_and_accounting(session, monkeypatch):
    # Fake env still executes real transitions through the unchanged harness.
    sequence = iter(['take apple 1 from cabinet 1', 'clean apple 1 with sinkbasin 1', 'put apple 1 in/on table 1'])
    def greedy(state):
        action = next(sequence)
        return dict(action=action, reply='Action: ' + action, prompt_tokens=1, generated_tokens=1)
    monkeypatch.setattr(session.student, 'greedy', greedy)
    r = session.call('rollout', dict(state='pair-0'))
    assert r['data']['success'] is True and r['data']['environment_steps'] == 4
    assert session.environment_steps == 4
    cached = session.call('rollout', dict(state='pair-0'))
    assert cached['cache_hit'] and session.environment_steps == 4
    receipts = [json.loads(l) for l in (session.directory / 'environment.jsonl').read_text().splitlines()]
    assert sum(r['interaction']['candidate_replay_steps'] for r in receipts if r['settled']) == 4


def test_diverged_replay_never_calls_policy(session, monkeypatch):
    class Diverged(FakeEnvironment):
        def reset(self, task_id):
            v, *rest = super().reset(task_id)
            v['observation'] = 'changed'
            return (v, *rest)
    session.env_factory = lambda pair: Diverged()
    monkeypatch.setattr(session.student, 'greedy', lambda *a: pytest.fail('unverified branch'))
    with pytest.raises(RuntimeError, match='replay/rollout'):
        session.call('rollout', dict(state='pair-0'))
    records = list((session.directory / 'executions').glob('*.json'))
    assert json.loads(records[0].read_text())['success'] == 'unknown'


@pytest.mark.parametrize('target,norm_name', [
    ('self_attn.o_proj', 'post_attention_layernorm'),
    ('mlp.down_proj', 'post_feedforward_layernorm')])
@pytest.mark.parametrize('dtype_name,tolerance', [('float32', 2e-6), ('bfloat16', .025)])
def test_norm_aware_edit_removes_residual_direction(student, target, norm_name, dtype_name, tolerance):
    import torch
    dtype = getattr(torch, dtype_name)
    student.model.to(dtype)
    block = student.layers[0]
    norm = getattr(block, norm_name)
    matrix = block.get_submodule(target)
    with torch.no_grad():
        norm.weight.copy_(torch.linspace(.2, 3.5, 16).to(dtype))
        matrix.weight[0, 0] = -0.0
    generator = torch.Generator().manual_seed(73)
    d = torch.randn(16, generator=generator)
    d /= d.norm()
    student.directions['fixture'] = {0: d}
    spec = dict(direction_id='fixture', layers=[0], targets=[target], alpha=1)
    old_hash = tensor_hash(matrix.weight)
    old = matrix.weight.detach().clone()
    inputs = torch.randn(64, matrix.in_features, generator=generator).to(dtype)
    # This nonuniform scale makes the old pre-norm formula visibly wrong.
    naive = old.float() - d[:, None] * (d @ old.float())[None, :]
    assert (norm((inputs.float() @ naive.T).to(dtype)).float() @ d).abs().max() > .1
    with student.edited(spec) as receipt:
        projected = norm(matrix(inputs)).float() @ d
        assert projected.abs().max() < tolerance
        row, = receipt['projections']
        gd = norm.weight.float() * d
        assert row['post_norm_module'] == 'layers.0.' + norm_name
        assert row['scaled_direction_norm'] == pytest.approx(float(gd.norm()))
        assert row['cosine_u_d'] == pytest.approx(float((gd / gd.norm()) @ d))
        assert receipt['formula'] == 'W - alpha*u*u.T*W, u=normalize(g*d)'
    assert tensor_hash(matrix.weight) == old_hash
    with student.edited(dict(spec, alpha=0)) as receipt:
        assert tensor_hash(matrix.weight) == old_hash
        assert receipt['changed_parameters'] == 0
    assert tensor_hash(matrix.weight) == old_hash


@pytest.mark.parametrize('bad_norm', ['post_feedforward_layernorm_1', 'post_feedforward_layernorm_2',
    'missing_attention', 'missing_feedforward'])
def test_unsupported_norm_paths_refused_before_mutation(student, bad_norm):
    import torch
    block = student.layers[1]
    if bad_norm.startswith('missing'):
        delattr(block, 'post_attention_layernorm' if bad_norm == 'missing_attention' else 'post_feedforward_layernorm')
    else:
        setattr(block, bad_norm, torch.nn.Identity())
    d = torch.ones(16) / 4
    student.directions['fixture'] = {0: d, 1: d}
    originals = {n: tensor_hash(p) for n, p in student.model.named_parameters()}
    with pytest.raises(ValueError, match='MoE|post-norm'):
        with student.edited(dict(direction_id='fixture', layers=[0, 1], targets=['self_attn.o_proj'], alpha=1)):
            pytest.fail('unsupported norm path accepted')
    assert student.active_edit is None
    assert {n: tensor_hash(p) for n, p in student.model.named_parameters()} == originals


def test_exact_original_thought_prefix_and_span(student):
    pair = fake_pair()
    raw = 'Thought:  Preserve spaces.\nSecond line.\nAction: look\nAction: inventory'
    pair['original_action_raw'] = raw
    base = render_state(pair, student.tokenizer)
    framed = student.decision_prefix(base)
    prefix = raw.split('\nAction:', 1)[0] + '\n'
    assert framed['text'] == base['text'] + prefix
    record = framed['framing_record']
    assert record['split_index'] == raw.index('\nAction:')
    assert record['before_action'] + record['separator'] + record['after_separator'] == raw
    assert record['effective'] == record['requested'] == 'original_thought'
    span, = [s for s in framed['spans'] if s['label'] == 'current_thought']
    assert framed['text'][span['start']:span['end']] == prefix
    assert student.decision_prefix(base, 'no_thought')['text'] == base['text']


@pytest.mark.parametrize('raw', ['Action: look', 'Thought: incomplete', 'look', ''])
def test_original_thought_missing_falls_back(student, raw):
    pair = dict(fake_pair(), original_action_raw=raw)
    base = render_state(pair, student.tokenizer)
    framed = student.decision_prefix(base)
    assert framed['text'] == base['text']
    assert framed['framing_record']['effective'] == 'no_thought'
    assert framed['framing_record']['fallback_reason'] == 'missing_thought_or_action_delimiter'
    assert not any(s['label'] == 'current_thought' for s in framed['spans'])


def test_framing_cache_separation_and_free_greedy(session, monkeypatch):
    seen = []
    base = session.state('pair-0')
    def greedy(state):
        seen.append(state['text'])
        assert state['text'] == base['text']
        return dict(action='look', reply='Thought: Greedy check.\nAction: look', prompt_tokens=1, generated_tokens=1)
    monkeypatch.setattr(session.student, 'greedy', greedy)
    results = [session.call('action_scores', dict(state='pair-0', candidates=['look'], framing=f))
               for f in ['original_thought', 'greedy_thought', 'no_thought']]
    assert len({r['cache_key'] for r in results}) == 3
    for result in results:
        assert result['framing'] == result['data']['framing']
        again = session.call('action_scores', dict(state='pair-0', candidates=['look'], framing=result['framing']))
        assert again['cache_hit'] and again['data'] == result['data']
    assert len(seen) == 4  # Three reproduction calls, one generated Thought.
    assert session.call('action_scores', dict(state='pair-0', candidates=['look']))['cache_hit']
    assert results[1]['data']['framing_record']['prefix'] == 'Thought: Greedy check.\n'
    pair = dict(fake_pair(), original_action_raw='Thought: Another.\nAction: look')
    other = render_state(pair, session.student.tokenizer)
    assert other['text'] == base['text'] and other['sha256'] != base['sha256']


def test_greedy_thought_cache_separates_active_edit_and_reuses_after_revert(session, monkeypatch):
    ident = direction(session)
    seen = []
    def greedy(state):
        assert not state.get('_prepared')
        seen.append(deepcopy(session.student.active_edit))
        thought = 'edited' if session.student.active_edit else 'baseline'
        return dict(action='look', reply=f'Thought: {thought}\nAction: look', prompt_tokens=1, generated_tokens=1)
    monkeypatch.setattr(session.student, 'greedy', greedy)
    base = session.state('pair-0')
    first = session.student.decision_prefix(base, 'greedy_thought')
    session.student.decision_prefix(base, 'greedy_thought')
    spec = dict(direction_id=ident, layers=[0], targets=['mlp.down_proj'], alpha=.5)
    with session.student.edited(spec):
        changed = session.student.decision_prefix(base, 'greedy_thought')
        session.student.decision_prefix(base, 'greedy_thought')
    assert session.student.decision_prefix(base, 'greedy_thought') == first
    assert len(seen) == 2 and seen == [None, spec]
    assert changed['text'] != first['text']


def test_current_thought_counterfactual_and_attention(session):
    original = session.student.decision_prefix(session.state('pair-0'))
    args = dict(state='pair-0', edit=dict(op='replace', span='current_thought', text='Thought: Check inventory.\n'))
    result = session.call('counterfactual_context', args)['data']
    state = session.state(result['state_id'])
    assert state['text'] == session.state('pair-0')['text']  # Free generation receives only the state.
    assert result['text'].endswith('Thought: Check inventory.\n')
    assert result['state_id'] != session.pair_states['pair-0']
    attention = session.call('attention_summary', dict(state=result['state_id'], layers=[0]))['data']
    assert any(s['label'] == 'current_thought' and s['tokens'] > 0 for s in attention['layers'][0]['spans'])
    scores = session.call('action_scores', dict(state=result['state_id'], candidates=['look'], include_greedy=False))['data']
    assert scores['framing_record']['prefix'] == 'Thought: Check inventory.\n'
    assert scores['framing_record']['source'] == 'teacher_counterfactual'
    del session.states[result['state_id']]
    assert session.call('counterfactual_context', args)['cache_hit']
    assert session.student.decision_prefix(session.state('pair-0')) == original
    deleted = session.call('counterfactual_context', dict(state=result['state_id'], edit=dict(op='delete', span='current_thought')))['data']
    assert deleted['framing_record']['effective'] == 'no_thought'


@pytest.mark.parametrize('framing', ['original_thought', 'greedy_thought', 'no_thought'])
def test_all_decision_tools_record_framing(session, framing, monkeypatch):
    monkeypatch.setattr(session.student, 'greedy', lambda state: dict(action='look',
        reply='Thought: A fresh check.\nAction: look', prompt_tokens=1, generated_tokens=1))
    ident = session.call('direction', dict(positive=[dict(state='pair-0', action='look')],
        negative=[dict(state='pair-1', action='inventory')], layers=[0], framing=framing))['data']['direction_id']
    requests = [
        ('logit_lens', dict(state='pair-0', a_f='look', a_r='inventory')),
        ('attention_summary', dict(state='pair-0', layers=[0])),
        ('patch', dict(target_state='pair-0', source_state='pair-1', layers=[0], positions=[dict(target=-1, source=-1)], candidates=['look'])),
        ('projection', dict(state='pair-0', action='look', direction_id=ident)),
        ('apply_edit', dict(state='pair-0', candidates=['look'], edit=dict(direction_id=ident, layers=[0], targets=['mlp.down_proj'], alpha=0)))]
    for name, args in requests:
        result = session.call(name, dict(args, framing=framing))
        assert result['framing'] == result['data']['framing'] == framing
    assert session.student.hidden_mean(session.state('pair-0'), 'look', [0], framing=framing)['framing'] == framing
