from copy import deepcopy
import json

import pytest

from repair import p0_agent
from repair.dry_fakes import FakeEnvironment
from repair.p0_pairs import CATEGORIES
from sed.config import load_config
from sed.storage import digest, write_json
from tests.p0_fakes import fake_pair, fake_greedy
from tests.test_p1_student import student
from tools.p0_pilot import ScriptedTeacherClient
from surgery.agent import (Agent, Budget, LIMITS, TOKEN_CAPS, Prompt, compact, parse_turn,
                           make_transport, integration_inputs)
from surgery.common import seal
from surgery.session import Session


def session(tmp_path, categories=('refused_action',)):
    s = student()
    s.greedy = fake_greedy
    pairs = [fake_pair(i + c * 6, category=cat) for c, cat in enumerate(categories) for i in range(6)]
    return Session(s, pairs, tmp_path / 'tools', env_factory=lambda p: FakeEnvironment())


def test_extended_parser_does_not_mutate_p0():
    value = json.dumps(dict(calls=[dict(tool='gradient_attribution', args=dict(pair_ids=['p']))]))
    assert parse_turn(value)[0]['tool'] == 'gradient_attribution'
    assert 'gradient_attribution' not in p0_agent.TOOL_DOCS
    with pytest.raises(ValueError):
        p0_agent.parse_turn(value)
    with pytest.raises(ValueError):
        parse_turn('{"calls":[],"calls":[]}')
    with pytest.raises(ValueError):
        parse_turn('{"submit":{"alpha":NaN}}')


@pytest.mark.parametrize('kind', LIMITS)
def test_registered_budgets(kind):
    b = Budget(kind)
    assert tuple(b.limits.values()) == LIMITS[kind]
    with pytest.raises(ValueError):
        Budget(kind, tool_calls=LIMITS[kind][0] + 1)


@pytest.mark.parametrize('limit', ['tool_calls', 'rollouts', 'candidate_edits', 'environment_steps'])
def test_each_budget_limit(limit):
    b = Budget(**{limit: 0})
    with pytest.raises(p0_agent.BudgetExceeded) as exc:
        b.reserve('rollout', dict(edit=[dict(type='scale_neurons', layers=[0], neurons={'0': [0]}, c=1)]))
    assert exc.value.budget == limit
    assert b.calls <= b.limits['tool_calls']


def test_training_and_submit_count_candidates():
    b = Budget(candidate_edits=1)
    b.reserve('local_finetune', dict(pair_ids=['p'], steps=1))
    with pytest.raises(p0_agent.BudgetExceeded):
        b.candidate([{'type': 'something_else'}])
    assert b.status()['candidate_edits'] == 1


def test_materialized_training_candidate_not_double_charged():
    b = Budget(candidate_edits=1)
    args = dict(pair_ids=['p'], steps=1)
    b.reserve('local_finetune', args)
    b.delta_candidates['frozen'] = b.training_candidate(args)
    b.reserve('local_finetune', dict(args, framing='original_thought'))
    b.candidate([dict(type='local_finetune', delta_id='frozen', alpha=1)])
    assert len(b.edits) == 1
    with pytest.raises(p0_agent.BudgetExceeded):
        b.candidate([dict(type='local_finetune', delta_id='frozen', alpha=0.5)])


@pytest.mark.parametrize('field', ['input_tokens', 'output_tokens'])
def test_each_shared_token_limit_independently(field, tmp_path):
    from repair.takeover import SharedTeacherBudget
    limits = dict(input_tokens=100, output_tokens=100)
    limits[field] = 1
    b = SharedTeacherBudget(tmp_path / 'budget.jsonl', **limits)
    with pytest.raises(Exception):
        b.reserve('request', input_bound=2 if field == 'input_tokens' else 1,
                  output_cap=2 if field == 'output_tokens' else 1)
    assert b.summary()['stop_reason']


def test_compaction_preserves_full_transcript_and_last_twenty():
    transcript = [dict(request=dict(tool='gradient_attribution', args={}),
        result=dict(data={'full': i}, summary=f'summary-{i}', sha256=str(i))) for i in range(25)]
    original = deepcopy(transcript)
    small = compact(transcript)
    assert transcript == original
    assert all(r['result']['compacted'] for r in small[:5])
    assert small[5:] == transcript[5:]
    assert all('data' not in r['result'] for r in small[:5])


def test_full_fake_teacher_transport_and_submission(tmp_path):
    sess = session(tmp_path)
    client = ScriptedTeacherClient([
        dict(calls=[dict(tool='rollout', args=dict(state='pair-0', edit=[]))]),
        dict(submit=dict(mechanism='No edit in this fixture.', edit=[]))])
    transport = make_transport(load_config('configs/base_e4b.yaml'), 'refused_action', tmp_path / 'transport', client=client)
    value = Agent(sess, transport, tmp_path / 'agent', category='refused_action', pairs_sha256='pairs').run()
    assert value['submission']['edit'] == []
    assert client.calls == 2
    assert value['budget']['rollouts'] == 1
    assert 'P1 preregistered' in client.requests[0]['input'][0]['content']
    assert 'gradient_attribution' in client.requests[0]['input'][0]['content']
    usage = [json.loads(row) for row in (tmp_path / 'transport/teacher_budget.jsonl').read_text().splitlines()]
    assert usage
    transcript = (tmp_path / 'agent/agent.jsonl').read_text()
    assert 'raw_output' in transcript and 'teacher_turn' in transcript


@pytest.mark.parametrize('kind,expected', [('category', (60_000_000, 3_000_000)), ('integration', (60_000_000, 3_000_000)), ('global_b', (100_000_000, 5_000_000))])
def test_token_caps(kind, expected, tmp_path):
    assert TOKEN_CAPS[kind] == expected
    with pytest.raises(ValueError):
        make_transport(load_config('configs/base_e4b.yaml'), kind, tmp_path, kind=kind, input_cap=expected[0] + 1)
    transport = make_transport(load_config('configs/base_e4b.yaml'), kind, tmp_path / 'tiny', kind=kind,
        input_cap=1, output_cap=1, client=ScriptedTeacherClient([dict(submit={})]))
    prompt = Prompt(dict(category=kind, pairs=[], submissions=[], transcript=[], budget={}))
    with pytest.raises(Exception):
        transport(prompt)


def frozen_categories(sess):
    return [seal(dict(schema='p1-submission/1', setting='A', kind='category', category=c,
        pairs_sha256='pairs', diagnosis_pair_ids=sorted(k for k, p in sess.pairs.items() if p['category'] == c),
        student_identity=sess.student.identity, submission=dict(mechanism='fixture', edit=[]),
        vectors={}, direction_specs={})) for c in CATEGORIES]


def test_integration_five_inputs_and_all_pair_gate(tmp_path):
    sess = session(tmp_path, CATEGORIES)
    submissions = frozen_categories(sess)
    assert len(integration_inputs(sess, submissions, 'pairs')) == 5
    for bad in (submissions[:4], [submissions[0]] * 5):
        with pytest.raises(ValueError):
            integration_inputs(sess, bad, 'pairs')
    broken = deepcopy(submissions)
    broken[0] = seal(dict(broken[0], pairs_sha256='other'))
    with pytest.raises(ValueError):
        integration_inputs(sess, broken, 'pairs')
    agent = Agent(sess, None, tmp_path / 'agent', category='integration', pairs_sha256='pairs',
        kind='integration', submissions=submissions)
    sess.rollout_edits.add(digest([]))
    with pytest.raises(ValueError, match='ALL'):
        agent.submit(dict(mechanism='fixture', edit=[]))
    sess.validated[digest([])] = set(sess.pairs)
    assert agent.submit(dict(mechanism='fixture', edit=[]))['kind'] == 'integration'


def test_integration_only_rescale_drop_combine(tmp_path):
    sess = session(tmp_path, CATEGORIES)
    agent = Agent(sess, None, tmp_path / 'agent', category='integration', pairs_sha256='pairs',
        kind='integration', submissions=frozen_categories(sess))
    with pytest.raises(ValueError, match='Integration'):
        agent.check_combination([dict(type='scale_neurons', layers=[0], neurons={'0': [1]}, c=2)])
    agent.execute_batch([dict(tool='action_scores', args=dict(state='pair-0', candidates=['look'])),
        dict(tool='local_finetune', args=dict(pair_ids=['pair-0'], modules=['0:mlp.down_proj'], rank=1, lr=1e-3, steps=1))], 0)
    assert agent.budget.calls == 2
    assert agent.transcript[0]['result']['data']['scores']
    assert agent.transcript[1]['result']['data']['status'] == 'rejected'
    assert not sess.deltas


def test_teacher_gate_before_load(tmp_path, monkeypatch):
    from tools import p1_surgery
    monkeypatch.setattr(p1_surgery, 'runtime', lambda *a, **k: pytest.fail('No GPU load before gate'))
    with pytest.raises(SystemExit):
        p1_surgery.main(['diagnose', '--output', str(tmp_path / 'out')])


def test_ninth_protocol_error_and_fifth_empty_halt(tmp_path):
    sess = session(tmp_path)
    agent = Agent(sess, None, tmp_path / 'agent', category='refused_action', pairs_sha256='pairs')
    for i in range(8):
        agent.protocol_error(ValueError('fixture'), i)
    with pytest.raises(p0_agent.BudgetExceeded):
        agent.protocol_error(ValueError('fixture'), 9)
    for i in range(4):
        agent.empty_reply(i)
    with pytest.raises(p0_agent.BudgetExceeded):
        agent.empty_reply(5)
