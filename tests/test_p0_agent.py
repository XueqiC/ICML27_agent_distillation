from copy import deepcopy
import json
from types import SimpleNamespace

import pytest

from repair.p0_agent import (AgentBudget, BudgetExceeded, P0Agent, AgentPrompt,
    make_transport, parse_turn, freeze_submission, load_submission, validate_submission,
    P0Teacher, ProviderClient)
from repair.p0_introspect import ToolSession
from sed.config import load_config
from sed.storage import write_json
from tools.p0_pilot import ScriptedTeacherClient
from p0_fakes import tiny_student, fake_pair, sample_submission


@pytest.fixture
def session(tmp_path):
    return ToolSession(tiny_student(), [fake_pair(i) for i in range(4)], tmp_path / 'tools')


def add_direction(session):
    return session.call('direction', dict(positive=[dict(state='pair-0', action='look')],
        negative=[dict(state='pair-1', action='inventory')], layers=[0]))['data']['direction_id']


@pytest.mark.parametrize('raw', ['{}', '[]', '', '{"tool":"action_scores","args":{},"x":1}',
    '{"tool":"shell","args":{}}', '{"tool":"rollout","tool":"direction","args":{}}',
    '{"submit":{"x":NaN}}'])
def test_strict_json_protocol_rejects_invalid(raw):
    with pytest.raises(ValueError):
        parse_turn(raw)


def test_tool_call_budget():
    budget = AgentBudget(tool_calls=2)
    budget.reserve('action_scores', {})
    budget.reserve('action_scores', {})
    with pytest.raises(BudgetExceeded, match='Tool-call'):
        budget.reserve('action_scores', {})


def test_rollout_budget_and_real_step_reservation():
    budget = AgentBudget(rollouts=1)
    budget.reserve('rollout', {})
    with pytest.raises(BudgetExceeded, match='Rollout'):
        budget.reserve('rollout', {})
    budget = AgentBudget(environment_steps=39)
    with pytest.raises(BudgetExceeded, match='environment step'):
        budget.reserve('rollout', {})


def test_edit_budget_includes_rollout_candidates_and_submit():
    budget = AgentBudget(candidate_edits=1)
    budget.reserve('rollout', {'edit': {'alpha': .5}})
    budget.reserve('apply_edit', {'edit': {'alpha': .5}})
    assert len(budget.edits) == 1
    with pytest.raises(BudgetExceeded, match='Candidate'):
        budget.candidate({'alpha': .6})


def test_submission_freezes_direction_and_detects_tampering(session, tmp_path):
    ident = add_direction(session)
    p = tmp_path / 'submission.json'
    value = freeze_submission(p, sample_submission(ident), session, category='refused_action',
                              pairs_sha256='pairs', budget={})
    assert load_submission(p, pairs_sha256='pairs') == value
    with pytest.raises(FileExistsError):
        freeze_submission(p, sample_submission(ident), session, category='refused_action', pairs_sha256='pairs', budget={})
    changed = deepcopy(value)
    changed['submission']['edit']['alpha'] = 999
    write_json(p, changed)
    with pytest.raises(ValueError, match='changed'):
        load_submission(p)


@pytest.mark.parametrize('bad', [dict(predictions=[]), dict(edit={'arbitrary_code': 'bad'}),
    dict(mechanism=''), dict(predictions=[{'metric': 'unsupported', 'direction': '>', 'threshold': 0}])])
def test_submission_schema_rejects_unsupported(session, bad):
    value = sample_submission(add_direction(session))
    value.update(bad)
    with pytest.raises(ValueError):
        validate_submission(value, session)


def test_agent_no_test_access_and_no_budget_reset(session, tmp_path):
    script = [dict(tool='direction', args=dict(positive=[dict(state='pair-0', action='look')],
        negative=[dict(state='pair-1', action='inventory')], layers=[0])),
        lambda p: dict(submit=sample_submission(p['transcript'][0]['result']['data']['direction_id']))]
    client = ScriptedTeacherClient(script)
    transport = make_transport(load_config('configs/base_e4b.yaml'), 'refused_action', tmp_path / 'transport', client=client)
    agent = P0Agent(session, transport, tmp_path / 'agent', category='refused_action', pairs_sha256='pairs', pairs_schema='p0-pairs/3')
    submission = agent.run()
    assert submission['budget']['tool_calls'] == 1
    assert client.calls == 2
    usage = transport.budget.summary()
    assert usage['reported_usage']['input_tokens'] == 200
    assert usage['reported_usage']['output_tokens'] == 80
    assert usage['unsettled_attempts'] == 0
    with pytest.raises(ValueError, match='already used'):
        agent.run()
    raw = (tmp_path / 'agent/agent.jsonl').read_text()
    assert 'internal_state' not in raw and 'audit_sha256' not in raw


def test_agent_rejects_unresolved_or_wrong_split(session, tmp_path):
    session.pairs['pair-0']['a_r'] = None
    with pytest.raises(ValueError, match='Resolve'):
        P0Agent(session, lambda p: pytest.fail('API'), tmp_path, category='refused_action', pairs_sha256='pairs', pairs_schema='p0-pairs/3')
    session.scope = 'evaluation'
    with pytest.raises(ValueError, match='diagnosis'):
        P0Agent(session, lambda p: None, tmp_path, category='refused_action', pairs_sha256='pairs', pairs_schema='p0-pairs/3')


def test_agent_v3_gate_accepts_whole_group_category_over_four(session, tmp_path):
    session.pairs['pair-4'] = fake_pair(4)
    agent = P0Agent(session, lambda p: pytest.fail('API'), tmp_path,
        category='refused_action', pairs_sha256='pairs', pairs_schema='p0-pairs/3')
    assert len(agent.session.pairs) == 5


@pytest.mark.parametrize('schema', ['p0-pairs/1', 'p0-pairs/2'])
def test_agent_old_freeze_refused_even_with_valid_actions(session, tmp_path, schema):
    with pytest.raises(ValueError, match='v1/v2 are refused'):
        P0Agent(session, lambda p: pytest.fail('API'), tmp_path,
            category='refused_action', pairs_sha256='pairs', pairs_schema=schema)


@pytest.mark.parametrize('alignment', ['first_divergence', 'aligned_patch_boundary', 'unresolved_local_repair', None])
def test_agent_gate_requires_every_pair_first_divergence(session, tmp_path, alignment):
    session.pairs['pair-0']['alignment'] = alignment
    with pytest.raises(ValueError, match='first-divergence'):
        P0Agent(session, lambda p: pytest.fail('API'), tmp_path,
            category='refused_action', pairs_sha256='pairs', pairs_schema='p0-pairs/3')


def test_agent_gate_refuses_fewer_than_four(session, tmp_path):
    session.pairs.pop('pair-0')
    with pytest.raises(ValueError, match='at least four'):
        P0Agent(session, lambda p: pytest.fail('API'), tmp_path,
            category='refused_action', pairs_sha256='pairs', pairs_schema='p0-pairs/3')


def test_rejected_tool_is_counted_and_returned_to_teacher(session, tmp_path):
    ident = add_direction(session)
    seen = []
    def transport(prompt):
        seen.append(prompt.payload)
        if len(seen) == 1:
            return json.dumps(dict(tool='action_scores', args=dict(state='test-forbidden', candidates=['look'])))
        assert prompt.payload['transcript'][0]['result']['data']['status'] == 'rejected'
        return json.dumps(dict(submit=sample_submission(ident)))
    result = P0Agent(session, transport, tmp_path / 'agent', category='refused_action', pairs_sha256='pairs', pairs_schema='p0-pairs/3').run()
    assert result['budget']['tool_calls'] == 1


@pytest.fixture
def fast_clock(monkeypatch):
    from test_stage2c_teacher import FakeClock
    from teachers.rate_limit import RateLimiter
    clock = FakeClock()
    original = P0Teacher.__init__
    def init(self, **kwargs):
        original(self, **kwargs)
        self.clock, self.sleep, self.jitter = clock, clock.sleep, lambda low, high: 0
    monkeypatch.setattr(P0Teacher, '__init__', init)
    original_limiter = RateLimiter.__init__
    def limiter(self, *args, **kwargs):
        original_limiter(self, *args, **dict(kwargs, clock=clock, sleep=clock.sleep))
    monkeypatch.setattr(RateLimiter, '__init__', limiter)
    return clock


def test_429_storm_probes_once_then_halts_loudly(tmp_path, fast_clock):
    from test_capped_teacher import http_429
    from teachers.openai_teacher import TeacherCallError
    class Storm:
        max_retries = 0
        def __init__(self):
            self.calls = 0
            self.responses = self
            self.with_raw_response = self
        def create(self, **kwargs):
            self.calls += 1
            raise http_429()
    client = Storm()
    transport = make_transport(load_config('configs/base_e4b.yaml'), 'refused_action', tmp_path, client=client)
    with pytest.raises(TeacherCallError):
        transport(AgentPrompt(dict(pairs=[], transcript=[])))
    assert client.calls == 2
    assert transport.budget.summary()['stop_reason']
    rows = [json.loads(l) for l in (tmp_path / 'transport/usage.jsonl').read_text().splitlines()]
    assert len([r for r in rows if r['status'] == 'failed']) == 2
    assert all(r['usage'] is None for r in rows)


def test_runaway_token_cap_prevents_transport(tmp_path):
    from teachers.openai_teacher import TeacherCallError
    client = ScriptedTeacherClient([dict(submit={})])
    transport = make_transport(load_config('configs/base_e4b.yaml'), 'refused_action', tmp_path,
                               client=client, input_cap=1, output_cap=1)
    with pytest.raises(Exception):
        transport(AgentPrompt(dict(pairs=[], transcript=[])))
    assert client.calls == 0


def test_azure_first_uses_existing_transport_injection(monkeypatch):
    import openai
    called = []
    class FakeSDK:
        def __init__(self, **kwargs):
            called.append(kwargs)
            self.responses = self
            self.with_raw_response = self
        def create(self, **kwargs):
            return 'fake response'
    monkeypatch.setenv('AZURE_OPENAI_ENDPOINT', 'https://fake.invalid')
    monkeypatch.setenv('AZURE_OPENAI_API_KEY', 'fake-key')
    monkeypatch.setenv('AZURE_OPENAI_API_VERSION', 'explicit-fixture-version')
    monkeypatch.setattr(openai, 'AzureOpenAI', FakeSDK)
    monkeypatch.setattr(openai, 'OpenAI', lambda **k: pytest.fail('Azure must be first'))
    client = ProviderClient(600)
    assert client.create(model='gpt-6-luna') == 'fake response'
    assert client.provider == 'azure' and called[0]['max_retries'] == 0


def test_incomplete_azure_never_silently_falls_back(monkeypatch):
    from teachers.openai_teacher import TeacherCallError
    monkeypatch.setenv('AZURE_OPENAI_ENDPOINT', 'https://fake.invalid')
    monkeypatch.delenv('AZURE_OPENAI_API_KEY', raising=False)
    with pytest.raises(TeacherCallError, match='Incomplete Azure'):
        ProviderClient(600).create(model='gpt-6-luna')
