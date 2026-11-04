from copy import deepcopy
import json
from types import SimpleNamespace

import pytest

from repair.p0_agent import (AgentBudget, BudgetExceeded, P0Agent, AgentPrompt,
    make_transport, parse_turn, freeze_submission, load_submission, validate_submission,
    P0Teacher, ProviderClient)
from repair.p0_introspect import ToolSession
from repair.dry_fakes import FakeEnvironment
from sed.config import load_config
from sed.storage import write_json
from tools.p0_pilot import ScriptedTeacherClient
from p0_fakes import tiny_student, fake_pair, sample_submission, fake_greedy


@pytest.fixture
def session(tmp_path, monkeypatch):
    student = tiny_student()
    monkeypatch.setattr(student, 'greedy', fake_greedy)
    return ToolSession(student, [fake_pair(i) for i in range(4)], tmp_path / 'tools',
                       env_factory=lambda pair: FakeEnvironment())


def add_direction(session):
    return session.call('direction', dict(positive=[dict(state='pair-0', action='look')],
        negative=[dict(state='pair-1', action='inventory')], layers=[0]))['data']['direction_id']


@pytest.mark.parametrize('raw', ['{}', '[]', '', '{"tool":"action_scores","args":{},"x":1}',
    '{"tool":"shell","args":{}}', '{"tool":"rollout","tool":"direction","args":{}}',
    '{"submit":{"x":NaN}}', '{"tool":[],"args":{}}', '{"tool":{},"args":{}}',
    '{"tool":"rollout","args":[]}', '{"submit":[]}'])
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
        lambda p: dict(tool='rollout', args=dict(state='pair-0',
            edit=sample_submission(p['transcript'][0]['result']['data']['direction_id'])['edit'])),
        lambda p: dict(submit=sample_submission(p['transcript'][0]['result']['data']['direction_id']))]
    client = ScriptedTeacherClient(script)
    transport = make_transport(load_config('configs/base_e4b.yaml'), 'refused_action', tmp_path / 'transport', client=client)
    agent = P0Agent(session, transport, tmp_path / 'agent', category='refused_action', pairs_sha256='pairs', pairs_schema='p0-pairs/3')
    submission = agent.run()
    assert submission['budget']['tool_calls'] == 2
    assert client.calls == 3
    usage = transport.budget.summary()
    assert usage['reported_usage']['input_tokens'] == 300
    assert usage['reported_usage']['output_tokens'] == 120
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
        if len(seen) == 2:
            return json.dumps(dict(tool='rollout', args=dict(state='pair-0', edit=sample_submission(ident)['edit'])))
        return json.dumps(dict(submit=sample_submission(ident)))
    result = P0Agent(session, transport, tmp_path / 'agent', category='refused_action', pairs_sha256='pairs', pairs_schema='p0-pairs/3').run()
    assert result['budget']['tool_calls'] == 2


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
            assert kwargs['reasoning'] == {'effort': 'high'}
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


def test_azure_first_uses_existing_transport_injection(monkeypatch, tmp_path):
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
    client = ProviderClient(600, tmp_path / 'provider.json')
    assert client.create(model='gpt-6-luna', reasoning={'effort': 'high'}) == 'fake response'
    assert client.provider == 'azure' and called[0]['max_retries'] == 0
    assert json.loads((tmp_path / 'provider.json').read_text())['reasoning'] == {'effort': 'high'}


def test_incomplete_azure_never_silently_falls_back(monkeypatch):
    from teachers.openai_teacher import TeacherCallError
    monkeypatch.setenv('AZURE_OPENAI_ENDPOINT', 'https://fake.invalid')
    monkeypatch.delenv('AZURE_OPENAI_API_KEY', raising=False)
    with pytest.raises(TeacherCallError, match='Incomplete Azure'):
        ProviderClient(600).create(model='gpt-6-luna')


def scripted_agent(session, tmp_path, script, *, budget=None):
    client = ScriptedTeacherClient(script)
    transport = make_transport(load_config('configs/base_e4b.yaml'), 'refused_action',
                               tmp_path / 'transport', client=client)
    agent = P0Agent(session, transport, tmp_path / 'agent', category='refused_action',
        pairs_sha256='pairs', pairs_schema='p0-pairs/3', budget=budget)
    return agent, client, transport


@pytest.mark.parametrize('raw', [
    'I will first inspect the student.',
    '{"tool":"rollout","args":{}}{"tool":"action_scores","args":{}}',
    '{"tool":"shell","args":{}}',
    '{"tool":[],"args":{}}',
    '{"tool":"rollout","args":[]}',
    '{"submit":{}}',
])
def test_protocol_error_feedback_then_valid_call_continues(session, tmp_path, fast_clock, raw):
    submission = sample_submission(add_direction(session))
    def rollout(payload):
        assert len(payload['transcript']) == 1
        feedback = payload['transcript'][0]
        assert feedback['protocol_error']
        assert feedback['expected'] == 'exactly one JSON object {"tool":...,"args":...} or {"submit":...}'
        assert payload['budget']['tool_calls'] == 1
        return dict(tool='rollout', args=dict(state='pair-0', edit=submission['edit']))
    agent, client, transport = scripted_agent(session, tmp_path, [raw, rollout, dict(submit=submission)])
    result = agent.run()
    assert result['budget']['tool_calls'] == 2 and result['budget']['rollouts'] == 1
    assert agent.protocol_errors == 1 and client.calls == 3
    assert transport.budget.summary()['unsettled_attempts'] == 0
    rows = [json.loads(line) for line in agent.ledger.path.read_text().splitlines()]
    assert next(row for row in rows if row['event'] == 'teacher_turn')['raw_output'] == raw
    assert any(row['event'] == 'protocol_error' for row in rows)


def test_six_protocol_errors_halt_with_recorded_reason(session, tmp_path, fast_clock):
    agent, client, _ = scripted_agent(session, tmp_path, ['prose'] * 6)
    with pytest.raises(BudgetExceeded, match='sixth error'):
        agent.run()
    assert client.calls == agent.budget.calls == agent.protocol_errors == 6
    assert len(agent.transcript) == 6
    rows = [json.loads(line) for line in agent.ledger.path.read_text().splitlines()]
    assert rows[-1]['event'] == 'halted' and 'maximum 5' in rows[-1]['reason']
    assert len([row for row in rows if row['event'] == 'protocol_error']) == 6
    assert not (agent.directory / 'submission.json').exists()


def test_five_protocol_errors_allow_recovery(session, tmp_path, fast_clock):
    submission = sample_submission(add_direction(session))
    agent, client, _ = scripted_agent(session, tmp_path, ['prose'] * 5 + [
        dict(tool='rollout', args=dict(state='pair-0', edit=submission['edit'])), dict(submit=submission)])
    result = agent.run()
    assert agent.protocol_errors == 5 and client.calls == 7
    assert result['budget']['tool_calls'] == 6


def test_submit_without_rollout_rejected_then_rollout_and_submit(session, tmp_path, fast_clock):
    submission = sample_submission(add_direction(session))
    def rollout(payload):
        assert not (tmp_path / 'agent/submission.json').exists()
        feedback = payload['transcript'][-1]['result']
        assert feedback['status'] == 'rejected'
        assert 'completed rollout during this session' in feedback['submission_error']
        assert 'exact submitted edit spec' in feedback['submission_error']
        assert payload['budget']['tool_calls'] == payload['budget']['candidate_edits'] == 1
        return dict(tool='rollout', args=dict(state='pair-0', edit=submission['edit']))
    agent, client, _ = scripted_agent(session, tmp_path, [dict(submit=submission), rollout, dict(submit=submission)])
    result = agent.run()
    assert result['submission'] == submission and client.calls == 3
    assert result['budget']['tool_calls'] == 2 and result['budget']['candidate_edits'] == 1
    assert result['budget']['rollouts'] == 1 and result['budget']['environment_steps'] == 4
    assert agent.protocol_errors == 0


@pytest.mark.parametrize('different', [None, {'alpha': .6}, {'layers': [1]}, {'targets': ['self_attn.o_proj']}])
def test_submission_requires_exact_rollout_edit(session, tmp_path, fast_clock, different):
    ident = session.call('direction', dict(positive=[dict(state='pair-0', action='look')],
        negative=[dict(state='pair-1', action='inventory')], layers=[0, 1]))['data']['direction_id']
    submission = sample_submission(ident)
    other = None if different is None else dict(submission['edit'], **different)
    agent, _, _ = scripted_agent(session, tmp_path, [
        dict(tool='rollout', args=dict(state='pair-0', edit=other)), dict(submit=submission),
        dict(tool='rollout', args=dict(state='pair-0', edit=submission['edit'])), dict(submit=submission)])
    result = agent.run()
    assert agent.transcript[1]['result']['status'] == 'rejected'
    assert result['budget']['tool_calls'] == 3 and result['budget']['rollouts'] == 2
    assert result['budget']['candidate_edits'] == (1 if different is None else 2)


def test_cached_rollout_from_before_session_does_not_open_submission_gate(session, tmp_path, fast_clock):
    submission = sample_submission(add_direction(session))
    args = dict(state='pair-0', edit=submission['edit'])
    session.call('rollout', args)
    agent, _, _ = scripted_agent(session, tmp_path, [dict(tool='rollout', args=args), dict(submit=submission),
        dict(tool='rollout', args=dict(state='pair-1', edit=submission['edit'])), dict(submit=submission)])
    result = agent.run()
    assert agent.transcript[0]['result']['cache_hit']
    assert agent.transcript[1]['result']['status'] == 'rejected'
    assert result['budget']['environment_steps'] == 4


def test_rejected_rollout_does_not_open_submission_gate(session, tmp_path, fast_clock):
    submission = sample_submission(add_direction(session))
    agent, _, _ = scripted_agent(session, tmp_path, [
        dict(tool='rollout', args=dict(state='unknown', edit=submission['edit'])), dict(submit=submission),
        dict(tool='rollout', args=dict(state='pair-0', edit=submission['edit'])), dict(submit=submission)])
    result = agent.run()
    assert agent.transcript[0]['result']['data']['status'] == 'rejected'
    assert agent.transcript[1]['result']['status'] == 'rejected'
    assert result['budget']['rollouts'] == 2 and result['budget']['candidate_edits'] == 1


@pytest.mark.parametrize('path', ['/tmp/private/file.json', '"/tmp/private folder/file.json"', r'C:\private\file.json'])
def test_validation_feedback_includes_message_without_paths(session, tmp_path, fast_clock, monkeypatch, path):
    submission = sample_submission(add_direction(session))
    def reject(**kwargs):
        raise ValueError('Invalid zero-based layer indices; see ' + path + ' ' + 'details ' * 100)
    monkeypatch.setattr(session.student, 'attention_summary', reject)
    agent, _, _ = scripted_agent(session, tmp_path, [dict(tool='attention_summary', args=dict(state='pair-0', layers=[999])),
        dict(tool='rollout', args=dict(state='pair-0', edit=submission['edit'])), dict(submit=submission)])
    result = agent.run()
    feedback = agent.transcript[0]['result']['data']
    assert feedback['status'] == 'rejected' and feedback['error_type'] == 'ValueError'
    assert feedback['message'].startswith('Invalid zero-based layer indices; see <path>')
    assert path not in feedback['message'] and len(feedback['message']) == 300
    assert result['budget']['tool_calls'] == 2


def test_reasoning_high_in_every_request_and_transport_disclosure(session, tmp_path, fast_clock):
    submission = sample_submission(add_direction(session))
    agent, client, _ = scripted_agent(session, tmp_path, ['prose',
        dict(tool='rollout', args=dict(state='pair-0', edit=submission['edit'])), dict(submit=submission)])
    agent.run()
    assert len(client.requests) == 3
    assert all(request['reasoning'] == {'effort': 'high'} for request in client.requests)
    rows = [json.loads(line) for line in (tmp_path / 'transport/transport/usage.jsonl').read_text().splitlines()]
    # state_link rows bind cached transport identity; they are not API attempts.
    rows = [row for row in rows if row['status'] != 'state_link']
    assert len([row for row in rows if row['status'] == 'started']) == 3
    assert all(row['sampling']['reasoning'] == {'effort': 'high'} for row in rows)
    assert all(row['baseline_transport']['reasoning'] == {'effort': 'high'} for row in rows)


def test_reasoning_high_on_output_cap_retry(tmp_path, fast_clock):
    class TruncatedThenCompleted(ScriptedTeacherClient):
        def create(self, **kwargs):
            if not self.requests:
                self.requests.append(kwargs)
                cap = kwargs['max_output_tokens']
                response = SimpleNamespace(output_text='', model='FAKE_NO_MODEL', id='fake-truncated',
                    status='incomplete', incomplete_details={'reason': 'max_output_tokens'},
                    usage={'input_tokens': 100, 'output_tokens': cap,
                           'output_tokens_details': {'reasoning_tokens': cap}})
                return SimpleNamespace(status_code=200, headers={}, parse=lambda: response)
            return super().create(**kwargs)
    client = TruncatedThenCompleted([dict(tool='action_scores', args={})])
    transport = make_transport(load_config('configs/base_e4b.yaml'), 'refused_action', tmp_path, client=client)
    assert parse_turn(transport(AgentPrompt(dict(pairs=[], transcript=[]))))['tool'] == 'action_scores'
    assert [request['max_output_tokens'] for request in client.requests] == [16384, 32768]
    assert all(request['reasoning'] == {'effort': 'high'} for request in client.requests)
    assert transport.budget.summary()['unsettled_attempts'] == 0


def test_base_teacher_still_rejects_reasoning_sampling(tmp_path):
    from teachers.openai_teacher import Teacher
    with pytest.raises(ValueError, match='Unsupported sampling'):
        Teacher(cache_dir=tmp_path / 'cache', usage_ledger=tmp_path / 'usage.jsonl', run_id='base',
                sampling={'max_output_tokens': 32, 'reasoning': {'effort': 'high'}})


def test_reasoning_parameter_rejection_halts_without_fallback(tmp_path, fast_clock):
    from teachers.openai_teacher import TeacherCallError
    class UnsupportedParameter(Exception):
        status_code = 400
        body = {'error': {'code': 'unsupported_parameter', 'param': 'reasoning'}}
    class RejectReasoning(ScriptedTeacherClient):
        def create(self, **kwargs):
            self.requests.append(kwargs)
            self.calls += 1
            raise UnsupportedParameter('Raw SDK exception must never be retained')
    client = RejectReasoning([])
    transport = make_transport(load_config('configs/base_e4b.yaml'), 'refused_action', tmp_path, client=client)
    with pytest.raises(TeacherCallError, match='halted'):
        transport(AgentPrompt(dict(pairs=[], transcript=[])))
    assert client.calls == 1 and client.requests[0]['reasoning'] == {'effort': 'high'}
    assert transport.budget.summary()['stop_reason']
    rows = (tmp_path / 'transport/usage.jsonl').read_text()
    assert 'Raw SDK exception' not in rows


def test_cli_refuses_existing_diagnosis_directory_before_loading(tmp_path, monkeypatch):
    import tools.p0_pilot as pilot
    monkeypatch.setattr(pilot, 'load_pairs', lambda _: pytest.fail('Existing output must fail before loading'))
    with pytest.raises(SystemExit) as exc:
        pilot.main(['diagnose', '--output', str(tmp_path), '--pairs', 'unused.json',
                    '--category', 'refused_action', '--allow-teacher-api'])
    assert exc.value.code == 2
