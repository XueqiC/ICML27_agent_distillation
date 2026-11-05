from copy import deepcopy
import json
from types import SimpleNamespace

import pytest

from repair.p0_agent import (AgentBudget, BudgetExceeded, P0Agent, AgentPrompt,
    make_transport, parse_turn, freeze_submission, load_submission, validate_submission,
    P0Teacher, ProviderClient, EXPECTED_TURN, SYSTEM, TOOL_DOCS, PROMPT_LAYOUT,
    SUBMIT_ONLY_NOTICE)
from repair.p0_introspect import ToolSession
from repair.dry_fakes import FakeEnvironment
from sed.config import load_config
from sed.storage import write_json, canonical_bytes, digest
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
    '{"tool":"rollout","args":[]}', '{"submit":[]}',
    '[{"tool":"rollout","args":{}},false]',
    '{"tool":"rollout","args":{}}, {"tool":"rollout","args":{}}',
    '[{"tool":"rollout","args":{}}] {"tool":"rollout","args":{}}',
    '[{"tool":"rollout","args":{}}] trailing prose',
    '{"tool":"rollout","args":{}} {"tool":"rollout","args":{"x":1,"x":2}}',
    '[{"tool":"rollout","args":{}},{"tool":"rollout","args":{"x":Infinity}}]',
    '{"tool":"rollout","args":{"x":-Infinity}}',
    '{"tool":"rollout","args":{"x":1e999}}'])
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


def empty_prompt():
    return AgentPrompt(dict(category='refused_action', pairs=[], transcript=[], budget=AgentBudget().status()))


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
        transport(empty_prompt())
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
        transport(empty_prompt())
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


def batch_reply(messages, form):
    if form == 'array':
        return json.dumps(messages)
    separator = {'concatenated': '', 'newlines': '\n', 'whitespace': ' \t\r\n '}[form]
    return ' \n' + separator.join(json.dumps(message) for message in messages) + '\t '


@pytest.mark.parametrize('form', ['concatenated', 'newlines', 'whitespace', 'array'])
def test_batch_executes_in_order_and_returns_each_result(session, tmp_path, fast_clock, form):
    calls = [dict(tool='action_scores', args=dict(state='pair-0', candidates=['look'])),
             dict(tool='direction', args=dict(positive=[dict(state='pair-0', action='look')],
                 negative=[dict(state='pair-1', action='inventory')], layers=[0]))]
    def rollout(payload):
        assert [item['request'] for item in payload['transcript']] == calls
        assert [item['result']['tool'] for item in payload['transcript']] == ['action_scores', 'direction']
        assert payload['budget']['tool_calls'] == 2
        ident = payload['transcript'][1]['result']['data']['direction_id']
        return dict(tool='rollout', args=dict(state='pair-0', edit=sample_submission(ident)['edit']))
    def submit(payload):
        return dict(submit=sample_submission(payload['transcript'][1]['result']['data']['direction_id']))
    agent, client, transport = scripted_agent(session, tmp_path, [batch_reply(calls, form), rollout, submit])
    result = agent.run()
    assert agent.protocol_errors == 0 and client.calls == 3
    assert result['budget']['tool_calls'] == 3 and result['budget']['rollouts'] == 1
    rows = [json.loads(line) for line in agent.ledger.path.read_text().splitlines()]
    reservations = [row for row in rows if row['event'] == 'tool_budget']
    assert [(row['turn'], row['call_index'], row['budget']['tool_calls']) for row in reservations] == [
        (0, 0, 1), (0, 1, 2), (1, 0, 3)]
    assert transport.budget.summary()['unsettled_attempts'] == 0


@pytest.mark.parametrize('form', ['concatenated', 'array'])
def test_eight_call_batch_can_use_last_tool_slot_then_submit(session, tmp_path, fast_clock, form):
    submission = sample_submission(add_direction(session))
    calls = [dict(tool='action_scores', args=dict(state='pair-0', candidates=['look']))] * 7
    calls += [dict(tool='rollout', args=dict(state='pair-0', edit=submission['edit']))]
    agent, client, _ = scripted_agent(session, tmp_path,
        [batch_reply(calls, form), dict(submit=submission)], budget=AgentBudget(tool_calls=8))
    result = agent.run()
    assert result['budget']['tool_calls'] == 8 and client.calls == 2
    assert len(agent.transcript) == 9 and agent.protocol_errors == 0
    assert agent.transcript[-1] == dict(notice=SUBMIT_ONLY_NOTICE)
    assert all(item['result']['cache_hit'] for item in agent.transcript[1:7])


@pytest.mark.parametrize('limits,resource', [
    (dict(rollouts=1), 'rollouts'),
    (dict(candidate_edits=1), 'candidate_edits'),
    (dict(environment_steps=43), 'environment_steps'),
])
def test_resource_budget_rejects_one_call_and_continues_batch(session, tmp_path, fast_clock, limits, resource):
    submission = sample_submission(add_direction(session))
    calls = [dict(tool='rollout', args=dict(state='pair-0', edit=submission['edit'])),
             dict(tool='rollout', args=dict(state='pair-1', edit=dict(submission['edit'], alpha=.6))),
             dict(tool='action_scores', args=dict(state='pair-0', candidates=['look']))]
    agent, client, _ = scripted_agent(session, tmp_path, [batch_reply(calls, 'concatenated'), dict(submit=submission)],
                                     budget=AgentBudget(**limits))
    result = agent.run()
    assert client.calls == 2 and result['budget']['tool_calls'] == 3 and agent.protocol_errors == 0
    assert session.environment_steps == agent.budget.steps == 4
    assert [item['request'] for item in agent.transcript] == calls
    assert agent.transcript[1]['result'] == dict(status='rejected', budget_error=resource)
    assert agent.transcript[2]['result']['tool'] == 'action_scores'
    rows = [json.loads(line) for line in agent.ledger.path.read_text().splitlines()]
    assert len([row for row in rows if row['event'] == 'tool_feedback']) == 3
    assert not any(row['event'] in ('batch_stopped', 'halted') for row in rows)


def test_eleventh_rollout_rejected_and_run_continues(session, tmp_path, fast_clock):
    submission = sample_submission(add_direction(session))
    rollout = dict(tool='rollout', args=dict(state='pair-0', edit=submission['edit']))
    score = dict(tool='action_scores', args=dict(state='pair-0', candidates=['look']))
    agent, client, _ = scripted_agent(session, tmp_path, [batch_reply([rollout] * 8, 'array'),
        batch_reply([rollout] * 3 + [score], 'array'), dict(submit=submission)])
    result = agent.run()
    assert client.calls == 3 and result['budget']['rollouts'] == 10
    assert result['budget']['tool_calls'] == 12 and result['budget']['environment_steps'] == 4
    assert agent.transcript[10]['result'] == dict(status='rejected', budget_error='rollouts')
    assert agent.transcript[11]['result']['tool'] == 'action_scores'
    assert all(item['result']['cache_hit'] for item in agent.transcript[1:10])


def test_400_environment_steps_rejects_next_rollout_and_continues(session, tmp_path, fast_clock, monkeypatch):
    submission = sample_submission(add_direction(session))
    # Ten distinct, actually executed fake-environment rollouts reach H=40;
    # an eleventh rollout slot isolates the real-step ceiling from rollout count.
    monkeypatch.setattr(session.student, 'greedy', lambda state:
        dict(action='look', reply='Action: look', prompt_tokens=1, generated_tokens=1))
    calls = [dict(tool='rollout', args=dict(state=f'pair-{i % 4}',
        edit=dict(submission['edit'], alpha=.5 - (i // 4) / 10))) for i in range(10)]
    blocked = dict(tool='rollout', args=dict(state='pair-3', edit=submission['edit']))
    score = dict(tool='action_scores', args=dict(state='pair-0', candidates=['look']))
    agent, _, _ = scripted_agent(session, tmp_path, [batch_reply(calls[:8], 'array'),
        batch_reply(calls[8:] + [blocked, score], 'array'), dict(submit=submission)],
        budget=AgentBudget(rollouts=11))
    result = agent.run()
    assert session.environment_steps == result['budget']['environment_steps'] == 400
    assert result['budget']['rollouts'] == 10 and result['budget']['tool_calls'] == 12
    assert agent.transcript[10]['result'] == dict(status='rejected', budget_error='environment_steps')
    assert agent.transcript[11]['result']['tool'] == 'action_scores'


@pytest.mark.parametrize('tool', ['apply_edit', 'rollout', 'submit'])
def test_sixth_distinct_edit_rejected_and_run_continues(session, tmp_path, fast_clock, tool):
    submission = sample_submission(add_direction(session))
    calls = [dict(tool='rollout', args=dict(state='pair-0', edit=submission['edit']))]
    calls += [dict(tool='apply_edit', args=dict(state='pair-0', candidates=['look'],
        edit=dict(submission['edit'], alpha=alpha))) for alpha in (.1, .2, .3, .4)]
    sixth = dict(submission['edit'], alpha=.6)
    if tool == 'submit':
        rejected = dict(submit=dict(submission, edit=sixth))
    else:
        args = dict(state='pair-0', edit=sixth)
        if tool == 'apply_edit':
            args['candidates'] = ['look']
        rejected = dict(tool=tool, args=args)
    score = dict(tool='action_scores', args=dict(state='pair-0', candidates=['look']))
    script = ([batch_reply(calls, 'array'), rejected, score] if tool == 'submit' else
              [batch_reply(calls + [rejected, score], 'array')])
    agent, _, _ = scripted_agent(session, tmp_path, script + [dict(submit=submission)])
    result = agent.run()
    assert result['budget']['candidate_edits'] == 5 and result['budget']['tool_calls'] == 7
    assert agent.protocol_errors == 0 and result['budget']['environment_steps'] == 4
    assert agent.transcript[5] == dict(request=rejected,
        result=dict(status='rejected', budget_error='candidate_edits'))
    assert agent.transcript[6]['result']['tool'] == 'action_scores'


def test_tool_budget_skips_each_remaining_call_then_allows_submit(session, tmp_path, fast_clock):
    submission = sample_submission(add_direction(session))
    calls = [dict(tool='rollout', args=dict(state='pair-0', edit=submission['edit'])),
             dict(tool='rollout', args=dict(state='pair-1', edit=dict(submission['edit'], alpha=.6))),
             dict(tool='action_scores', args=dict(state='pair-0', candidates=['look']))]
    def submit(payload):
        assert [item['request'] for item in payload['transcript'][:-1]] == calls
        assert payload['transcript'][-1] == dict(notice=SUBMIT_ONLY_NOTICE)
        for item in payload['transcript'][1:-1]:
            assert item['result'] == dict(status='not_executed', budget_error='tool_calls')
        assert payload['budget']['tool_calls'] == payload['budget']['rollouts'] == 1
        assert payload['budget']['candidate_edits'] == 1
        return dict(submit=submission)
    agent, client, _ = scripted_agent(session, tmp_path, [batch_reply(calls, 'array'), submit],
                                     budget=AgentBudget(tool_calls=1))
    agent.run()
    assert client.calls == 2 and session.environment_steps == agent.budget.steps == 4
    rows = [json.loads(line) for line in agent.ledger.path.read_text().splitlines()]
    stopped = [row for row in rows if row['event'] == 'batch_stopped']
    assert len(stopped) == 1 and stopped[0]['skipped_tool_calls'] == 2
    assert len([row for row in rows if row['event'] == 'tool_feedback']) == 3
    assert not any(row['event'] == 'halted' for row in rows)


@pytest.mark.parametrize('last_slot', ['call', 'rejected_call', 'resource_rejection',
                                     'protocol_error', 'empty_reply', 'rejected_submit'])
def test_sixtieth_slot_enters_submit_only_and_valid_submit_freezes(session, tmp_path, fast_clock, last_slot):
    submission = sample_submission(add_direction(session))
    rollout = dict(tool='rollout', args=dict(state='pair-0', edit=submission['edit']))
    score = dict(tool='action_scores', args=dict(state='pair-0', candidates=['look']))
    calls = [rollout] + [score] * 58
    last = {'call': score, 'rejected_call': dict(tool='action_scores', args={}),
            'resource_rejection': rollout, 'protocol_error': 'prose', 'empty_reply': '',
            'rejected_submit': dict(submit=dict(submission, edit=dict(submission['edit'], alpha=.6)))}[last_slot]
    def submit(payload):
        assert payload['budget']['tool_calls'] == 60
        assert payload['transcript'][-1] == dict(notice=SUBMIT_ONLY_NOTICE)
        return dict(submit=submission)
    script = [batch_reply(calls[i:i + 8], 'array') for i in range(0, len(calls), 8)] + [last, submit]
    agent, client, _ = scripted_agent(session, tmp_path, script, budget=AgentBudget(rollouts=1))
    result = agent.run()
    assert result['budget']['tool_calls'] == 60 and client.calls == len(script)
    assert load_submission(agent.directory / 'submission.json') == result


def test_submit_only_tools_and_protocol_error_cost_nothing_then_third_reply_can_submit(session, tmp_path, fast_clock):
    submission = sample_submission(add_direction(session))
    calls = [dict(tool='rollout', args=dict(state='pair-1', edit=dict(submission['edit'], alpha=.6))),
             dict(tool='action_scores', args=dict(state='pair-0', candidates=['look']))]
    exhausted = None
    def tools(payload):
        nonlocal exhausted
        exhausted = deepcopy(payload['budget'])
        assert payload['transcript'][-1] == dict(notice=SUBMIT_ONLY_NOTICE)
        return batch_reply(calls, 'array')
    def malformed(payload):
        assert payload['budget'] == exhausted
        assert [item['request'] for item in payload['transcript'][-2:]] == calls
        assert all(item['result'] == dict(status='not_executed', budget_error='tool_calls')
                   for item in payload['transcript'][-2:])
        return 'prose'
    def submit(payload):
        assert payload['budget'] == exhausted
        assert payload['transcript'][-1]['protocol_error']
        return dict(submit=submission)
    agent, client, _ = scripted_agent(session, tmp_path, [
        dict(tool='rollout', args=dict(state='pair-0', edit=submission['edit'])), tools, malformed, submit],
        budget=AgentBudget(tool_calls=1))
    result = agent.run()
    assert client.calls == 4 and result['budget'] == exhausted
    assert agent.protocol_errors == 1


@pytest.mark.parametrize('reply', ['tools', 'protocol_error', 'empty_reply', 'gated_submit', 'invalid_submit'])
def test_submit_only_halts_after_three_replies_without_valid_submission(session, tmp_path, fast_clock, reply):
    submission = sample_submission(add_direction(session))
    responses = {'tools': dict(tool='rollout', args=dict(state='pair-1', edit=submission['edit'])),
                 'protocol_error': 'prose', 'empty_reply': '',
                 'invalid_submit': dict(submit=dict(submission, predictions=[],
                     edit=dict(submission['edit'], alpha=.6))),
                 'gated_submit': dict(submit=dict(submission, edit=dict(submission['edit'], alpha=.6)))}
    agent, client, transport = scripted_agent(session, tmp_path, [
        dict(tool='rollout', args=dict(state='pair-0', edit=submission['edit']))] + [responses[reply]] * 3,
        budget=AgentBudget(tool_calls=1))
    reason = 'no valid submission after tool budget exhausted'
    with pytest.raises(BudgetExceeded, match=reason):
        agent.run()
    assert client.calls == 4 and agent.budget.calls == agent.budget.rollouts == 1
    assert len(agent.budget.edits) == 1
    assert agent.budget.steps == session.environment_steps == 4
    assert agent.transcript[1] == dict(notice=SUBMIT_ONLY_NOTICE)
    assert len(agent.transcript) == 5
    if reply == 'gated_submit':
        assert all(item['result']['status'] == 'rejected' and 'exact submitted edit' in
                   item['result']['submission_error'] for item in agent.transcript[-3:])
    elif reply in ('protocol_error', 'invalid_submit'):
        assert agent.protocol_errors == 3
    elif reply == 'empty_reply':
        assert agent.consecutive_empty_replies == 3 and agent.protocol_errors == 0
    rows = [json.loads(line) for line in agent.ledger.path.read_text().splitlines()]
    assert rows[-1]['event'] == 'halted' and rows[-1]['reason'] == reason
    assert not (agent.directory / 'submission.json').exists()
    assert transport.budget.summary()['unsettled_attempts'] == 0


@pytest.mark.parametrize('raw,limit,reason', [('prose', 9, 'ninth error'), ('', 5, '5 consecutive empty replies')])
@pytest.mark.parametrize('spare_slots', [0, -1, 1])
def test_error_limits_still_halt_at_and_across_tool_exhaustion(session, tmp_path, fast_clock, raw, limit, reason, spare_slots):
    agent, client, _ = scripted_agent(session, tmp_path, [raw] * limit,
        budget=AgentBudget(tool_calls=limit + spare_slots))
    with pytest.raises(BudgetExceeded, match=reason):
        agent.run()
    assert client.calls == limit and agent.budget.calls == min(limit, limit + spare_slots)
    rows = [json.loads(line) for line in agent.ledger.path.read_text().splitlines()]
    assert rows[-1]['event'] == 'halted' and reason in rows[-1]['reason']


@pytest.mark.parametrize('form', ['concatenated', 'newlines', 'array'])
@pytest.mark.parametrize('invalid', ['nine_calls', 'submit_first', 'submit_last', 'duplicate_key', 'nonfinite'])
def test_invalid_batch_is_one_protocol_error_with_nothing_executed(session, tmp_path, fast_clock, form, invalid):
    submission = sample_submission(add_direction(session))
    call = dict(tool='action_scores', args=dict(state='pair-0', candidates=['look']))
    calls = [call] * 9 if invalid == 'nine_calls' else [call, dict(submit=submission)]
    if invalid == 'submit_first':
        calls.reverse()
    raw = batch_reply(calls, form)
    if invalid in ('duplicate_key', 'nonfinite'):
        malformed = ('{"tool":"rollout","args":{"state":"pair-0","state":"pair-1"}}'
                     if invalid == 'duplicate_key' else '{"tool":"rollout","args":{"edit":{"alpha":NaN}}}')
        raw = '[' + json.dumps(call) + ',' + malformed + ']' if form == 'array' else json.dumps(call) + '\n' + malformed
    def recovery(payload):
        assert len(payload['transcript']) == 1 and payload['transcript'][0]['protocol_error']
        assert payload['budget']['tool_calls'] == 1
        assert payload['budget']['rollouts'] == payload['budget']['candidate_edits'] == 0
        return dict(tool='rollout', args=dict(state='pair-0', edit=submission['edit']))
    agent, client, _ = scripted_agent(session, tmp_path, [raw, recovery, dict(submit=submission)])
    result = agent.run()
    assert agent.protocol_errors == 1 and client.calls == 3
    assert result['budget']['tool_calls'] == 2
    rows = [json.loads(line) for line in agent.ledger.path.read_text().splitlines()]
    assert not any(row['event'] == 'tool_feedback' and row['turn'] == 0 for row in rows)


@pytest.mark.parametrize('bad_call,feedback', [
    (dict(tool='action_scores', args=dict(state='<STATE_ID_FROM_PREVIOUS_RESULT>', candidates=['look'])), 'Placeholder'),
    (dict(tool='action_scores', args=dict(state='unknown-state', candidates=['look'])), 'unknown-state'),
    (dict(tool='projection', args=dict(state='pair-0', action='look', direction_id='unknown-direction')), 'Direction'),
    (dict(tool='apply_edit', args=dict(state='pair-0', candidates=['look'],
        edit=dict(direction_id='<DIRECTION_ID>', layers=[0], targets=['mlp.down_proj'], alpha=.5))), 'Placeholder'),
    (dict(tool='direction', args=dict(positive=[dict(state='<STATE>', action='look')],
        negative=[dict(state='pair-1', action='inventory')], layers=[0])), 'Placeholder'),
    (dict(tool='counterfactual_context', args=dict(state='pair-0',
        edit=dict(op='replace', span='current_thought', text='Thought: <INSERT_TEXT>\n'))), 'Placeholder'),
    (dict(tool='counterfactual_context', args=dict(state='pair-0',
        edit=dict(op='delete', span='unknown-span'))), 'unknown-span'),
])
def test_bad_argument_in_batch_rejected_but_later_calls_run(session, tmp_path, fast_clock, bad_call, feedback):
    submission = sample_submission(add_direction(session))
    calls = [dict(tool='counterfactual_context', args=dict(state='pair-0',
                 edit=dict(op='replace', span='current_thought', text='Thought: Check the room.\n'))),
             bad_call, dict(tool='rollout', args=dict(state='pair-0', edit=submission['edit']))]
    def submit(payload):
        assert [item['request'] for item in payload['transcript']] == calls
        assert payload['transcript'][0]['result']['data']['state_id'] in session.states
        rejected = payload['transcript'][1]['result']['data']
        assert rejected['status'] == 'rejected' and feedback in rejected['message']
        assert payload['transcript'][2]['result']['data']['verification'] == 'real_environment'
        return dict(submit=submission)
    agent, client, _ = scripted_agent(session, tmp_path, [batch_reply(calls, 'array'), submit])
    result = agent.run()
    assert result['budget']['tool_calls'] == 3 and client.calls == 2 and agent.protocol_errors == 0


@pytest.mark.parametrize('raw', [
    'I will first inspect the student.',
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
        assert feedback['expected'] == EXPECTED_TURN
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


def test_ninth_protocol_error_halts_with_recorded_reason(session, tmp_path, fast_clock):
    agent, client, _ = scripted_agent(session, tmp_path, ['prose'] * 9)
    with pytest.raises(BudgetExceeded, match='ninth error'):
        agent.run()
    assert client.calls == agent.budget.calls == agent.protocol_errors == 9
    assert len(agent.transcript) == 9
    rows = [json.loads(line) for line in agent.ledger.path.read_text().splitlines()]
    assert rows[-1]['event'] == 'halted' and 'maximum 8' in rows[-1]['reason']
    assert len([row for row in rows if row['event'] == 'protocol_error']) == 9
    assert not (agent.directory / 'submission.json').exists()


def test_eight_protocol_errors_allow_recovery(session, tmp_path, fast_clock):
    submission = sample_submission(add_direction(session))
    agent, client, _ = scripted_agent(session, tmp_path, ['prose'] * 8 + [
        dict(tool='rollout', args=dict(state='pair-0', edit=submission['edit'])), dict(submit=submission)])
    result = agent.run()
    assert agent.protocol_errors == 8 and client.calls == 10
    assert result['budget']['tool_calls'] == 9


@pytest.mark.parametrize('raw', ['', ' \n\t\r '])
def test_empty_reply_consumes_slot_but_no_protocol_error(session, tmp_path, fast_clock, raw):
    submission = sample_submission(add_direction(session))
    def rollout(payload):
        assert payload['transcript'] == [dict(protocol_error='empty reply')]
        assert payload['budget']['tool_calls'] == 1
        return dict(tool='rollout', args=dict(state='pair-0', edit=submission['edit']))
    agent, client, transport = scripted_agent(session, tmp_path, [raw, rollout, dict(submit=submission)])
    result = agent.run()
    assert agent.protocol_errors == agent.consecutive_empty_replies == 0
    assert result['budget']['tool_calls'] == 2 and client.calls == 3
    rows = [json.loads(line) for line in agent.ledger.path.read_text().splitlines()]
    assert not any(row['event'] == 'protocol_error' for row in rows)
    assert next(row for row in rows if row['event'] == 'teacher_turn')['raw_output'] == raw
    assert next(row for row in rows if row['event'] == 'empty_reply')['consecutive_empty_replies'] == 1
    assert transport.budget.summary()['unsettled_attempts'] == 0


def test_five_consecutive_empty_replies_halt_with_recorded_reason(session, tmp_path, fast_clock):
    agent, client, transport = scripted_agent(session, tmp_path, ['', ' ', '\n', '\t', '\r\n'])
    with pytest.raises(BudgetExceeded, match='5 consecutive empty replies'):
        agent.run()
    assert client.calls == agent.budget.calls == agent.consecutive_empty_replies == 5
    assert agent.protocol_errors == 0 and len(agent.transcript) == 5
    rows = [json.loads(line) for line in agent.ledger.path.read_text().splitlines()]
    assert rows[-1]['event'] == 'halted' and '5 consecutive empty replies' in rows[-1]['reason']
    assert [row['consecutive_empty_replies'] for row in rows if row['event'] == 'empty_reply'] == [1, 2, 3, 4, 5]
    assert not any(row['event'] == 'protocol_error' for row in rows)
    assert transport.budget.summary()['unsettled_attempts'] == 0
    assert not (agent.directory / 'submission.json').exists()


@pytest.mark.parametrize('nonempty,errors', [
    ('prose', 1), (dict(tool='action_scores', args=dict(state='pair-0', candidates=['look'])), 0)])
def test_nonempty_reply_resets_empty_streak(session, tmp_path, fast_clock, nonempty, errors):
    submission = sample_submission(add_direction(session))
    agent, client, _ = scripted_agent(session, tmp_path, [''] * 4 + [nonempty] + ['\n'] * 4 + [
        dict(tool='rollout', args=dict(state='pair-0', edit=submission['edit'])), dict(submit=submission)])
    result = agent.run()
    assert result['budget']['tool_calls'] == 10 and client.calls == 11
    assert agent.protocol_errors == errors and agent.consecutive_empty_replies == 0
    rows = [json.loads(line) for line in agent.ledger.path.read_text().splitlines()]
    assert [row['consecutive_empty_replies'] for row in rows if row['event'] == 'empty_reply'] == [1, 2, 3, 4] * 2


def test_empty_reply_does_not_reset_category_protocol_error_count(session, tmp_path, fast_clock):
    agent, client, _ = scripted_agent(session, tmp_path, ['prose'] * 8 + [''] + ['prose'])
    with pytest.raises(BudgetExceeded, match='ninth error'):
        agent.run()
    assert agent.protocol_errors == 9 and client.calls == agent.budget.calls == 10


def test_prompt_messages_keep_stable_prefix_and_trailing_budget(session, tmp_path, fast_clock):
    submission = sample_submission(add_direction(session))
    script = [' ', 'prose', dict(tool='rollout', args=dict(state='pair-0', edit=submission['edit'])),
              dict(submit=submission)]
    agent, client, _ = scripted_agent(session, tmp_path, script)
    agent.run()
    rows = [json.loads(line) for line in agent.ledger.path.read_text().splitlines()]
    turns = [row for row in rows if row['event'] == 'teacher_turn']
    previous = None
    for request, turn in zip(client.requests, turns):
        messages = request['input']
        assert [message['role'] for message in messages] == ['system', 'user', 'user']
        assert messages[:2] == client.requests[0]['input'][:2]
        payload = turn['request']['payload']
        assert messages[0]['content'] == SYSTEM + '\nTool documentation:\n' + canonical_bytes(TOOL_DOCS).decode()
        assert messages[1]['content'] == canonical_bytes(dict(category=payload['category'], pairs=payload['pairs'])).decode()
        assert 'budget' not in json.loads(messages[1]['content'])
        history, budget = messages[2]['content'].split('\nBUDGET ')
        assert history == canonical_bytes(dict(transcript=payload['transcript'])).decode()
        assert budget == canonical_bytes(payload['budget']).decode()
        assert messages[2]['content'].count('\n') == 1
        if previous is not None:
            # Canonical JSON keeps prior entries byte-identical; only the closing
            # list/object delimiters move as history is appended before the budget.
            assert history.startswith(previous[:-2])
        previous = history
        assert turn['request']['kind'] == 'p0-agent/3'
        assert turn['request']['template_sha256'] == digest([SYSTEM, TOOL_DOCS, PROMPT_LAYOUT])


def test_prompt_envelope_binds_new_layout_and_rejects_old_or_changed_template(tmp_path, monkeypatch):
    teacher = P0Teacher(cache_dir=tmp_path / 'cache', usage_ledger=tmp_path / 'usage.jsonl', run_id='fake',
                        client=ScriptedTeacherClient([]))
    prompt = empty_prompt()
    def visible(envelope):
        return dict(observation=canonical_bytes(envelope).decode(), admissible_commands=[])
    assert teacher.prompt([], {}, visible(prompt.to_json())) == prompt.messages()
    for changes in (dict(kind='p0-agent/1'), dict(template_sha256=digest([SYSTEM, TOOL_DOCS])), dict(extra='unknown')):
        with pytest.raises(ValueError, match='Invalid P0 prompt envelope'):
            teacher.prompt([], {}, visible(dict(prompt.to_json(), **changes)))
    with pytest.raises(ValueError, match='Invalid P0 prompt envelope'):
        teacher.prompt([dict(role='user', content='extra')], {}, visible(prompt.to_json()))
    envelope = prompt.to_json()
    monkeypatch.setattr('repair.p0_agent.PROMPT_LAYOUT', ('changed layout',))
    assert envelope['template_sha256'] != prompt.to_json()['template_sha256']
    with pytest.raises(ValueError, match='Invalid P0 prompt envelope'):
        teacher.prompt([], {}, visible(envelope))
    with pytest.raises(ValueError, match='Invalid P0 prompt payload'):
        AgentPrompt(dict(prompt.payload, audit='must not enter prompts'))


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
    assert parse_turn(transport(empty_prompt()))[0]['tool'] == 'action_scores'
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
        transport(empty_prompt())
    assert client.calls == 1 and client.requests[0]['reasoning'] == {'effort': 'high'}
    assert transport.budget.summary()['stop_reason']
    rows = (tmp_path / 'transport/usage.jsonl').read_text()
    assert 'Raw SDK exception' not in rows


def test_cli_refuses_existing_diagnosis_directory_before_loading(tmp_path, monkeypatch, capsys):
    import tools.p0_pilot as pilot
    monkeypatch.setattr(pilot, 'load_pairs', lambda _: pytest.fail('Existing output must fail before loading'))
    with pytest.raises(SystemExit) as exc:
        pilot.main(['diagnose', '--output', str(tmp_path), '--pairs', 'unused.json',
                    '--category', 'refused_action', '--allow-teacher-api'])
    assert exc.value.code == 2
    assert 'reports/p0_runs/diagnose_v3/<category>' in capsys.readouterr().err
