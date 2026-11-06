"""CPU-only reservation boundaries through the real transport and fake endpoint."""
import json

import pytest

from repair.p0_agent import BudgetExceeded
from repair.p0_pairs import CATEGORIES, teacher_pair
from sed.config import load_config
from sed.storage import canonical_bytes
from surgery.agent import Agent, Budget, SUBMIT_ONLY_NOTICE, TOKEN_SUBMIT_ONLY_NOTICE, make_transport
from tests.test_p1_agent import session, frozen_categories
from tools.p0_pilot import ScriptedTeacherClient


SUBMIT = dict(submit=dict(mechanism='Keep the tested no-op.', edit=[]))


def rows(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def setup_agent(tmp_path, kind='category', *, client=None, validated=True, budget=None):
    categories = CATEGORIES if kind == 'integration' else (('none',) if kind == 'global_b' else ('refused_action',))
    sess = session(tmp_path, categories)
    client = client or ScriptedTeacherClient([SUBMIT])
    transport = make_transport(load_config('configs/base_e4b.yaml'), kind, tmp_path / 'transport',
                               kind=kind, client=client)
    agent = Agent(sess, transport, tmp_path / 'agent', category=categories[0] if kind == 'category' else kind,
                  kind=kind, setting='B' if kind == 'global_b' else 'A', pairs_sha256='pairs', budget=budget,
                  submissions=frozen_categories(sess) if kind == 'integration' else ())
    if validated:
        # Exercise real reset/replay on fake environments, including the ALL-pair
        # integration gate. No hand-installed success labels or fake submissions.
        ids = list(sess.pairs)
        agent.execute_batch([dict(tool='rollout', args=dict(state=i, edit=[])) for i in ids], -1)
        assert agent.submit(SUBMIT['submit']) is None  # Table has been returned; next submit confirms.
    initial = [teacher_pair(p, sess.student.decision_prefix(sess.state(p['pair_id']))) for p in sess.pairs.values()]
    return agent, client, initial


def leave_budget(agent, **remaining):
    caps = agent.transport.budget.caps
    # Unsettled bounds must count, even when no reported usage is available.
    agent.transport.budget.reserve('earlier-attempt',
        input_bound=caps['input_tokens'] - remaining.get('input_tokens', caps['input_tokens']),
        output_cap=caps['output_tokens'] - remaining.get('output_tokens', caps['output_tokens']))


@pytest.mark.parametrize('kind', ['category', 'integration', 'global_b'])
@pytest.mark.parametrize('field', ['input_tokens', 'output_tokens'])
@pytest.mark.parametrize('offset', [-1, 0])
def test_next_turn_plus_submit_boundary_and_freeze(tmp_path, kind, field, offset):
    agent, client, initial = setup_agent(tmp_path, kind)
    normal = agent.transport.reservation_estimate(agent.prompt(initial))
    submit = agent.transport.reservation_estimate(agent.prompt(initial, TOKEN_SUBMIT_ONLY_NOTICE))
    remaining = normal[field] + submit[field] + offset
    leave_budget(agent, **{field: remaining})
    frozen = agent.run()
    assert frozen['submission']['edit'] == []
    assert json.loads((agent.directory / 'submission.json').read_text()) == frozen
    assert client.calls == 1
    events = rows(agent.ledger.path)
    check = next(r for r in events if r['event'] == 'token_budget_preflight')
    assert check['remaining_budget'][field] == remaining
    assert check['reservation_estimate'] == normal
    assert check['submit_only_reservation_estimate'] == submit
    assert check['submit_only_triggered'] == (offset == -1)
    assert agent.submit_only == (offset == -1)
    notices = [r for r in events if r['event'] == 'submit_only']
    assert len(notices) == (offset == -1)
    request = next(r['request'] for r in events if r['event'] == 'teacher_turn')
    assert (dict(notice=TOKEN_SUBMIT_ONLY_NOTICE) in request['payload']['transcript']) == (offset == -1)
    assert not agent.transport.budget.summary()['stop_reason']


@pytest.mark.parametrize('kind', ['category', 'integration', 'global_b'])
@pytest.mark.parametrize('field', ['input_tokens', 'output_tokens'])
@pytest.mark.parametrize('offset', [-1, 0])
def test_submit_only_exact_affordability(tmp_path, kind, field, offset):
    agent, client, initial = setup_agent(tmp_path, kind)
    submit = agent.transport.reservation_estimate(agent.prompt(initial, TOKEN_SUBMIT_ONLY_NOTICE))
    leave_budget(agent, **{field: submit[field] + offset})
    if offset == 0:
        assert agent.run()['submission']['edit'] == []
        assert client.calls == 1
    else:
        frozen = agent.run()
        assert frozen['verification']['complete']
        assert frozen['confirmation_status'] == 'unconfirmed_at_budget_exhaustion'
        assert client.calls == 0
        assert (agent.directory / 'submission.json').exists()
        events = rows(agent.ledger.path)
        assert events[-1]['event'] == 'frozen'
        assert field in events[-2]['limiting_budgets']
        assert events[-2]['stop_reason'] == 'insufficient_token_budget_for_submit_only'
        assert not (agent.transport.directory / 'requests.jsonl').exists()
    assert agent.submit_only
    assert not agent.transport.budget.summary()['stop_reason']


class FullUsageClient(ScriptedTeacherClient):
    def __init__(self, script, *, truncate_first=False):
        super().__init__(script)
        self.truncate_first = truncate_first

    def create(self, **kwargs):
        response = super().create(**kwargs)
        value = response.parse()
        value.usage['input_tokens'] = len(canonical_bytes(kwargs['input'])) + 16 * (len(kwargs['input']) + 1)
        value.usage['output_tokens'] = kwargs['max_output_tokens']
        if self.calls == 1 and self.truncate_first:
            value.status = 'incomplete'
            value.incomplete_details = dict(reason='max_output_tokens')
        return response


def test_estimate_matches_transport_reservations_including_truncation_retry(tmp_path):
    client = FullUsageClient(['{"submit":', SUBMIT], truncate_first=True)
    agent, _, initial = setup_agent(tmp_path, client=client)
    agent.transcript.append(dict(notice='UTF-8 byte bound: café ☃'))
    estimate = agent.transport.reservation_estimate(agent.prompt(initial, TOKEN_SUBMIT_ONLY_NOTICE))
    assert estimate['output_caps'] == [16384, 32768]
    leave_budget(agent, **{k: estimate[k] for k in ('input_tokens', 'output_tokens')})
    assert agent.run()['submission']['edit'] == []
    assert client.calls == 2
    reservations = [r['amounts'] for r in rows(agent.transport.budget.path)
                    if r['event'] == 'reservation' and r['request_id'] != 'earlier-attempt']
    assert reservations == [dict(input_tokens=estimate['input_bound'], output_tokens=cap)
                            for cap in estimate['output_caps']]
    assert all(agent.transport.budget.summary()['tokens_or_unsettled_bounds'][k] == cap
               for k, cap in agent.transport.budget.caps.items())


def test_guard_rechecks_after_normal_turn_and_usage_settlement(tmp_path):
    client = FullUsageClient([dict(calls=[dict(tool='rollout', args=dict(state='pair-0', edit=[]))]), SUBMIT, SUBMIT])
    agent, _, initial = setup_agent(tmp_path, client=client, validated=False)
    normal = agent.transport.reservation_estimate(agent.prompt(initial))
    submit = agent.transport.reservation_estimate(agent.prompt(initial, TOKEN_SUBMIT_ONLY_NOTICE))
    leave_budget(agent, output_tokens=normal['output_tokens'] + submit['output_tokens'])
    assert agent.run()['submission']['edit'] == []
    checks = [r for r in rows(agent.ledger.path) if r['event'] == 'token_budget_preflight']
    assert [r['submit_only_triggered'] for r in checks] == [False, True, False]
    assert checks[1]['remaining_budget']['output_tokens'] == 98304 - 16384
    assert agent.budget.rollouts == 6


def test_submit_only_rejects_tools_and_rechecks_feedback_cost(tmp_path):
    client = ScriptedTeacherClient([dict(calls=[dict(tool='rollout', args=dict(state='pair-0', edit=[]))]), SUBMIT, SUBMIT])
    agent, _, initial = setup_agent(tmp_path, client=client)
    estimate = agent.transport.reservation_estimate(agent.prompt(initial, TOKEN_SUBMIT_ONLY_NOTICE))
    leave_budget(agent, output_tokens=estimate['output_tokens'] + 100)
    calls = agent.budget.calls
    assert agent.run()['submission']['edit'] == []
    assert agent.budget.calls == calls
    assert agent.protocol_errors == 1
    checks = [r for r in rows(agent.ledger.path) if r['event'] == 'token_budget_preflight']
    assert len(checks) == 3 and all(r['submit_only'] for r in checks)
    assert checks[1]['reservation_estimate']['input_bound'] > checks[0]['selected_reservation_estimate']['input_bound']


def test_submit_only_keeps_three_reply_limit(tmp_path):
    calls = dict(calls=[dict(tool='rollout', args=dict(state='pair-0', edit=[]))])
    agent, client, initial = setup_agent(tmp_path, client=ScriptedTeacherClient([calls] * 3), validated=False)
    estimate = agent.transport.reservation_estimate(agent.prompt(initial, TOKEN_SUBMIT_ONLY_NOTICE))
    leave_budget(agent, output_tokens=estimate['output_tokens'] + 1000)
    with pytest.raises(BudgetExceeded, match='submit-only reply limit'):
        agent.run()
    assert client.calls == 3
    assert not (agent.directory / 'submission.json').exists()
    assert agent.budget.calls == 0


def test_tool_exhaustion_notice_and_token_guard(tmp_path):
    agent, client, initial = setup_agent(tmp_path, budget=Budget(tool_calls=1))
    assert agent.submit_only
    assert agent.run()['submission']['edit'] == []
    events = rows(agent.ledger.path)
    notice = next(r for r in events if r['event'] == 'submit_only')
    assert notice['result'] == dict(notice=SUBMIT_ONLY_NOTICE)
    check = next(r for r in events if r['event'] == 'token_budget_preflight')
    assert check['submit_only'] and not check['submit_only_triggered']
    assert check['required_with_submit_only']['input_tokens'] == check['reservation_estimate']['input_tokens']


def test_feedback_cannot_consume_unaffordable_submit_retry(tmp_path):
    agent, client, initial = setup_agent(tmp_path, client=ScriptedTeacherClient([''] * 3), validated=False)
    estimate = agent.transport.reservation_estimate(agent.prompt(initial, TOKEN_SUBMIT_ONLY_NOTICE))
    leave_budget(agent, output_tokens=estimate['output_tokens'])
    with pytest.raises(BudgetExceeded, match='Insufficient token budget for submit-only'):
        agent.run()
    assert client.calls == 1
    assert not (agent.directory / 'submission.json').exists()


def test_existing_shared_halt_is_not_bypassed(tmp_path):
    agent, client, _ = setup_agent(tmp_path)
    agent.transport.budget.halt('fixture_unknown_outcome')
    with pytest.raises(BudgetExceeded, match='fixture_unknown_outcome'):
        agent.run()
    assert client.calls == 0
    assert not (agent.directory / 'submission.json').exists()
