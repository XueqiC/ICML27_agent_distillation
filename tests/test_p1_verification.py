"""Prereg revision 2: real reset/replay through fake environments, CPU only."""
from copy import deepcopy
import json

import pytest

from repair.dry_fakes import FakeEnvironment
from repair.p0_agent import BudgetExceeded
from repair.p0_pairs import CATEGORIES
from sed.config import load_config
from sed.storage import digest, file_hash
from surgery.agent import Agent, Budget, CATEGORY_LIMITS, LIMITS, make_transport
from surgery.common import seal
from surgery.session import Session
from tests.p0_fakes import fake_pair, fake_greedy
from tests.test_p1_agent import frozen_categories
from tests.test_p1_student import student, scale, hashes
from tests.test_p1_token_budget import FullUsageClient, leave_budget
from tools.p0_pilot import ScriptedTeacherClient


COUNTS = dict(zip(CATEGORIES, [13, 18, 9, 6, 9]))


def proposal(edits=None):
    return dict(mechanism='Synthetic fixture; execution determines success.', edit=[scale()] if edits is None else edits)


def make_agent(tmp_path, *, kind='category', count=6, script=(), budget=None, categories=None):
    s = student()
    s.greedy = fake_greedy
    groups = categories or {'missing_precondition' if kind == 'category' else 'none': count}
    pairs = []
    for category, size in groups.items():
        pairs.extend([fake_pair(len(pairs) + i, category=category) for i in range(size)])
    sess = Session(s, pairs, tmp_path / 'tools', env_factory=lambda p: FakeEnvironment())
    submissions = []
    if kind == 'integration':
        # Supply one deployable category edit, preserving the five-input contract.
        submissions = frozen_categories(sess)
        submissions[0] = seal(dict(submissions[0], **sess.bundle(proposal())))
    client = ScriptedTeacherClient(script)
    transport = make_transport(load_config('configs/base_e4b.yaml'), kind, tmp_path / 'transport',
                               kind=kind, client=client)
    agent = Agent(sess, transport, tmp_path / 'agent', category=next(iter(groups)) if kind == 'category' else kind,
                  kind=kind, setting='B' if kind == 'global_b' else 'A', pairs_sha256='pairs',
                  budget=budget, submissions=submissions)
    return agent, client


def events(agent, name):
    return [r for line in agent.ledger.path.read_text().splitlines()
            if (r := json.loads(line))['event'] == name]


@pytest.mark.parametrize('kind,count', [('category', 18), ('global_b', 90), ('integration', 55)])
def test_submit_verifies_every_pair_then_teacher_confirms(tmp_path, kind, count):
    value = proposal()

    def confirm(payload):
        assert not (tmp_path / 'agent/submission.json').exists()
        table = payload['transcript'][-1]['result']['data']
        assert table['pair_count'] == count and table['complete']
        assert table['edit_sha256'] == digest(value['edit'])
        assert len(table['rows']) == count
        assert table['totals'] == {label: dict(success=count, failure=0, unknown=0, verified=count)
                                   for label in ('unedited', 'edited')}
        assert 'internal_state' not in json.dumps(payload)
        return dict(submit=value)

    agent, client = make_agent(tmp_path, kind=kind, count=count,
        categories=COUNTS if kind == 'integration' else None,
        script=[dict(submit=value), confirm])
    before = hashes(agent.session.student)
    frozen = agent.run()
    assert client.calls == 2 and agent.protocol_errors == 0
    assert frozen['confirmation_status'] == 'confirmed'
    assert not frozen['unverified_at_budget_exhaustion']
    assert frozen['budget']['rollouts'] == 2 * count
    assert frozen['budget']['environment_steps'] == agent.session.environment_steps == 8 * count
    assert frozen['budget']['tool_calls'] == 0
    assert frozen['budget']['candidate_edits'] == 1
    assert len(events(agent, 'verification_rollout')) == 2 * count
    assert hashes(agent.session.student) == before
    for row in frozen['verification']['rows']:
        for label in ('unedited', 'edited'):
            ref = row[label + '_receipt']
            receipt = frozen['verification_receipts'][ref['cache_key']]
            assert ref['sha256'] == digest(receipt['data'])
            data = receipt['data']
            assert data['verification'] == 'real_environment' and data['framing'] == 'free_greedy'
            assert data['prefix_turns'] == 1 and data['continuation_turns'] == 3
            assert not data['counterfactual_first_decision']
            execution = next(p for p in (agent.session.directory / 'executions').glob('*.json')
                             if file_hash(p) == ref['execution_sha256'])
            assert json.loads(execution.read_text())['max_steps'] == 40


def test_two_of_eighteen_checks_only_missing_and_keeps_receipts(tmp_path):
    agent, _ = make_agent(tmp_path, count=18)
    value = proposal()
    agent.execute_batch([dict(tool='rollout', args=dict(state=f'pair-{i}', edit=value['edit']))
                         for i in range(2)], 0)
    prior = deepcopy(agent.session.rollout_results[digest(value['edit'])])
    assert agent.submit(value) is None
    check = events(agent, 'verification_preflight')[-1]
    assert check['required'] == dict(rollouts=34, environment_steps=1360)
    frozen = agent.submit(value)
    assert frozen['budget']['rollouts'] == 36 and frozen['budget']['tool_calls'] == 2
    for receipt in prior.values():
        assert frozen['verification_receipts'][receipt['cache_key']] == receipt


def test_revision_rechecks_exact_edit_and_reuses_baselines(tmp_path):
    first = proposal()
    revised = proposal([dict(scale(), c=0.25)])
    agent, client = make_agent(tmp_path, script=[dict(submit=first), dict(submit=revised), dict(submit=revised)])
    frozen = agent.run()
    assert client.calls == 3
    assert frozen['verification']['edit_sha256'] == digest(revised['edit'])
    assert frozen['budget']['rollouts'] == 18  # 6 baseline + 6 of each edited list.
    assert frozen['budget']['candidate_edits'] == 2
    tables = events(agent, 'submission_verification')
    assert len(tables) == 2
    assert [r['required']['rollouts'] for r in events(agent, 'verification_preflight')] == [12, 6]
    assert [r['unedited_receipt'] for r in tables[0]['result']['data']['rows']] == [
        r['unedited_receipt'] for r in frozen['verification']['rows']]


def test_integration_final_order_requires_new_all_pair_rollouts(tmp_path):
    first = proposal([scale(), dict(scale(), c=0.25)])
    final = proposal(list(reversed(first['edit'])))
    agent, _ = make_agent(tmp_path, kind='integration', categories={c: 6 for c in CATEGORIES},
        script=[dict(submit=first), dict(submit=final), dict(submit=final)])
    frozen = agent.run()
    assert frozen['verification']['edit_sha256'] == digest(final['edit'])
    assert frozen['budget']['rollouts'] == 90  # 30 baseline + both exact ordered lists.
    assert [r['required']['rollouts'] for r in events(agent, 'verification_preflight')] == [60, 30]


@pytest.mark.parametrize('success', [False, 'unknown'])
def test_negative_and_unknown_real_outcomes_remain_in_table(tmp_path, monkeypatch, success):
    state = FakeEnvironment.state

    def outcome(self, text):
        visible, reward, terminated, truncated, audit = state(self, text)
        if self.held:  # After the checked prefix; stop at the first new action.
            return visible, reward, True, truncated, dict(audit, won=success)
        return visible, reward, terminated, truncated, audit

    # Both the recorded prefix and verification use this same fake environment.
    monkeypatch.setattr(FakeEnvironment, 'state', outcome)
    agent, _ = make_agent(tmp_path)
    assert agent.submit(proposal()) is None
    frozen = agent.submit(proposal())
    table = frozen['verification']
    assert table['complete'] and table['totals']['edited']['verified'] == 6
    assert table['totals']['edited']['failure' if success is False else 'unknown'] == 6
    assert all(r['edited_success'] == success and r['unedited_success'] == success for r in table['rows'])


def test_tool_after_table_requires_new_confirmation_without_redundant_rollouts(tmp_path):
    value = proposal()
    agent, _ = make_agent(tmp_path)
    assert agent.submit(value) is None
    agent.execute_batch([dict(tool='rollout', args=dict(state='pair-0', edit=value['edit']))], 0)
    assert agent.transcript[-1]['result']['cache_hit']
    assert agent.budget.rollouts == 13 and agent.budget.steps == 48
    assert agent.submit(value) is None
    assert not (agent.directory / 'submission.json').exists()
    assert agent.submit(value)['budget']['rollouts'] == 13


@pytest.mark.parametrize('field,cap', [('rollouts', 11), ('environment_steps', 479)])
@pytest.mark.parametrize('submit_only', [False, True])
def test_insufficient_verification_budget_preserves_submission(tmp_path, field, cap, submit_only):
    budget = Budget(**{field: cap}, **({'tool_calls': 0} if submit_only else {}))
    agent, client = make_agent(tmp_path, budget=budget, script=[dict(submit=proposal())])
    frozen = agent.run()
    assert client.calls == 1 and frozen['submission']['edit'] == proposal()['edit']
    assert frozen['unverified_at_budget_exhaustion']
    table = frozen['verification']
    assert table['status'] == 'unverified_at_budget_exhaustion' and not table['complete']
    assert table['limiting_budgets'] == [field]
    assert len(table['rows']) == 6 and not frozen['verification_receipts']
    assert frozen['budget']['rollouts'] == frozen['budget']['environment_steps'] == 0
    assert events(agent, 'frozen')[0]['unverified_at_budget_exhaustion']


def test_exhaustion_preserves_partial_and_unknown_results(tmp_path):
    agent, _ = make_agent(tmp_path, budget=Budget(rollouts=2))
    value = proposal()
    agent.execute_batch([dict(tool='rollout', args=dict(state='pair-0', edit=value['edit']))], 0)
    frozen = agent.submit(value)
    assert frozen['unverified_at_budget_exhaustion']
    assert len(frozen['verification_receipts']) == 1
    rows = {r['pair_id']: r for r in frozen['verification']['rows']}
    assert rows['pair-0']['edited_success'] is True
    assert rows['pair-1']['edited_success'] == 'unknown'
    assert rows['pair-1']['edited_receipt'] is None


def test_exact_reservation_boundary_allows_verification(tmp_path):
    agent, _ = make_agent(tmp_path, budget=Budget(rollouts=12, environment_steps=480, tool_calls=0))
    assert agent.submit(proposal()) is None
    frozen = agent.submit(proposal())
    assert frozen['verification']['complete'] and not frozen['unverified_at_budget_exhaustion']
    assert frozen['budget']['environment_steps'] == 48


def test_local_delta_verification_uses_deployed_list_sha(tmp_path):
    agent, _ = make_agent(tmp_path)
    agent.execute_batch([dict(tool='local_finetune', args=dict(pair_ids=['pair-0'],
        modules=['0:mlp.down_proj'], rank=1, lr=1e-3, steps=1))], 0)
    delta_id = agent.transcript[-1]['result']['data']['delta_id']
    value = proposal([dict(type='local_finetune', delta_id=delta_id, alpha=1)])
    assert agent.submit(value) is None
    frozen = agent.submit(value)
    assert frozen['verification']['edit_sha256'] == digest(frozen['submission']['edit'])
    assert frozen['verification']['edit_sha256'] != digest(value['edit'])
    assert frozen['budget']['candidate_edits'] == 1  # Training + verification is one candidate.


def test_scores_and_counterfactual_rollouts_do_not_satisfy_gate(tmp_path):
    agent, _ = make_agent(tmp_path)
    value = proposal()
    agent.execute_batch([dict(tool='apply_edit', args=dict(state='pair-0', edit=value['edit'], candidates=['look']))], 0)
    cf = agent.session.call('counterfactual_context', dict(state='pair-0',
        edit=dict(op='delete', span='current_thought')))['data']['state_id']
    agent.execute_batch([dict(tool='rollout', args=dict(state=cf, edit=value['edit']))], 1)
    assert not agent.session.rollout_results
    assert agent.submit(value) is None
    assert events(agent, 'verification_preflight')[-1]['required']['rollouts'] == 12


def test_failed_reset_replay_is_not_verification_and_settles_steps(tmp_path):
    agent, _ = make_agent(tmp_path)

    class WrongPrefix(FakeEnvironment):
        def step(self, action):
            visible, *rest = super().step(action)
            return dict(visible, observation='Replay mismatch'), *rest

    agent.session.env_factory = lambda p: WrongPrefix()
    with pytest.raises(RuntimeError, match='replay/rollout failed'):
        agent.submit(proposal())
    assert agent.budget.rollouts == 1 and agent.budget.steps == agent.session.environment_steps == 1
    assert not agent.session.rollout_results and not (agent.directory / 'submission.json').exists()
    assert events(agent, 'verification_rollout_budget')[-1]['budget']['environment_steps'] == 1


@pytest.mark.parametrize('category,count', COUNTS.items())
def test_category_limits_bound_to_actual_session(tmp_path, category, count):
    agent, _ = make_agent(tmp_path, categories={category: count}, budget=Budget())
    assert tuple(agent.budget.limits.values()) == CATEGORY_LIMITS[category]
    assert agent.budget.limits['rollouts'] == 50 + 4 * count
    assert agent.budget.limits['environment_steps'] == 40 * (50 + 4 * count)


def test_reduced_budget_cannot_exceed_actual_category_limit(tmp_path):
    with pytest.raises(ValueError, match='only reduce session limits'):
        make_agent(tmp_path, budget=Budget(rollouts=75))


@pytest.mark.parametrize('count', [6, 90])
@pytest.mark.parametrize('supplied_budget', [False, True])
def test_global_b_limits_bound_to_actual_session(tmp_path, count, supplied_budget):
    agent, _ = make_agent(tmp_path, kind='global_b', count=count,
                          budget=Budget('global_b') if supplied_budget else None)
    assert tuple(agent.budget.limits.values()) == (500, 80 + 4 * count, 40, 40 * (80 + 4 * count))
    assert LIMITS['global_b'] == (500, 440, 40, 17600)


@pytest.mark.parametrize('field,cap', [('rollouts', 104), ('environment_steps', 4160)])
@pytest.mark.parametrize('offset', [-1, 0, 1])
def test_global_b_trial_budget_respects_actual_session_limit(tmp_path, field, cap, offset):
    budget = Budget('global_b', **{field: cap + offset})
    if offset > 0:
        with pytest.raises(ValueError, match='only reduce session limits'):
            make_agent(tmp_path, kind='global_b', count=6, budget=budget)
    else:
        agent, _ = make_agent(tmp_path, kind='global_b', count=6, budget=budget)
        assert agent.budget.limits[field] == cap + offset


@pytest.mark.parametrize('kind,count,exploration', [
    ('category', 18, 50), ('integration', 55, 30), ('global_b', 90, 80)])
def test_full_exploration_plus_two_verifications_fit(kind, count, exploration):
    budget = Budget(kind)
    budget.bind_pairs(count)
    for _ in range(exploration):
        budget.reserve('rollout', dict(edit=[]))
        budget.steps += 40  # Settle the maximum H=40, including prefix replay.
    for _ in range(4 * count):
        budget.reserve_verification_rollout()
        budget.steps += 40
    assert budget.calls == exploration
    assert budget.rollouts == budget.limits['rollouts'] == exploration + 4 * count
    assert budget.steps == budget.limits['environment_steps'] == 40 * budget.rollouts
    with pytest.raises(BudgetExceeded, match='Rollout budget exhausted'):
        budget.reserve_verification_rollout()
    assert LIMITS['integration'] == (100, 250, 10, 10000)


def test_token_submit_only_still_freezes_unverified_when_rollouts_unaffordable(tmp_path):
    from surgery.agent import TOKEN_SUBMIT_ONLY_NOTICE
    from repair.p0_pairs import teacher_pair
    agent, client = make_agent(tmp_path, budget=Budget(rollouts=0), script=[dict(submit=proposal())])
    initial = [teacher_pair(p, agent.session.student.decision_prefix(agent.session.state(p['pair_id'])))
               for p in agent.session.pairs.values()]
    estimate = agent.transport.reservation_estimate(agent.prompt(initial, TOKEN_SUBMIT_ONLY_NOTICE))
    leave_budget(agent, output_tokens=estimate['output_tokens'])
    frozen = agent.run()
    assert agent.token_submit_only and client.calls == 1
    assert frozen['unverified_at_budget_exhaustion']


def test_verified_pending_submission_survives_unaffordable_confirmation(tmp_path):
    from surgery.agent import TOKEN_SUBMIT_ONLY_NOTICE
    from repair.p0_pairs import teacher_pair
    agent, _ = make_agent(tmp_path)
    # Both output attempts consume the entire last affordable exchange.
    client = FullUsageClient(['{"submit":', dict(submit=proposal())], truncate_first=True)
    agent.transport = make_transport(load_config('configs/base_e4b.yaml'), 'category', tmp_path / 'last', client=client)
    initial = [teacher_pair(p, agent.session.student.decision_prefix(agent.session.state(p['pair_id'])))
               for p in agent.session.pairs.values()]
    estimate = agent.transport.reservation_estimate(agent.prompt(initial, TOKEN_SUBMIT_ONLY_NOTICE))
    leave_budget(agent, output_tokens=estimate['output_tokens'])
    frozen = agent.run()
    assert client.calls == 2
    assert frozen['verification']['complete']
    assert frozen['confirmation_status'] == 'unconfirmed_at_budget_exhaustion'
    assert not frozen['unverified_at_budget_exhaustion']
