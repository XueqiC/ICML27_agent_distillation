"""Joint protocol exercised entirely through fake clients and environments."""
from copy import deepcopy
from functools import partial
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from repair.facts import FactTable
from repair.joint import (JointPolicy, JointPrompt, JointTeacher, FakeStudent, Reviewer, OUTPUT_CAP,
    ReviewInvalid, contradiction, execute_joint, fact_index, injected_context, make_reviewer,
    review_prompt, validate_review)
from repair.takeover import SharedTeacherBudget
from sed.config import load_config
from sed.react import build_messages, action_outcome
from sed.storage import canonical_bytes, digest, write_json
from students.sampled_student import SampledStudent
from teachers.openai_teacher import TeacherCallError
from tools.joint_guidance import main, summarize_records
from tools.teacher_takeover import main as takeover_main
from test_cg_probe import COMMANDS, INITIAL, GROUPS, ScriptEnvironment, original, coverage_fixture
from test_teacher import response
from test_takeover import HTTPFailure


@pytest.fixture
def budget(tmp_path):
    return SharedTeacherBudget(tmp_path / 'global.jsonl', input_tokens=10_000_000, output_tokens=1_000_000)


def review(decision='guide', action='go to cabinet 2', **changes):
    value = dict(decision=decision, facts_cited=[dict(key='goal', evidence_steps=[-1])],
                 need='More information is needed.', action=action,
                 check='Check the public contents.', guidance='Try ' + action + ' and check the public contents for your goal.')
    if decision == 'accept':
        value.update(facts_cited=[], action='', need='', check='', guidance='')
    return {**value, **changes}


class ScriptReviewer:
    failure = None
    def __init__(self, values):
        self.values, self.prompts = list(values), []
    def request(self, prompt, facts, commands, attempts):
        self.prompts.append(prompt)
        attempts.append(dict(request_id='r-' + str(len(self.prompts)), status='valid'))
        return self.values.pop(0)


class CaptureStudent(FakeStudent):
    def __init__(self, replies):
        super().__init__(replies=replies)
        self.prompts = []
    def generate(self, history, task, visible):
        self.prompts.append(build_messages(history, task, visible))
        return super().generate(history, task, visible)


def context(text=INITIAL, commands=COMMANDS):
    return dict(observation=text, admissible_commands=commands)


def test_U_is_byte_identical_and_never_constructs_reviewer(original, tmp_path, budget):
    student = CaptureStudent(['Action: look'])
    p = JointPolicy(student, original.task_id, 'U', reviewer=SimpleNamespace(request=lambda *_: pytest.fail('R called')))
    h = [{'visible': context(), 'reply': 'Thought: My plan.\nAction: look'}]
    v = context('Current')
    before = deepcopy((h, v))
    p(h, v)
    assert canonical_bytes(student.prompts[0]) == canonical_bytes(build_messages(h, {'task_id': original.task_id}, v))
    assert (h, v) == before
    assert p.steps[0]['review_validation'] == 'not_called'
    row = execute_joint(case(original), original, mode='U', decoding='greedy', seed=None,
        env_factory=ScriptEnvironment, student_factory=lambda _: CaptureStudent(['Action: take apple 1 from cabinet 2']),
        reviewer_factory=lambda _: pytest.fail('reviewer factory called in U'),
        output=tmp_path, run_sha256='unit', budget=budget)
    assert row['success'] and row['metrics']['reviews'] == 0


def test_injection_isolation_persistence_and_G1_switch():
    student = CaptureStudent(['Action: look', 'Action: open cabinet 2', 'Action: look'])
    r = ScriptReviewer([review()])
    p = JointPolicy(student, 'private_task', 'G1', r)
    v = context()
    before = deepcopy(v)
    executed = p([], v)
    assert p.switched and executed == 'Action: open cabinet 2'  # Alternative valid route is retained.
    assert student.prompts[0] == build_messages([], {'task_id': 'private_task'}, v)
    expected = deepcopy(v)
    expected['observation'] += '\n\nGuidance: ' + review()['guidance']
    assert student.prompts[1] == build_messages([], {'task_id': 'private_task'}, expected)
    h = [{'visible': v, 'reply': executed}]
    current = context('You see an apple 1.')
    p(h, current)
    assert len(r.prompts) == 1 and v == before
    assert student.prompts[2][-1] == build_messages(h, {}, current)[-1]
    assert json.loads(student.prompts[2][1]['content'])['observation'] == expected['observation']
    assert '\n\nGuidance:' not in h[0]['visible']['observation']


def test_G1_accepts_until_first_intervention_then_stops():
    r = ScriptReviewer([review('accept'), review('demonstrate')])
    p = JointPolicy(CaptureStudent(['Action: look']), 't', 'G1', r)
    v, h = context(), []
    for i in range(4):
        reply = p(h, v)
        h.append({'visible': v, 'reply': reply})
    assert len(r.prompts) == 2
    assert [s['executor'] for s in p.steps] == ['student', 'teacher', 'student', 'student']
    assert p.steps[1]['executed_reply'] == 'Thought: Check the public contents.\nAction: go to cabinet 2'


@pytest.mark.parametrize('reply,reason', [('invalid', 'not_admissible'), ('Action: fly', 'not_admissible'),
                                          ('Action: take apple 1 from cabinet 2', 'repeat_refused')])
def test_regeneration_falls_back_on_illegal_or_refused_repeat(reply, reason):
    v = context()
    h = [{'visible': v, 'reply': 'Action: take apple 1 from cabinet 2'}]
    current = context('Nothing happens.')
    p = JointPolicy(CaptureStudent(['Action: look', reply]), 't', 'G', ScriptReviewer([review()]))
    assert p(h, current).endswith('Action: go to cabinet 2')
    assert p.steps[0]['fallback_reason'] == reason and p.steps[0]['executor'] == 'teacher'
    assert p.steps[0]['regenerated_reply'] == reply


def test_known_unsatisfied_guard_checks_only_cited_fact():
    h = [{'visible': context(), 'reply': 'Action: inventory'}]
    v = context('You are carrying: a creditcard 1.')
    citation = [dict(key='conditions.hands_free', evidence_steps=[0])]
    r = ScriptReviewer([review(facts_cited=citation)])
    p = JointPolicy(CaptureStudent(['Action: look', 'Action: take apple 1 from cabinet 2']), 't', 'G', r)
    p(h, v)
    assert p.steps[0]['fallback_reason'] == 'conditions.hands_free'
    facts = FactTable.from_record(context(), [dict(action_parsed='inventory', action_outcome='legal', observation=v['observation'])])
    assert contradiction('go to cabinet 2', facts, review(facts_cited=citation)) is None
    assert contradiction('take apple 1 from cabinet 2', facts, review()) is None


def test_review_payload_allowlist_and_hidden_thinking_strip():
    v = dict(context(), audit='SECRET', policy_commands=['SECRET'], extra={'expert_plan': 'SECRET'})
    h = [{'visible': v, 'reply': '<|channel>thought\nSECRET<channel|>Thought: My plan.\nAction: look', 'hidden': 'SECRET'}]
    facts = FactTable.from_record(v, [])
    prompt = review_prompt(h, v, '<|channel>thought\nSECRET<channel|>Thought: Next.\nAction: look', facts, 40)
    text = json.dumps(prompt.messages())
    assert 'SECRET' not in text and 'My plan.' in text and 'Next.' in text
    assert 'admissible_commands' in prompt.payload and prompt.payload['facts'] == facts.prompt_text()
    assert prompt.payload['history'][0]['observation'] == v['observation']


@pytest.mark.parametrize('changes,error', [
    ({'decision': 'other'}, 'decision'), ({'action': 'fly'}, 'admissible'),
    ({'facts_cited': [{'key': 'hidden', 'evidence_steps': [-1]}]}, 'unknown_fact'),
    ({'facts_cited': [{'key': 'goal', 'evidence_steps': [9]}]}, 'unsupported_evidence'),
    ({'facts_cited': []}, 'facts_required'), ({'guidance': 'x ' * 61}, 'word_limit'),
    ({'check': 'x ' * 31}, 'check_word_limit'), ({'facts_cited': ['goal']}, 'citation_schema')])
def test_schema_validation(changes, error):
    with pytest.raises(ReviewInvalid, match=error):
        validate_review(json.dumps(review(**changes)), FactTable(INITIAL), COMMANDS)


class RawClient:
    def __init__(self, responses, budget=None):
        self.results, self.requests, self.budget = list(responses), [], budget
        self.responses = SimpleNamespace(with_raw_response=self)
    def create(self, **kwargs):
        if self.budget:
            assert self.budget.summary()['reserved_attempts'] > len(self.requests)
        self.requests.append(kwargs)
        result = self.results.pop(0)
        if isinstance(result, Exception):
            raise result
        return SimpleNamespace(status_code=200, headers={}, parse=lambda: result)


def reviewer(tmp_path, budget, client):
    task = dict(task_id='train/private/trial', split='train', sealed=False)
    r = make_reviewer(load_config('configs/base_e4b.yaml'), task, tmp_path / 'review', tmp_path, budget, client=client)
    r.teacher.teacher.sleep = lambda _: None
    r.teacher.teacher.jitter = lambda *_: 0
    return r


def test_format_retry_and_second_invalid_accept(tmp_path, budget):
    client = RawClient([response('{bad'), response(json.dumps(review()))], budget)
    r = reviewer(tmp_path, budget, client)
    p = JointPolicy(CaptureStudent(['Action: look']), 't', 'G', r)
    p([], context())
    row = p.steps[0]
    assert [a['status'] for a in row['review_attempts']] == ['invalid', 'valid']
    assert len(client.requests) == 2
    assert 'invalid_json' in client.requests[1]['input'][0]['content']
    assert all(c['max_output_tokens'] == 4096 for c in client.requests)
    assert budget.summary()['reserved_attempts'] == 2
    client = RawClient([response('{}'), response('[]')], budget)
    r = reviewer(tmp_path / 'second', budget, client)
    p = JointPolicy(CaptureStudent(['Action: look']), 't', 'G1', r)
    assert p([], context()) == 'Action: look'
    assert p.steps[0]['review_validation'] == 'review_invalid' and not p.switched
    assert budget.summary()['reserved_attempts'] == 4


class Limit429(Exception):
    status_code = 429
    body = {'error': {'code': 'rate_limit_exceeded'}}


class Quota429(Exception):
    status_code = 429
    body = {'error': {'code': 'insufficient_quota', 'message': 'SECRET'}}


def test_transport_retry_reservation_order_and_quota_halt(tmp_path, budget):
    client = RawClient([Limit429('SECRET'), response(json.dumps(review('accept')))], budget)
    r = reviewer(tmp_path, budget, client)
    JointPolicy(CaptureStudent(['Action: look']), 't', 'G', r)([], context())
    assert len(client.requests) == 2 and budget.summary()['unsettled_attempts'] == 1
    assert budget.summary()['tokens_or_unsettled_bounds']['output_tokens'] == OUTPUT_CAP + 17
    quota = RawClient([Quota429('SECRET')], budget)
    r = reviewer(tmp_path / 'quota', budget, quota)
    with pytest.raises(TeacherCallError):
        JointPolicy(CaptureStudent(['Action: look']), 't', 'G', r)([], context())
    assert len(quota.requests) == 1 and budget.summary()['stop_reason'] == 'teacher_quota'
    assert 'SECRET' not in (tmp_path / 'quota/review/usage.jsonl').read_text()


@pytest.mark.parametrize('failure,count,recovers,kind', [
    (HTTPFailure(503), 1, True, 'TRANSIENT'),
    (HTTPFailure(503), 7, False, 'TRANSIENT'),
    (Quota429('SECRET'), 1, False, 'QUOTA'),
    (HTTPFailure(400), 1, False, 'OTHER'),
    (TimeoutError('SECRET'), 1, True, 'TRANSIENT'),
    (ConnectionError('SECRET'), 1, True, 'TRANSIENT'),
])
def test_reviewer_outer_retries_and_fresh_reservations(tmp_path, budget, failure, count, recovers, kind):
    client = RawClient([failure] * count + [response(json.dumps(review('accept')))], budget)
    r = reviewer(tmp_path, budget, client)
    waits = []
    r.teacher.teacher.sleep = waits.append
    p = JointPolicy(CaptureStudent(['Action: look']), 't', 'G', r)
    if recovers:
        assert p([], context()) == 'Action: look'
        assert p.steps[0]['review_validation'] == 'valid'
        assert r.failure is None and budget.summary()['stop_reason'] is None
        r.teacher.check_health()  # The retained failed receipt cannot re-latch the cap.
    else:
        with pytest.raises(TeacherCallError):
            p([], context())
        assert p.steps[0]['executed_reply'] is None
        assert budget.summary()['stop_reason'] == ('teacher_quota' if kind == 'QUOTA' else 'teacher_halted')
    calls = count + int(recovers)
    assert len(client.requests) == calls and all(c == client.requests[0] for c in client.requests)
    attempts = p.steps[0]['review_attempts']
    assert [a['attempt_index'] for a in attempts] == list(range(calls))
    assert len({a['request_id'] for a in attempts}) == calls
    assert len({a['logical_request_id'] for a in attempts}) == 1
    assert len({a['request_sha256'] for a in attempts}) == 1
    assert all(a['failure_class'] == kind for a in attempts[:count])
    reservations = [json.loads(line) for line in budget.path.read_text().splitlines()
                    if json.loads(line)['event'] == 'reservation']
    assert len(reservations) == calls == len({a['request_id'] for a in reservations})
    assert {a['transport_request_id'] for a in attempts} == {a['request_id'] for a in reservations}
    rows = [json.loads(line) for line in (r.directory / 'usage.jsonl').read_text().splitlines()]
    failures = [row for row in rows if row['status'] == 'failed']
    assert all(row['failure_class'] == kind and not row['transport_will_retry'] for row in failures)
    assert [row['repair_request_id'] for row in failures] == [a['request_id'] for a in attempts[:count]]
    expected = [5] if recovers else [5, 10, 20, 40, 80, 160] if count == 7 else []
    assert [a['retry_wait_seconds'] for a in attempts if a.get('will_retry')] == expected
    assert sum(waits) == sum(expected) <= 360
    assert budget.summary()['unsettled_attempts'] == count
    assert budget.summary()['tokens_or_unsettled_bounds']['output_tokens'] == count * OUTPUT_CAP + (17 if recovers else 0)
    counter = json.loads((r.directory / 'counter.json').read_text())
    assert counter['attempts'] == calls and counter['identity']['cap'] == 80
    assert len(list((r.directory / 'recoveries').glob('*.json'))) == len(expected)
    assert 'PRIVATE' not in json.dumps(rows) and 'SECRET' not in json.dumps(rows)
    review_rows = [json.loads(line) for line in (r.directory / 'reviews.jsonl').read_text().splitlines()]
    assert len([row for row in review_rows if row['status'] == 'started']) == calls


def test_transient_retries_do_not_consume_format_retry(tmp_path, budget):
    client = RawClient([HTTPFailure(503), response('{bad'), HTTPFailure(503),
                        response(json.dumps(review('accept')))], budget)
    r = reviewer(tmp_path, budget, client)
    p = JointPolicy(CaptureStudent(['Action: look']), 't', 'G', r)
    assert p([], context()) == 'Action: look'
    attempts = p.steps[0]['review_attempts']
    assert [a['retry'] for a in attempts] == [0, 0, 1, 1]
    assert [a['attempt_index'] for a in attempts] == [0, 1, 0, 1]
    assert [a['status'] for a in attempts] == ['failed', 'invalid', 'failed', 'valid']
    assert client.requests[0] == client.requests[1] and client.requests[2] == client.requests[3]
    assert 'invalid_json' in client.requests[2]['input'][0]['content']
    assert budget.summary()['reserved_attempts'] == 4 and budget.summary()['stop_reason'] is None


def test_reviewer_retries_exhausted_rate_limits(tmp_path, budget):
    client = RawClient([Limit429('SECRET')] * 20 + [response(json.dumps(review('accept')))], budget)
    r = reviewer(tmp_path, budget, client)
    p = JointPolicy(CaptureStudent(['Action: look']), 't', 'G', r)
    assert p([], context()) == 'Action: look'
    assert len(p.steps[0]['review_attempts']) == 2 and len(client.requests) == 21
    assert budget.summary()['reserved_attempts'] == 21 and budget.summary()['unsettled_attempts'] == 20
    rows = [json.loads(line) for line in (r.directory / 'usage.jsonl').read_text().splitlines()]
    failures = [row for row in rows if row['status'] == 'failed']
    assert all(row['transport_will_retry'] and not row['repair_will_retry'] for row in failures[:-1])
    assert not failures[-1]['transport_will_retry'] and failures[-1]['repair_will_retry']


@pytest.mark.parametrize('stop', ['teacher_quota', 'global_output_tokens_cap'])
def test_reviewer_retry_observes_other_worker_stop(tmp_path, budget, stop):
    client = RawClient([HTTPFailure(503), response(json.dumps(review('accept')))], budget)
    r = reviewer(tmp_path, budget, client)
    other = SharedTeacherBudget(budget.path, **budget.caps)
    waits = []
    def stop_during_wait(seconds):
        waits.append(seconds)
        other.halt(stop)
    r.teacher.teacher.sleep = stop_during_wait
    with pytest.raises(TeacherCallError):
        JointPolicy(CaptureStudent(['Action: look']), 't', 'G', r)([], context())
    assert waits == [1] and len(client.requests) == budget.summary()['reserved_attempts'] == 1
    assert r.failure == budget.summary()['stop_reason'] == stop
    assert not list((r.directory / 'recoveries').glob('*.json'))


def test_reviewer_retry_cannot_exceed_budget(tmp_path):
    budget = SharedTeacherBudget(tmp_path / 'global.jsonl', input_tokens=1_000_000, output_tokens=OUTPUT_CAP)
    client = RawClient([HTTPFailure(503), response(json.dumps(review('accept')))], budget)
    r = reviewer(tmp_path, budget, client)
    with pytest.raises(TeacherCallError):
        JointPolicy(CaptureStudent(['Action: look']), 't', 'G', r)([], context())
    assert len(client.requests) == budget.summary()['reserved_attempts'] == 1
    assert r.failure == 'global_output_tokens_cap'


def test_reviewer_does_not_recover_historical_halt(tmp_path, budget):
    client = RawClient([HTTPFailure(503), response(json.dumps(review('accept')))], budget)
    r = reviewer(tmp_path, budget, client)
    prompt = review_prompt([], context(), 'Action: look', FactTable(INITIAL), 40)
    r.reservation.input_bound = 10000
    with pytest.raises(TeacherCallError):
        r.teacher.generate([], r.task, dict(observation=canonical_bytes(prompt.to_json()).decode(),
                                           admissible_commands=[]))
    assert budget.summary()['stop_reason'] is None  # Only the local cap is halted.
    with pytest.raises(TeacherCallError):
        JointPolicy(CaptureStudent(['Action: look']), 't', 'G', r)([], context())
    assert len(client.requests) == 1 and not list((r.directory / 'recoveries').glob('*.json'))


def test_reviewer_recovered_failure_survives_resume_reconciliation(tmp_path, budget):
    from tools.teacher_takeover import _reconcile
    client = RawClient([HTTPFailure(503), response(json.dumps(review('accept')))], budget)
    r = reviewer(tmp_path / 'calls/key', budget, client)
    p = JointPolicy(CaptureStudent(['Action: look']), 't', 'G', r)
    assert p([], context()) == 'Action: look'
    before = budget.summary()
    _reconcile(tmp_path, budget)
    assert budget.summary() == before and before['stop_reason'] is None


def test_reviewer_cache_hit_is_other_and_never_retried(tmp_path, budget):
    client = RawClient([response(json.dumps(review('accept')))], budget)
    r = reviewer(tmp_path, budget, client)
    p = JointPolicy(CaptureStudent(['Action: look']), 't', 'G', r)
    assert p([], context()) == 'Action: look'
    with pytest.raises(TeacherCallError):
        p([], context())
    assert len(client.requests) == budget.summary()['reserved_attempts'] == 1
    attempts = p.steps[-1]['review_attempts']
    assert len(attempts) == 1 and attempts[0]['failure_class'] == 'OTHER'
    assert not attempts[0]['will_retry'] and budget.summary()['stop_reason'] == 'teacher_halted'


def test_reviewer_retries_connection_failure_before_transport_start(tmp_path, budget):
    client = RawClient([response(json.dumps(review('accept')))], budget)
    r = reviewer(tmp_path, budget, client)
    native_client = r.teacher.teacher._client
    failures = [ConnectionError('SECRET')]
    def connect():
        if failures:
            raise failures.pop()
        return native_client()
    r.teacher.teacher._client = connect
    p = JointPolicy(CaptureStudent(['Action: look']), 't', 'G', r)
    assert p([], context()) == 'Action: look'
    assert len(client.requests) == 1 and budget.summary()['reserved_attempts'] == 2
    assert budget.summary()['stop_reason'] is None
    assert [a['failure_class'] for a in p.steps[0]['review_attempts']] == ['TRANSIENT', None]


def case(original):
    return dict(case_id='case', task_id=original.task_id, group=GROUPS[5], source={}, branch_index=10)


def test_execution_records_metrics_prefix_and_resume(original, tmp_path, budget):
    r = ScriptReviewer([review(action='open cabinet 2'), review('accept')])
    student = CaptureStudent(['Action: look', 'Action: open cabinet 2', 'Action: take apple 1 from cabinet 2'])
    kwargs = dict(mode='G', decoding='greedy', seed=None, env_factory=ScriptEnvironment,
        student_factory=lambda _: student, reviewer_factory=lambda _: r, output=tmp_path,
        run_sha256='unit', budget=budget)
    row = execute_joint(case(original), original, **kwargs)
    assert row['prefix_verified'] and row['success'] and not row['void']
    assert row['metrics']['found'] and row['metrics']['picked']
    assert row['metrics']['guides'] == row['metrics']['guided_steps_followed'] == 1
    assert row['metrics']['teacher_executed_steps'] == 0
    assert row['execution']['steps'][:10] == original.to_json()['steps'][:10]
    assert row['steps'][0]['observation'] == 'You see an apple 1.'
    assert row['steps'][0]['refused'] is False
    assert all({'proposal_reply', 'proposal_action', 'proposal_outcome', 'facts_sha256', 'review_request_id',
                'review', 'review_validation', 'decision', 'guidance_text', 'regenerated_reply', 'regenerated_action',
                'executed_reply', 'executed_action', 'executor', 'observation', 'refused'} <= s.keys() for s in row['steps'])
    assert execute_joint(case(original), original, **kwargs) == row and len(r.prompts) == 2
    assert 'Guidance:' not in json.dumps(row['execution'])


def test_failed_review_void_and_no_execution(original, tmp_path):
    budget = SharedTeacherBudget(tmp_path / 'shared.jsonl', input_tokens=100_000, output_tokens=0)
    client = RawClient([response(json.dumps(review('accept')))], budget)
    r = reviewer(tmp_path, budget, client)
    row = execute_joint(case(original), original, mode='G', decoding='greedy', seed=None,
        env_factory=ScriptEnvironment, student_factory=lambda _: CaptureStudent(['Action: look']),
        reviewer_factory=lambda _: r, output=tmp_path, run_sha256='unit', budget=budget)
    assert row['void'] and row['success'] == 'unknown' and row['metrics'] is None
    assert row['void_reason'] == 'global_output_tokens_cap'
    assert len(row['execution']['steps']) == 10 and not client.requests
    assert row['steps'][0]['review_validation'] == 'failed'
    assert row['steps'][0]['executed_reply'] is None


def test_prefix_mismatch_no_calls_and_from_start(original, tmp_path, budget):
    student = CaptureStudent(['Action: take apple 1 from cabinet 2'])
    args = dict(mode='U', decoding='greedy', seed=None, student_factory=lambda _: student,
                reviewer_factory=lambda _: pytest.fail('R in U'), run_sha256='unit', budget=budget)
    row = execute_joint(case(original), original, env_factory=partial(ScriptEnvironment, mismatch=True), output=tmp_path / 'bad', **args)
    assert row['void'] and not row['prefix_verified'] and not student.prompts
    row = execute_joint(case(original), original, env_factory=ScriptEnvironment, output=tmp_path / 'start', from_start=True, **args)
    assert row['branch_index'] == 0 and row['success'] and len(student.prompts[0]) == 2


@pytest.mark.parametrize('branch_index', [1, 3, 15, 39])
def test_non_cg_branch_replay_and_metrics(original, tmp_path, budget, branch_index):
    student = CaptureStudent(['Action: take apple 1 from cabinet 2'])
    row = execute_joint(dict(case(original), branch_index=branch_index), original,
        mode='U', decoding='sampled', seed=2, env_factory=ScriptEnvironment,
        student_factory=lambda _: student, reviewer_factory=lambda _: pytest.fail('R in U'),
        run_sha256='variable-branch', budget=budget, output=tmp_path)
    assert row['prefix_verified'] and row['success'] and row['branch_index'] == branch_index
    assert row['execution']['steps'][:branch_index] == original.to_json()['steps'][:branch_index]
    assert len(student.prompts[0]) == 2 + 2 * branch_index
    assert row['metrics']['steps_used_after_branch'] == row['metrics']['steps_to_success'] == 1
    assert row['metrics']['steps_to_find'] == 1 and row['metrics']['picked']
    assert row['metrics']['steps_remaining_when_found'] == 39 - branch_index


def test_sampled_regeneration_uses_same_request_seed():
    student = object.__new__(SampledStudent)
    student.sampling_seed, student.task_id = 7, 'train/t'
    h = [{'visible': context(), 'reply': 'Action: look'}]
    injected, _ = injected_context(h, context(), {0: 'test'})
    assert student.request_sampling(h, {}) == student.request_sampling(injected, {})
    assert student.request_sampling(h, {})['seed'] == int(digest([7, 'train/t', 1])[:8], 16)


def test_dry_cli_shared_budget_two_invocations_pool_resume_and_summary(original, tmp_path):
    coverage, _ = coverage_fixture(tmp_path, original)
    spec = tmp_path / 'spec.json'
    assert takeover_main(['spec-from-cg', '--failures', str(coverage / 'failures.json'), '--coverage-root', str(coverage),
        '--groups', GROUPS[5], '--dry-run', '--output', str(spec)]) == 0
    ledger = tmp_path / 'shared.jsonl'
    common = ['run', '--spec', str(spec), '--dry-run', '--seeds', '0', '--workers', '2', '--budget-ledger', str(ledger)]
    first = tmp_path / 'first'
    assert main([*common, '--output', str(first)]) == 0
    records = {p.name: p.read_bytes() for p in (first / 'records').glob('*.json')}
    assert len(records) == 6
    state1 = json.loads(ledger.read_text().splitlines()[-1])
    assert main([*common, '--output', str(first), '--resume']) == 0
    assert records == {p.name: p.read_bytes() for p in (first / 'records').glob('*.json')}
    assert json.loads(ledger.read_text().splitlines()[-1])['reserved_attempts'] == state1['reserved_attempts']
    assert main([*common, '--output', str(tmp_path / 'second')]) == 0
    assert json.loads(ledger.read_text().splitlines()[-1])['reserved_attempts'] == 2 * state1['reserved_attempts']
    assert main(['summarize', '--rollouts', str(first), '--output', str(tmp_path / 'summary')]) == 0
    summary = json.loads((tmp_path / 'summary/summary.json').read_text())
    assert len(summary['per_case']) == 6 and not summary['missing_records']
    assert summary['bootstrap'] == dict(unit='case', resamples=10000, numpy_seed=0)
    assert summary['unknown_usage_attempts'] == 0
    assert {p['comparison'] for p in summary['paired_differences']} == {'G-U', 'G1-U', 'G-G1'}
    with pytest.raises(ValueError):
        main([*common, '--output', str(first), '--resume', '--max-teacher-output-tokens', '1'])
    plan = json.loads((first / 'plan.json').read_text())
    rows = [json.loads(v) for v in records.values()]
    with pytest.raises(ValueError):
        summarize_records(plan, [*rows, rows[0]])
    incomplete = summarize_records(plan, rows[:-1])
    assert incomplete['missing_records']
    assert any(r['success_rate'] is None for r in incomplete['per_case'])


def test_continuous_G_injects_only_guided_observations():
    r = ScriptReviewer([review(), review('accept'), review(action='go to cabinet 1')])
    student = CaptureStudent(['Action: look'])
    p = JointPolicy(student, 't', 'G', r)
    h, observations = [], [context(), context('Second'), context('Third')]
    for visible in observations:
        reply = p(h, visible)
        h.append({'visible': visible, 'reply': reply})
    assert len(student.prompts) == 5
    assert p.injections.keys() == {0, 2}
    final = student.prompts[-1]
    assert 'Guidance:' in json.loads(final[1]['content'])['observation']
    assert 'Guidance:' not in json.loads(final[3]['content'])['observation']
    assert 'Guidance:' in json.loads(final[5]['content'])['observation']
    assert 'Guidance:' not in json.dumps(observations)
    assert all('Guidance:' not in json.dumps(prompt.payload) for prompt in r.prompts)


class RefusedEnvironment(ScriptEnvironment):
    def step(self, action):
        self.index += 1
        return self.result('Nothing happens.')


def test_refused_counts_include_prefix_and_do_not_call_reviewer(original, tmp_path, budget):
    row = execute_joint(case(original), original, mode='U', decoding='greedy', seed=None,
        env_factory=RefusedEnvironment, student_factory=lambda _: CaptureStudent(['Action: look']),
        reviewer_factory=lambda _: pytest.fail('R called'), output=tmp_path, run_sha256='refused',
        budget=budget, from_start=True)
    assert not row['void'] and row['metrics']['refused_actions'] == 40
    assert row['metrics']['repeated_refused_actions'] == 39
    assert row['metrics']['proposals_repeating_refused'] == 39
    assert row['steps'][0]['repeated_refused_action'] is False
    assert row['steps'][1]['repeated_refused_action'] is True


@pytest.mark.parametrize('bad', ['empty', 'truncated', 'unknown_usage'])
def test_failed_review_never_becomes_accept(tmp_path, budget, bad):
    result = response('' if bad == 'empty' else json.dumps(review('accept')))
    if bad == 'truncated':
        result.status = 'incomplete'
        result.incomplete_details = {'reason': 'max_output_tokens'}
    if bad == 'unknown_usage':
        result.usage = None
    client = RawClient([result], budget)
    p = JointPolicy(CaptureStudent(['Action: look']), 't', 'G', reviewer(tmp_path, budget, client))
    with pytest.raises(TeacherCallError):
        p([], context())
    assert p.steps[0]['decision'] is None and p.steps[0]['executed_reply'] is None
    assert p.steps[0]['review_validation'] == 'failed'
    assert budget.summary()['stop_reason'] is not None
    assert len(client.requests) == 1
    assert p.steps[0]['review_attempts'][0]['failure_class'] == 'OTHER'


def test_case_cluster_bootstrap_and_complete_pair_denominators():
    from repair.joint import run_key
    plan = {'run_sha256': 'stat', 'identity': dict(cases=[dict(case_id=c) for c in ['a', 'b', 'c']],
        modes=['U', 'G', 'G1'], seeds=[0, 1], dry_run=True)}
    rows = []
    for case_id in ['a', 'b', 'c']:
        for mode in ['U', 'G', 'G1']:
            for decoding, seed in [('greedy', None), ('sampled', 0), ('sampled', 1)]:
                success = mode == 'G' and case_id != 'c'
                metrics = dict(success=success, picked=success, found=success, teacher_executed_steps=0,
                    steps_used_after_branch=10, guided_steps_followed=1, guides=1,
                    steps_to_find=2 if success else None, steps_remaining_when_found=28 if success else None)
                rows.append(dict(run_key=run_key(case_id, mode, decoding, seed), run_sha256='stat',
                    case_id=case_id, mode=mode, decoding=decoding, seed=seed, void=False, metrics=metrics, steps=[]))
    summary = summarize_records(plan, rows)
    pair = next(r for r in summary['paired_differences'] if
                (r['decoding'], r['comparison'], r['metric']) == ('sampled', 'G-U', 'success_rate'))
    assert pair['difference'] == pytest.approx(2/3)
    assert pair['ci95'] == [0., 1.] and pair['included_cases'] == ['a', 'b', 'c']
    assert summarize_records(plan, rows) == summary
    next(r for r in rows if (r['case_id'], r['mode'], r['decoding'], r['seed']) == ('a', 'G', 'sampled', 1))['void'] = True
    incomplete = summarize_records(plan, rows)
    pair = next(r for r in incomplete['paired_differences'] if
                (r['decoding'], r['comparison'], r['metric']) == ('sampled', 'G-U', 'success_rate'))
    assert pair['excluded_cases'] == ['a'] and pair['difference'] == .5
