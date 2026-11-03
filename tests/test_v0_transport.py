"""Incident 2: fake-only timeout recovery, JSON receipts and accounting."""
from dataclasses import replace
import json

import httpx
import openai
import pytest

from repair.records import TrajectoryRecord
from repair.dry_fakes import FakeEnvironment, N
from repair.execute import execute_actions
from repair.v0_data import decision_rows
from repair.v0_common import completed
from repair.v0_review import WholeReviewer, WholeTransport
from repair.v0_transport import TeacherTimeout, V0TeacherMixin
from teachers.openai_teacher import TeacherCallError
from teachers.rate_limit import RateLimiter
from tools import v0_round as cli
from tools.v0_usage_check import audit_usage
from test_capped_teacher import http_429
from test_v0_pipeline import arguments, fake_runtime, setup_synth
from test_v0_round import TASK, failure, manifest_at
from test_v0_truncation import OutputClient, prompt_for, read_rows, transport_at
from test_stage2c_teacher import FakeClock


def timeout():
    return openai.APITimeoutError(request=httpx.Request('POST', 'https://fake.invalid'))


@pytest.fixture(autouse=True)
def no_backoff_sleep(monkeypatch):
    clock = FakeClock()
    original = V0TeacherMixin.__init__
    def initialize(self, **kwargs):
        original(self, **kwargs)
        self.clock, self.sleep, self.jitter = clock, clock.sleep, lambda lo, hi: 0
    monkeypatch.setattr(V0TeacherMixin, '__init__', initialize)
    original_limiter = RateLimiter.__init__
    def initialize_limiter(self, *args, **kwargs):
        original_limiter(self, *args, **dict(kwargs, clock=clock, sleep=clock.sleep))
    monkeypatch.setattr(RateLimiter, '__init__', initialize_limiter)
    return clock.waits


@pytest.mark.parametrize('label', [False, True])
def test_sdk_gets_600_second_timeout_without_hidden_retries(tmp_path, monkeypatch, label):
    constructors = []
    client = OutputClient(['ok'], label=label)
    def fake_sdk(**kwargs):
        constructors.append(kwargs)
        return client
    monkeypatch.setattr(openai, 'OpenAI', fake_sdk)
    monkeypatch.setenv('OPENAI_API_KEY', 'offline-test-only')
    transport, _ = transport_at(tmp_path, None, label=label)
    transport(prompt_for(label))
    options, = constructors
    assert options['max_retries'] == 0
    assert (options['timeout'].read, options['timeout'].write, options['timeout'].pool,
            options['timeout'].connect) == (600, 600, 600, 5)


@pytest.mark.parametrize('label', [False, True])
@pytest.mark.parametrize('error', [timeout(), ConnectionError(),
    openai.APIConnectionError(request=httpx.Request('POST', 'https://fake.invalid')),
    httpx.ConnectError('fake connection error')])
def test_connection_failure_then_success_settles_each_attempt_once(tmp_path, label, error, no_backoff_sleep):
    client = OutputClient([error, 'ok'], label=label)
    transport, budget = transport_at(tmp_path, client, label=label)
    raw = transport(prompt_for(label))
    assert no_backoff_sleep == [.5]
    summary = budget.summary()
    assert summary['reserved_attempts'] == 2 and summary['unsettled_attempts'] == 0
    assert summary['reported_usage']['input_tokens'] == 100
    assert summary['reported_usage']['output_tokens'] == 80
    assert summary['stop_reason'] is None
    usage = read_rows(tmp_path / 'teacher/usage.jsonl')
    failed, = [r for r in usage if r['status'] == 'failed']
    assert failed['usage'] is None and failed['usage_unreported'] is True
    assert failed['reported_usage'] == dict(input_tokens=0, output_tokens=0, reasoning_tokens=0)
    assert failed['will_retry'] and failed['retryable']
    success, = [r for r in usage if r['status'] == 'success']
    assert success['response_status'] == 'completed' and success['parser_error'] is None
    events = read_rows(budget.path)
    assert [r['event'] for r in events] == ['initialized', 'reservation', 'settlement', 'reservation', 'settlement']
    assert events[1]['amounts']['output_tokens'] == 16384 and events[2]['usage']['output_tokens'] == 0
    transport.teacher.check_health()
    resumed, _ = transport_at(tmp_path, client, label=label)
    assert resumed(prompt_for(label)) == raw
    assert budget.summary() == summary and len(client.caps) == 2
    assert audit_usage(tmp_path / 'teacher', budget.path)['settled_usage']['output_tokens'] == 80


@pytest.mark.parametrize('label', [False, True])
def test_three_timeouts_are_durable_and_next_request_can_run(tmp_path, label, no_backoff_sleep):
    client = OutputClient([timeout(), timeout(), timeout(), 'ok'], label=label)
    transport, budget = transport_at(tmp_path, client, label=label)
    prompt = prompt_for(label)
    with pytest.raises(TeacherTimeout) as error:
        transport(prompt)
    assert len(error.value.attempts) == 3 and no_backoff_sleep == [.5, 1.]
    assert error.value.attempts[-1]['recoverable_timeout'] is True
    assert budget.summary()['reported_usage']['output_tokens'] == 0
    assert budget.summary()['unsettled_attempts'] == 0 and budget.summary()['stop_reason'] is None
    transport.teacher.check_health()
    resumed, _ = transport_at(tmp_path, client, label=label)
    with pytest.raises(TeacherTimeout) as repeated:
        resumed(prompt)
    assert repeated.value.attempts == error.value.attempts and len(client.caps) == 3
    next_prompt = replace(prompt, **({'format_retry': True} if label else {'retry_reason': 'different_request'}))
    assert json.loads(resumed(next_prompt))
    resumed.teacher.check_health()
    assert budget.summary()['reserved_attempts'] == 4


def test_three_timeouts_record_round_and_continue_other_failures(tmp_path, monkeypatch):
    root = tmp_path / 'round'
    args = arguments(root)
    cli.round_root(args)
    manifest = setup_synth(root)
    fake_runtime(monkeypatch, manifest)
    cli.run_attempts(args, root, manifest)
    client = OutputClient([timeout(), timeout(), timeout(), 'ok'])
    monkeypatch.setattr('repair.v0_review.WholeTransport', lambda *a, **kw: WholeTransport(*a, **kw, client=client))
    repairs = cli.run_reviews(args, root, manifest, root / 'attempt')
    assert [r['verified'] for r in repairs] == [False, True]
    row, = repairs[0]['rounds']
    assert row['status'] == 'teacher_timeout' and row['review'] is None and row['executions'] == []
    assert 'patch' not in row and 'execution' not in row
    assert len(row['requests'][0]['teacher_attempts']) == 3
    assert completed(root / 'review') and cli.budget_for(root).summary()['stop_reason'] is None
    args.resume = True
    assert cli.run_reviews(args, root, manifest, root / 'attempt') == repairs
    assert len(client.caps) == 4


@pytest.mark.parametrize('error,reason', [
    (http_429(body={'error': {'code': 'insufficient_quota'}}), 'teacher_quota'),
    (401, 'review_transport_failed'), (403, 'review_transport_failed'),
])
def test_fatal_errors_after_timeout_still_halt_immediately(tmp_path, error, reason):
    client = OutputClient([timeout(), error, 'ok'])
    transport, budget = transport_at(tmp_path, client)
    with pytest.raises(TeacherCallError):
        transport(prompt_for(False))
    assert len(client.caps) == 2 and budget.summary()['stop_reason'] == reason
    with pytest.raises(Exception):
        budget.reserve('other-worker', 1, 1)


def test_rate_limits_have_their_own_retry_budget(tmp_path, no_backoff_sleep):
    client = OutputClient([timeout(), http_429(headers={'retry-after': '7'}), timeout(), 'ok'])
    transport, budget = transport_at(tmp_path, client)
    assert json.loads(transport(prompt_for(False)))
    assert len(client.caps) == 4 and no_backoff_sleep == [.5, 7, 1.]
    assert budget.summary()['stop_reason'] is None
    transport.teacher.check_health()


@pytest.mark.parametrize('outcomes,status', [(['invalid', 'ok'], 'valid'),
                                           (['invalid', 'invalid'], 'invalid')])
def test_strict_json_rejection_has_exactly_one_format_retry(tmp_path, outcomes, status):
    client = OutputClient(outcomes)
    transport, budget = transport_at(tmp_path, client)
    trace = TrajectoryRecord.from_json(failure()['execution'])
    value, attempts = WholeReviewer(transport, tmp_path / 'requests').request(trace, 0, [])
    assert len(attempts) == 2 and attempts[-1]['status'] == status
    assert bool(value) == (status == 'valid')
    rows = read_rows(tmp_path / 'teacher/usage.jsonl')
    ends = [r for r in rows if r['status'] in ('rejected', 'success')]
    assert ends[0]['error_type'] == 'ReviewInvalid'
    assert [r['status'] for r in ends] == ['rejected', 'success' if value else 'rejected']
    assert budget.summary()['reported_usage']['output_tokens'] == 160
    # The caller can recover both raw outputs without charging either again.
    assert WholeReviewer(transport, tmp_path / 'requests').request(trace, 0, [])[0] == value
    assert budget.summary()['reserved_attempts'] == 2
    # A recovered/state-link copy carrying usage must not be charged again.
    with (tmp_path / 'teacher/usage.jsonl').open('a') as handle:
        handle.write(json.dumps(dict(ends[0], status='state_link')) + '\n')
    report = audit_usage(tmp_path / 'teacher', budget.path)
    assert report['settled_attempts'] == 2 and report['settled_usage']['output_tokens'] == 160


def test_invalid_json_exhaustion_records_round_without_executing_patch(tmp_path, monkeypatch):
    root = tmp_path / 'round'
    args = arguments(root)
    cli.round_root(args)
    manifest = setup_synth(root)
    fake_runtime(monkeypatch, manifest)
    cli.run_attempts(args, root, manifest)
    client = OutputClient(['invalid', 'invalid', 'ok'])
    monkeypatch.setattr('repair.v0_review.WholeTransport', lambda *a, **kw: WholeTransport(*a, **kw, client=client))
    repairs = cli.run_reviews(args, root, manifest, root / 'attempt')
    assert [r['verified'] for r in repairs] == [False, True]
    row, = repairs[0]['rounds']
    assert row['status'] == 'format_retry_exhausted' and row['executions'] == [] and row['review'] is None
    assert len(row['requests']) == 2 and len(client.caps) == 3


def test_empty_completed_response_is_a_format_error(tmp_path):
    class EmptyClient(OutputClient):
        def create(self, **kwargs):
            response = super().create(**kwargs)
            if len(self.caps) == 1:
                response.parse().output_text = ''
            return response
    client = EmptyClient(['invalid', 'ok'])
    transport, budget = transport_at(tmp_path, client)
    trace = TrajectoryRecord.from_json(failure()['execution'])
    value, attempts = WholeReviewer(transport, tmp_path / 'requests').request(trace, 0, [])
    assert value and [r['status'] for r in attempts] == ['invalid', 'valid']
    assert budget.summary()['stop_reason'] is None


@pytest.mark.parametrize('outcomes,source,check', [
    ([timeout(), timeout(), timeout()], 'action_only_timeout_label', 'teacher_timeout'),
    (['invalid'], 'action_only_invalid_label', 'unexpected_fields'),
])
def test_failed_labels_are_audited_action_only_and_resumable(tmp_path, outcomes, source, check):
    execution = execute_actions(FakeEnvironment, TASK, N, 40)
    episode = dict(episode_id='repair-f1', task_id=TASK, success=True, execution=execution.to_json(),
        prefix_length=0, executors=['student'] * 3 + ['patch'] + ['student'] * (len(N) - 4))
    client = OutputClient(outcomes, label=True)
    transport, budget = transport_at(tmp_path, client, label=True)
    labeler = cli.DurableLabels(lambda: transport, tmp_path / 'requests')
    audit = []
    rows, excluded = decision_rows(episode, manifest_at(tmp_path), bucket='R', episode_ref='repair.json',
                                  labeler=labeler, audit=audit.append)
    assert rows[3]['evidence']['label_source'] == source
    assert rows[3]['messages'][-1]['content'] == 'Action: ' + N[3]
    assert audit[0]['label_check'] == check and budget.summary()['stop_reason'] is None
    if check == 'teacher_timeout':
        assert len(audit[0]['teacher_attempts']) == 3
    else:
        usage = read_rows(tmp_path / 'teacher/usage.jsonl')
        assert next(r for r in usage if r['status'] == 'rejected')['error_type'] == 'SchemaError'
    again = decision_rows(episode, manifest_at(tmp_path), bucket='R', episode_ref='repair.json', labeler=labeler)
    assert again == (rows, excluded) and len(client.caps) == len(outcomes)
