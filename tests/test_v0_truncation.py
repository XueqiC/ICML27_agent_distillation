"""Reasoning-only truncation is recoverable; all clients/environments are fake."""
import json
from types import SimpleNamespace as NS

import pytest

from repair.dry_fakes import FakeEnvironment, N
from repair.execute import execute_actions
from repair.prompts import rationale_prompt
from repair.records import TrajectoryRecord
from repair.takeover import SharedTeacherBudget
from repair.v0_common import completed
from repair.v0_data import decision_rows
from repair.v0_review import WholeReviewer, WholeTransport, make_rationale_labeler, review_prompt
from repair.v0_transport import TeacherOutputTruncated
from sed.config import load_config
from teachers.openai_teacher import TeacherCallError
from tools import v0_round as cli
from test_capped_teacher import http_429
from test_v0_pipeline import WholeClient, arguments, fake_runtime, setup_synth
from test_v0_round import TASK, failure, manifest_at


def read_rows(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


class OutputClient(WholeClient):
    def __init__(self, outcomes, *, label=False, before_call=None):
        super().__init__()
        self.outcomes, self.caps, self.label, self.before_call = iter(outcomes), [], label, before_call

    def create(self, **kwargs):
        cap = kwargs['max_output_tokens']
        self.caps.append(cap)
        if self.before_call:
            self.before_call(cap)
        outcome = next(self.outcomes)
        if isinstance(outcome, Exception):
            raise outcome
        if isinstance(outcome, int):
            return NS(status_code=outcome, headers={}, parse=lambda: {})
        if outcome == 'truncated':
            result = NS(output_text='', model='FAKE_NO_MODEL', id='truncated', status='incomplete',
                incomplete_details={'reason': 'max_output_tokens'},
                usage={'input_tokens': 100, 'output_tokens': cap,
                       'output_tokens_details': {'reasoning_tokens': cap}})
        elif outcome in ('unknown', 'missing_usage'):
            result = NS(output_text='', model='FAKE_NO_MODEL', id='unknown', status='incomplete',
                incomplete_details={'reason': 'content_filter' if outcome == 'unknown' else 'max_output_tokens'},
                usage=None if outcome == 'missing_usage' else {'input_tokens': 100, 'output_tokens': cap})
        elif outcome == 'invalid' or self.label:
            result = NS(output_text='{}' if outcome == 'invalid' else json.dumps(dict(
                reasoning='Act on the public evidence.', evidence_steps=[-1], unsupported=False)),
                model='FAKE_NO_MODEL', id='fake', status='completed',
                usage={'input_tokens': 100, 'output_tokens': 80, 'output_tokens_details': {'reasoning_tokens': 0}})
        else:
            return super().create(**kwargs)
        return NS(status_code=200, headers={}, parse=lambda: result)


def transport_at(tmp_path, client, *, output_tokens=200000, label=False):
    budget = SharedTeacherBudget(tmp_path / 'budget.jsonl', input_tokens=1000000, output_tokens=output_tokens)
    make = make_rationale_labeler if label else WholeTransport
    transport = make(load_config('configs/base_e4b.yaml'), TASK, tmp_path / 'teacher', budget, client=client)
    return transport, budget


def prompt_for(label):
    if label:
        return rationale_prompt({'instruction': 'Look around.'}, 'Room.', [], 'look')
    return review_prompt(TrajectoryRecord.from_json(failure()['execution']))


@pytest.mark.parametrize('label', [False, True])
def test_truncated_then_ok_reserves_each_cap_and_settles_actual_usage(tmp_path, label):
    def check_reserved(cap):
        reservation = [r for r in read_rows(tmp_path / 'budget.jsonl') if r['event'] == 'reservation'][-1]
        assert reservation['amounts']['output_tokens'] == cap
    client = OutputClient(['truncated', 'ok'], label=label, before_call=check_reserved)
    transport, budget = transport_at(tmp_path, client, label=label)
    prompt = prompt_for(label)
    raw = transport(prompt)
    assert json.loads(raw) and client.caps == [16384, 32768]
    summary = budget.summary()
    assert summary['stop_reason'] is None and summary['unsettled_attempts'] == 0
    assert summary['reserved_attempts'] == 2
    assert summary['tokens_or_unsettled_bounds'] == {'input_tokens': 200, 'output_tokens': 16384 + 80}
    assert summary['reported_usage']['reasoning_tokens'] == 16384
    receipts = read_rows(tmp_path / 'teacher/requests.jsonl')
    assert [r['status'] for r in receipts] == ['started', 'teacher_output_truncated', 'started', 'completed']
    assert receipts[1]['raw_output'] == ''
    assert receipts[1]['metadata']['incomplete_reason'] == 'max_output_tokens'
    # Recover the result even if the caller crashed before writing its receipt.
    resumed, _ = transport_at(tmp_path, client, label=label)
    assert resumed(prompt) == raw and client.caps == [16384, 32768]


def test_truncation_does_not_consume_the_format_retry_and_caps_reset(tmp_path):
    client = OutputClient(['truncated', 'invalid', 'ok'])
    transport, budget = transport_at(tmp_path, client)
    trace = TrajectoryRecord.from_json(failure()['execution'])
    value, attempts = WholeReviewer(transport, tmp_path / 'requests').request(trace, 0, [])
    assert value and [r['status'] for r in attempts] == ['invalid', 'valid']
    assert client.caps == [16384, 32768, 16384]
    assert budget.summary()['stop_reason'] is None


@pytest.mark.parametrize('label', [False, True])
def test_truncated_twice_is_durable_and_never_halts_shared_budget(tmp_path, label):
    client = OutputClient(['truncated', 'truncated'], label=label)
    transport, budget = transport_at(tmp_path, client, label=label)
    prompt = prompt_for(label)
    with pytest.raises(TeacherOutputTruncated) as failure_info:
        transport(prompt)
    assert [r['output_cap'] for r in failure_info.value.attempts] == [16384, 32768]
    assert budget.summary()['stop_reason'] is None
    assert budget.summary()['reported_usage']['output_tokens'] == 16384 + 32768
    resumed, _ = transport_at(tmp_path, client, label=label)
    with pytest.raises(TeacherOutputTruncated) as resumed_info:
        resumed(prompt)
    assert resumed_info.value.attempts == failure_info.value.attempts
    assert client.caps == [16384, 32768]


def test_unknown_started_retry_still_halts_on_resume(tmp_path):
    # Crash after starting the retry, before its transport reservation.
    class Interrupted(WholeTransport):
        def _transport(self, cap):
            if cap == self.retry_output_cap:
                raise KeyboardInterrupt
            return super()._transport(cap)
    client = OutputClient(['truncated', 'ok'])
    transport, budget = transport_at(tmp_path, client)
    interrupted = Interrupted(load_config('configs/base_e4b.yaml'), TASK, transport.directory, budget, client=client)
    with pytest.raises(KeyboardInterrupt):
        interrupted(prompt_for(False))
    # The second attempt is only a local start; no transport was reserved. This
    # unknown caller outcome deliberately remains fail-loud on resume.
    with pytest.raises(TeacherCallError):
        transport(prompt_for(False))
    assert budget.summary()['stop_reason'] == 'v0_interrupted_review_request'
    assert client.caps == [16384]


def test_resume_after_settled_truncation_uses_only_the_larger_cap(tmp_path, monkeypatch):
    client = OutputClient(['truncated', 'ok'])
    transport, budget = transport_at(tmp_path, client)
    original_check = budget.check
    def crash_after_receipt():
        original_check()
        path = tmp_path / 'teacher/requests.jsonl'
        if path.exists() and read_rows(path)[-1]['status'] == 'teacher_output_truncated':
            raise KeyboardInterrupt
    monkeypatch.setattr(budget, 'check', crash_after_receipt)
    with pytest.raises(KeyboardInterrupt):
        transport(prompt_for(False))
    assert client.caps == [16384]
    resumed, resumed_budget = transport_at(tmp_path, client)
    assert json.loads(resumed(prompt_for(False)))
    assert client.caps == [16384, 32768]
    assert resumed_budget.summary()['stop_reason'] is None
    assert resumed_budget.summary()['reserved_attempts'] == 2


def test_truncated_twice_records_round_and_continues_other_failures(tmp_path, monkeypatch):
    root = tmp_path / 'round'
    args = arguments(root)
    cli.round_root(args)
    manifest = setup_synth(root)
    fake_runtime(monkeypatch, manifest)
    cli.run_attempts(args, root, manifest)
    client = OutputClient(['truncated', 'truncated', 'ok'])
    monkeypatch.setattr('repair.v0_review.WholeTransport', lambda *a, **kw: WholeTransport(*a, **kw, client=client))
    repairs = cli.run_reviews(args, root, manifest, root / 'attempt')
    assert len(repairs) == 2 and [r['verified'] for r in repairs] == [False, True]
    failed_round, = repairs[0]['rounds']
    assert failed_round['status'] == 'teacher_output_truncated'
    assert failed_round['review'] is None and failed_round['executions'] == []
    assert 'execution' not in failed_round and 'patch' not in failed_round
    assert repairs[0]['success'] is False
    assert repairs[0]['teacher_tokens']['output_tokens'] == 16384 + 32768
    assert len(failed_round['requests'][0]['teacher_attempts']) == 2
    assert client.caps == [16384, 32768, 16384]
    assert completed(root / 'review') and cli.budget_for(root).summary()['stop_reason'] is None
    args.resume = True
    assert cli.run_reviews(args, root, manifest, root / 'attempt') == repairs
    assert client.caps == [16384, 32768, 16384]


@pytest.mark.parametrize('error,reason', [
    (http_429(body={'error': {'code': 'insufficient_quota'}}), 'teacher_quota'),
    (401, 'review_transport_failed'), (503, 'review_transport_failed'),
    ('unknown', 'review_transport_failed'), ('missing_usage', 'review_transport_failed'),
])
def test_real_transport_and_unknown_outcomes_still_halt(tmp_path, error, reason):
    client = OutputClient([error])
    transport, budget = transport_at(tmp_path, client)
    trace = TrajectoryRecord.from_json(failure()['execution'])
    with pytest.raises(TeacherCallError):
        WholeReviewer(transport, tmp_path / 'requests').request(trace, 0, [])
    assert budget.summary()['stop_reason'] == reason
    with pytest.raises(Exception):
        budget.reserve('other-worker', 1, 16384)
    assert client.caps == [16384] and not (tmp_path / 'requests').exists()
    assert 'PRIVATE_PROVIDER_EXCEPTION' not in (tmp_path / 'teacher/requests.jsonl').read_text()


def test_retry_cap_exceeding_remaining_budget_halts_before_second_call(tmp_path):
    client = OutputClient(['truncated', 'ok'])
    transport, budget = transport_at(tmp_path, client, output_tokens=16384 + 32768 - 1)
    with pytest.raises(TeacherCallError):
        transport(prompt_for(False))
    assert client.caps == [16384]
    assert budget.summary()['stop_reason'] == 'global_output_tokens_cap'
    assert budget.summary()['tokens_or_unsettled_bounds']['output_tokens'] == 16384
    assert budget.summary()['unsettled_attempts'] == 0


def test_truncated_rationale_is_audited_action_only_and_not_training_text(tmp_path):
    execution = execute_actions(FakeEnvironment, TASK, N, 40)
    episode = dict(episode_id='repair-f1', task_id=TASK, success=True, execution=execution.to_json(),
                   prefix_length=0, executors=['student'] * 3 + ['patch'] + ['student'] * (len(N) - 4))
    client = OutputClient(['truncated', 'truncated'], label=True)
    transport, budget = transport_at(tmp_path, client, label=True)
    audit = []
    labeler = cli.DurableLabels(lambda: transport, tmp_path / 'requests')
    rows, excluded = decision_rows(episode, manifest_at(tmp_path), bucket='R', episode_ref='repair.json',
                                  labeler=labeler, audit=audit.append)
    assert rows[3]['evidence']['label_source'] == 'action_only_truncated_label'
    assert rows[3]['messages'][-1]['content'] == 'Action: ' + N[3]
    assert excluded['truncated_rationale_label'] == 1
    assert audit[0]['label_check'] == 'teacher_output_truncated' and audit[0]['raw_output'] is None
    assert len(audit[0]['teacher_attempts']) == 2 and budget.summary()['stop_reason'] is None
    again, counts = decision_rows(episode, manifest_at(tmp_path), bucket='R', episode_ref='repair.json', labeler=labeler)
    assert (again, counts) == (rows, excluded) and client.caps == [16384, 32768]
