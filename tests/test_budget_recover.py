"""Offline maintenance of synthetic SharedTeacherBudget ledgers only."""
from datetime import datetime, timezone
import hashlib
import json

import pytest

from repair.runtime import CaseStopped
from repair.takeover import SharedTeacherBudget
from sed.storage import canonical_bytes
from tools import budget_recover


STATE_FIELDS = ('caps', 'tokens_or_unsettled_bounds', 'reported_usage',
                'unknown_quantities', 'reserved_attempts')


@pytest.fixture
def budget(tmp_path):
    return SharedTeacherBudget(tmp_path / 'budget.jsonl', input_tokens=1000, output_tokens=500)


def recover(path, capsys, *, expect='teacher_quota', reason='quota_topped_up', dry_run=False):
    args = ['--ledger', str(path), '--expect-stop', expect, '--reason', reason]
    if dry_run:
        args.append('--dry-run')
    code = budget_recover.main(args)
    captured = capsys.readouterr()
    assert not captured.err
    assert len(captured.out.splitlines()) == 1
    return code, json.loads(captured.out)


def settle(budget, request_id):
    budget.reserve(request_id, 40, 20)
    budget.settle(request_id, {'input_tokens': 10, 'output_tokens': 5})


@pytest.mark.parametrize('stop', ['teacher_quota', 'teacher_halted'])
def test_recover_preserves_checkpoint_and_supports_repeated_recovery(budget, capsys, stop):
    settle(budget, 'first')
    reason = '  Quota topped up / crédits restored  '
    for index in range(2):
        budget.halt(stop)
        before = budget.path.read_bytes()
        state = budget.summary()
        assert state['unknown_quantities']  # Unknown metadata must survive too.
        code, report = recover(budget.path, capsys, expect=stop, reason=reason)
        assert code == 0 and report['status'] == 'ok'
        assert report['ledger'] == str(budget.path)
        assert report['previous_stop_reason'] == stop
        assert report['unsettled_attempts'] == 0
        assert report['dry_run'] is False and report['recovery_row_durable'] is True
        assert report['rows_before'] == len(before.splitlines())
        assert report['rows_after'] == report['rows_before'] + 1
        after = budget.path.read_bytes()
        assert after.startswith(before)
        appended = after[len(before):]
        assert len(appended.splitlines()) == 1
        row = json.loads(appended)
        assert appended == canonical_bytes(row) + b'\n'
        timestamp = datetime.fromisoformat(row['timestamp'])
        assert timestamp.utcoffset() == timezone.utc.utcoffset(timestamp)
        assert row == {
            'event': 'recovered', 'reason': reason, 'previous_stop_reason': stop,
            'timestamp': row['timestamp'], 'through_rows': report['rows_before'],
            'ledger_prefix_sha256': hashlib.sha256(before).hexdigest(),
            **{key: state[key] for key in STATE_FIELDS}, 'stop_reason': None,
        }
        resumed = SharedTeacherBudget(budget.path, **budget.caps)
        summary = resumed.summary()
        assert summary['stop_reason'] is None
        for key in STATE_FIELDS:
            assert summary[key] == state[key] == report[key]
        settle(resumed, f'after-recovery-{index}')
        assert resumed.summary()['reserved_attempts'] == state['reserved_attempts'] + 1


def test_dry_run_checks_and_reports_without_appending(budget, capsys):
    settle(budget, 'first')
    budget.halt('teacher_quota')
    before = budget.path.read_bytes()
    state = budget.summary()
    code, report = recover(budget.path, capsys, dry_run=True)
    assert code == 0 and report['status'] == 'ok'
    assert report['dry_run'] is True and report['recovery_row_durable'] is False
    assert report['rows_before'] == report['rows_after'] == len(before.splitlines())
    assert report['previous_stop_reason'] == 'teacher_quota'
    assert report['unsettled_attempts'] == 0
    for key in STATE_FIELDS:
        assert report[key] == state[key]
    assert budget.path.read_bytes() == before
    assert budget.summary() == state


@pytest.mark.parametrize('stop', [
    'global_input_tokens_cap', 'global_output_tokens_cap',
    'global_reported_usage_exceeded_reservation', None, 'teacher_halted', 'other_failure',
])
@pytest.mark.parametrize('dry_run', [False, True])
def test_refuse_unexpected_stop_without_writing(budget, capsys, stop, dry_run):
    if stop in ('global_input_tokens_cap', 'global_output_tokens_cap'):
        with pytest.raises(CaseStopped, match=stop):
            budget.reserve('too-large', 1001 if stop == 'global_input_tokens_cap' else 1,
                           501 if stop == 'global_output_tokens_cap' else 1)
    elif stop == 'global_reported_usage_exceeded_reservation':
        budget.reserve('excess-usage', 1, 1)
        with pytest.raises(CaseStopped, match=stop):
            budget.settle('excess-usage', {'input_tokens': 1001, 'output_tokens': 1})
    elif stop is not None:
        budget.halt(stop)
    assert budget.summary()['stop_reason'] == stop
    before = budget.path.read_bytes()
    code, report = recover(budget.path, capsys, dry_run=dry_run)
    assert code != 0 and report['status'] == 'error'
    assert report['error_type'] == 'ValueError'
    assert 'does not match --expect-stop' in report['message']
    assert budget.path.read_bytes() == before


@pytest.mark.parametrize('truncate', [1, 10])
def test_damaged_ledger_uses_shared_loader_rejection(budget, capsys, truncate):
    budget.halt('teacher_quota')
    before = budget.path.read_bytes()[:-truncate]
    budget.path.write_bytes(before)
    with pytest.raises(ValueError) as error:
        SharedTeacherBudget(budget.path, **budget.caps)
    code, report = recover(budget.path, capsys)
    assert code != 0 and report['status'] == 'error'
    assert report['message'] == str(error.value) == 'Damaged global budget ledger'
    assert budget.path.read_bytes() == before


def test_unsettled_reservations_keep_bounds_and_match_request_ids(budget, capsys):
    budget.reserve('unsettled', 80, 30)
    budget.settle('unsettled', None)
    settle(budget, 'settled')
    budget.settle('settled', {'input_tokens': 10, 'output_tokens': 5})
    budget.halt('teacher_quota')
    state = budget.summary()
    code, report = recover(budget.path, capsys)
    assert code == 0
    assert report['unsettled_attempts'] == 1
    assert report['tokens_or_unsettled_bounds'] == {'input_tokens': 90, 'output_tokens': 35}
    summary = budget.summary()
    for key in STATE_FIELDS:
        assert report[key] == state[key] == summary[key]
    assert summary['unsettled_attempts'] == 1


@pytest.mark.parametrize('exists', [False, True])
def test_missing_or_empty_ledger_is_not_initialized(tmp_path, capsys, exists):
    path = tmp_path / 'missing.jsonl'
    if exists:
        path.write_bytes(b'')
    code, report = recover(path, capsys)
    assert code != 0 and report['status'] == 'error'
    assert ('empty' if exists else 'does not exist') in report['message']
    assert path.exists() == exists
    if exists:
        assert path.read_bytes() == b''


@pytest.mark.parametrize('reason', ['x' * 201, 'line\nbreak', 'line\rbreak', 'line\u2028break'])
def test_invalid_reason_refuses_without_writing(budget, capsys, reason):
    budget.halt('teacher_quota')
    before = budget.path.read_bytes()
    code, report = recover(budget.path, capsys, reason=reason)
    assert code != 0 and report['status'] == 'error'
    assert '--reason must' in report['message']
    assert budget.path.read_bytes() == before


def test_reason_at_character_limit_is_stored_verbatim(budget, capsys):
    budget.halt('teacher_quota')
    reason = 'é' * 200
    code, _ = recover(budget.path, capsys, reason=reason)
    assert code == 0
    assert json.loads(budget.path.read_bytes().splitlines()[-1])['reason'] == reason


@pytest.mark.parametrize('arguments', [
    ['--expect-stop', 'global_input_tokens_cap', '--reason', 'recovery'],
    ['--expect-stop', 'teacher_quota'],
    ['--reason', 'recovery'],
])
def test_cli_requires_reason_and_an_allowed_expected_stop(budget, capsys, arguments):
    budget.halt('teacher_quota')
    before = budget.path.read_bytes()
    assert budget_recover.main(['--ledger', str(budget.path), *arguments]) != 0
    report = json.loads(capsys.readouterr().out)
    assert report['status'] == 'error' and report['message']
    assert budget.path.read_bytes() == before


@pytest.mark.parametrize('failure', ['stop_reason', 'totals', 'exception'])
def test_post_append_verification_failure_reports_durable_row(budget, capsys, monkeypatch, failure):
    budget.halt('teacher_quota')
    before = budget.path.read_bytes()
    original = SharedTeacherBudget.summary

    def changed_summary(self):
        summary = original(self)
        if failure == 'exception':
            raise RuntimeError('PRIVATE_EXTERNAL_EXCEPTION')
        if failure == 'stop_reason':
            summary['stop_reason'] = 'teacher_quota'
        else:
            summary['reserved_attempts'] += 1
        return summary

    monkeypatch.setattr(SharedTeacherBudget, 'summary', changed_summary)
    code, report = recover(budget.path, capsys)
    assert code != 0 and report['status'] == 'error'
    assert report['recovery_row_durable'] is True
    assert 'already durable' in report['note']
    assert report['rows_after'] == report['rows_before'] + 1
    assert 'PRIVATE_EXTERNAL_EXCEPTION' not in json.dumps(report)
    appended = budget.path.read_bytes()[len(before):]
    assert len(appended.splitlines()) == 1
    assert json.loads(appended)['event'] == 'recovered'
