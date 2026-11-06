"""Offline maintenance of synthetic SharedTeacherBudget ledgers only."""
from datetime import datetime, timezone
import hashlib
import json

import pytest

from repair.runtime import CaseStopped
from repair.takeover import SharedTeacherBudget
from sed.storage import canonical_bytes, JsonlLedger
from tools import budget_recover


STATE_FIELDS = ('caps', 'tokens_or_unsettled_bounds', 'reported_usage',
                'unknown_quantities', 'reserved_attempts')


@pytest.fixture
def budget(tmp_path):
    return SharedTeacherBudget(tmp_path / 'budget.jsonl', input_tokens=1000, output_tokens=500)


def recover(path, capsys, *, expect='teacher_quota', reason='quota_topped_up', dry_run=False,
            acknowledge=()):
    args = ['--ledger', str(path), '--expect-stop', expect, '--reason', reason]
    if dry_run:
        args.append('--dry-run')
    for request_id in acknowledge:
        args.extend(['--acknowledge-quota-rejected', request_id])
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


def quota_budget(tmp_path, request_ids=('quota',)):
    from tools.v0_round import CAPS
    budget = SharedTeacherBudget(tmp_path / 'teacher_budget.jsonl', **CAPS)
    settle(budget, 'reported')  # Preserve both known counts and unknown metadata.
    for request_id in request_ids:
        usage = JsonlLedger(tmp_path / request_id / 'usage.jsonl')
        budget.reserve(request_id, 80, 30, usage_ledger=str(usage.path))
        usage.append(dict(call_id=request_id, status='started', usage=None))
        usage.append(dict(call_id=request_id, status='failed', error_type='TeacherQuotaError',
                          usage=None, http_status=None))
        budget.settle(request_id, None)  # The public API intentionally leaves this open.
    budget.halt('teacher_quota')
    budget.halt('teacher_quota')
    return budget


@pytest.mark.parametrize('request_ids', [('quota',), ('quota', 'second')])
def test_acknowledge_quota_preserves_bounds_and_resumes_v0(tmp_path, capsys, request_ids):
    from repair.ledger import teacher_token_usage
    from tools.v0_round import budget_for
    budget = quota_budget(tmp_path, request_ids)
    before = budget.path.read_bytes()
    state = budget.summary()
    reason = 'credits restored; provider rejected call, bound kept'
    code, report = recover(budget.path, capsys, acknowledge=request_ids, reason=reason)
    assert code == 0 and report['status'] == 'ok'
    assert report['acknowledged_ids'] == report['settlement_rows_durable'] == list(request_ids)
    assert report['unsettled_attempts_before'] == len(request_ids)
    assert report['unsettled_attempts_after'] == 0
    assert report['rows_after'] == report['rows_before'] + len(request_ids) + 1
    assert report['recovery_row_durable'] is True
    after = budget.path.read_bytes()
    assert after.startswith(before)
    appended = [json.loads(line) for line in after[len(before):].splitlines()]
    unknown = dict(state['unknown_quantities'])
    prefix = before
    for request_id, row in zip(request_ids, appended[:-1], strict=True):
        usage = tmp_path / request_id / 'usage.jsonl'
        for key in teacher_token_usage({}):
            unknown[key] = unknown.get(key, 0) + 1
        assert row == {
            'event': 'settlement', 'request_id': request_id, 'timestamp': row['timestamp'],
            'acknowledged': 'teacher_quota_rejection', 'reason': reason,
            'usage_ledger': str(usage), 'usage_ledger_sha256': hashlib.sha256(usage.read_bytes()).hexdigest(),
            'usage': {key: None for key in state['reported_usage']},
            **{key: state[key] for key in STATE_FIELDS}, 'unknown_quantities': dict(unknown),
            'stop_reason': 'teacher_quota',
        }
        assert datetime.fromisoformat(row['timestamp']).utcoffset() == timezone.utc.utcoffset(None)
        prefix += canonical_bytes(row) + b'\n'
        # Each intermediate row must be independently loadable as a checkpoint.
        checkpoint = tmp_path / f'{request_id}-checkpoint.jsonl'
        checkpoint.write_bytes(prefix)
        loaded = SharedTeacherBudget(checkpoint, **budget.caps).summary()
        assert loaded['unknown_quantities'] == unknown
        assert loaded['tokens_or_unsettled_bounds'] == state['tokens_or_unsettled_bounds']
        assert loaded['reported_usage'] == state['reported_usage']
    assert appended[-1] == {
        'event': 'recovered', 'reason': reason, 'previous_stop_reason': 'teacher_quota',
        'timestamp': appended[-1]['timestamp'], 'through_rows': len(prefix.splitlines()),
        'ledger_prefix_sha256': hashlib.sha256(prefix).hexdigest(),
        **{key: state[key] for key in STATE_FIELDS}, 'unknown_quantities': unknown, 'stop_reason': None,
    }
    summary = SharedTeacherBudget(budget.path, **budget.caps).summary()
    assert summary['stop_reason'] is None and summary['unsettled_attempts'] == 0
    assert report['state_before'] == {key: state[key] for key in (*STATE_FIELDS, 'stop_reason')}
    assert report['state_after'] == {key: summary[key] for key in (*STATE_FIELDS, 'stop_reason')}
    for key in STATE_FIELDS:
        assert summary[key] == (unknown if key == 'unknown_quantities' else state[key])
    assert budget_for(tmp_path).summary() == summary
    assert budget.path.read_bytes() == after  # No v0_unsettled_teacher_attempt halt.


@pytest.mark.parametrize('failure,message', [
    ('known_usage', 'receipt with usage'),
    ('completed_usage', 'receipt with usage'),
    ('success_usage', 'receipt with usage'),
    ('rejected_usage', 'receipt with usage'),
    ('other_error', 'must end with TeacherQuotaError'),
    ('not_reserved', 'exactly one reservation'),
    ('duplicate_reservation', 'exactly one reservation'),
    ('already_settled', 'already settled'),
    ('missing_ledger', 'usage ledger is missing'),
    ('missing_receipt', 'must end with TeacherQuotaError'),
    ('missing_usage', 'must end with TeacherQuotaError'),
    ('unfinished', 'must end with TeacherQuotaError'),
    ('damaged_receipts', 'Damaged usage ledger'),
    ('other_unsettled', 'Other unsettled reservations'),
    ('duplicate_flag', 'Duplicate --acknowledge'),
    ('wrong_expect', 'requires --expect-stop teacher_quota'),
    ('stop_mismatch', 'does not match --expect-stop'),
])
@pytest.mark.parametrize('dry_run', [False, True])
def test_quota_acknowledgement_refuses_without_writes(tmp_path, capsys, failure, message, dry_run):
    budget = quota_budget(tmp_path, ('quota', 'other') if failure == 'other_unsettled' else ('quota',))
    usage = tmp_path / 'quota/usage.jsonl'
    receipts = [json.loads(line) for line in usage.read_bytes().splitlines()]
    ids, expect = ['quota'], 'teacher_quota'
    if failure == 'known_usage':
        receipts[-1]['usage'] = {'input_tokens': 0, 'output_tokens': 0}
    elif failure in ('completed_usage', 'success_usage', 'rejected_usage'):
        receipts.insert(1, dict(call_id='quota', status=failure.removesuffix('_usage'),
                                usage={'input_tokens': 1}))
    elif failure == 'other_error':
        receipts[-1]['error_type'] = 'TeacherCallError'
    elif failure == 'not_reserved':
        ids = ['absent']
    elif failure == 'duplicate_reservation':
        rows = [json.loads(line) for line in budget.path.read_bytes().splitlines()]
        reservation = next(row for row in rows if row.get('request_id') == 'quota')
        JsonlLedger(budget.path).append({**reservation, **budget.summary()})
    elif failure == 'already_settled':
        budget.settle('quota', {'input_tokens': 1, 'output_tokens': 1})
    elif failure == 'missing_receipt':
        receipts = [dict(call_id='other', status='failed', error_type='TeacherQuotaError', usage=None)]
    elif failure == 'missing_usage':
        del receipts[-1]['usage']
    elif failure == 'unfinished':
        receipts.append(dict(call_id='quota', status='started', usage=None))
    elif failure == 'duplicate_flag':
        ids *= 2
    elif failure == 'wrong_expect':
        expect = 'teacher_halted'
        budget.halt(expect)
    elif failure == 'stop_mismatch':
        budget.halt('teacher_halted')
    usage.write_bytes(b''.join(canonical_bytes(row) + b'\n' for row in receipts))
    if failure == 'damaged_receipts':
        usage.write_bytes(usage.read_bytes()[:-1])
    if failure == 'missing_ledger':
        usage.unlink()
    before = budget.path.read_bytes()
    code, report = recover(budget.path, capsys, acknowledge=ids, expect=expect, dry_run=dry_run)
    assert code != 0 and report['status'] == 'error'
    assert report['error_type'] == 'ValueError' and message in report['message']
    assert budget.path.read_bytes() == before


def test_quota_dry_run_is_read_only_and_reports_exact_planned_rows(tmp_path, capsys, monkeypatch):
    import builtins
    import os
    budget = quota_budget(tmp_path)
    # Unrelated calls must neither supply evidence nor disqualify this call.
    JsonlLedger(tmp_path / 'quota/usage.jsonl').append(
        dict(call_id='unrelated', status='success', usage={'input_tokens': 3, 'output_tokens': 1}))
    before = {path: path.read_bytes() for path in tmp_path.rglob('*') if path.is_file()}
    original_open = builtins.open

    def read_only_open(file, mode='r', *args, **kwargs):
        assert not any(flag in mode for flag in 'wax+')
        return original_open(file, mode, *args, **kwargs)

    def forbidden(*args, **kwargs):
        raise AssertionError('Dry run attempted a write or created a lock')

    with monkeypatch.context() as patch:
        patch.setattr(builtins, 'open', read_only_open)
        patch.setattr(os, 'open', forbidden)  # Includes sed.storage.locked's O_CREAT.
        patch.setattr(JsonlLedger, 'append', forbidden)
        code, report = recover(budget.path, capsys, acknowledge=['quota'], dry_run=True)
    assert code == 0 and report['status'] == 'ok'
    assert report['acknowledged_ids'] == ['quota']
    assert report['unsettled_attempts_before'] == 1 and report['unsettled_attempts_after'] == 0
    assert report['rows_before'] == report['rows_after']
    assert report['settlement_rows_durable'] == [] and report['recovery_row_durable'] is False
    assert {path: path.read_bytes() for path in tmp_path.rglob('*') if path.is_file()} == before
    planned = report['rows_to_append']
    assert [row['event'] for row in planned] == ['settlement', 'recovered']
    # Publish the proposal only to a synthetic copy and exercise the real loader.
    copy = tmp_path / 'planned.jsonl'
    copy.write_bytes(before[budget.path] + b''.join(canonical_bytes(row) + b'\n' for row in planned))
    summary = SharedTeacherBudget(copy, **budget.caps).summary()
    assert summary['unsettled_attempts'] == 0
    assert report['state_after'] == {key: summary[key] for key in (*STATE_FIELDS, 'stop_reason')}


def test_all_acknowledgements_validate_before_any_append(tmp_path, capsys):
    budget = quota_budget(tmp_path, ('quota', 'second'))
    (tmp_path / 'second/usage.jsonl').unlink()
    before = budget.path.read_bytes()
    code, report = recover(budget.path, capsys, acknowledge=['quota', 'second'])
    assert code != 0 and 'usage ledger is missing' in report['message']
    assert budget.path.read_bytes() == before


@pytest.mark.parametrize('failure', ['unsettled_attempts', 'tokens_or_unsettled_bounds', 'reported_usage'])
def test_quota_post_append_checks_unsettled_and_preserved_totals(tmp_path, capsys, monkeypatch, failure):
    budget = quota_budget(tmp_path)
    original = SharedTeacherBudget.summary

    def changed_summary(self):
        result = original(self)
        if failure == 'unsettled_attempts':
            result[failure] = 1
        else:
            result[failure]['input_tokens'] -= 1
        return result

    monkeypatch.setattr(SharedTeacherBudget, 'summary', changed_summary)
    code, report = recover(budget.path, capsys, acknowledge=['quota'])
    assert code != 0 and report['recovery_row_durable'] is True
    assert report['settlement_rows_durable'] == ['quota']
    assert 'already durable' in report['note']


@pytest.mark.parametrize('dry_run', [False, True])
def test_no_acknowledgement_retains_exact_legacy_report(tmp_path, capsys, dry_run):
    budget = quota_budget(tmp_path)
    state, before = budget.summary(), budget.path.read_bytes()
    code, report = recover(budget.path, capsys, dry_run=dry_run)
    assert code == 0
    count = len(before.splitlines())
    assert report == {
        'ledger': str(budget.path), 'dry_run': dry_run, 'recovery_row_durable': not dry_run,
        'rows_before': count, 'rows_after': count + int(not dry_run),
        'previous_stop_reason': 'teacher_quota', **{key: state[key] for key in STATE_FIELDS},
        'unsettled_attempts': 1, 'status': 'ok',
    }


def provider_budget(tmp_path, request_ids=('provider',), stop='review_transport_failed'):
    budget = SharedTeacherBudget(tmp_path / 'teacher_budget.jsonl', input_tokens=1000, output_tokens=500)
    settle(budget, 'reported')
    for request_id in request_ids:
        usage = JsonlLedger(tmp_path / request_id / 'usage.jsonl')
        budget.reserve(request_id, 80, 30, usage_ledger=str(usage.path))
        usage.append(dict(call_id=request_id, status='started', usage=None))
        usage.append(dict(call_id=request_id, status='failed', error_type='InternalServerError',
                          http_status=503, usage=None))
    budget.halt(stop)
    return budget


def recover_provider(path, capsys, *, request_ids=('provider',), expect='review_transport_failed',
                     dry_run=False):
    args = ['--ledger', str(path), '--expect-stop', expect,
            '--reason', 'provider back; bound kept']
    for request_id in request_ids:
        args.extend(['--acknowledge-provider-error', request_id])
    if dry_run:
        args.append('--dry-run')
    code = budget_recover.main(args)
    captured = capsys.readouterr()
    assert not captured.err and len(captured.out.splitlines()) == 1
    return code, json.loads(captured.out)


@pytest.mark.parametrize('stop', ['review_transport_failed', 'label_transport_failed', 'phase_2_transport_failed'])
@pytest.mark.parametrize('request_ids', [('provider',), ('provider', 'second')])
@pytest.mark.parametrize('dry_run', [False, True])
def test_provider_recovery_preserves_bounds_and_dry_run_is_read_only(
        tmp_path, capsys, monkeypatch, stop, request_ids, dry_run):
    import builtins
    import io
    import os
    budget = provider_budget(tmp_path, request_ids, stop)
    # Only this call's receipts determine whether it may be acknowledged.
    JsonlLedger(tmp_path / 'provider/usage.jsonl').append(
        dict(call_id='unrelated', status='success', usage={'input_tokens': 1}))
    state = budget.summary()
    before = {path: path.read_bytes() for path in tmp_path.rglob('*') if path.is_file()}
    original_open = builtins.open

    def read_only_open(file, mode='r', *args, **kwargs):
        assert not any(flag in mode for flag in 'wax+')
        return original_open(file, mode, *args, **kwargs)

    def forbidden(*args, **kwargs):
        raise AssertionError('Dry run attempted a write or created a lock')

    with monkeypatch.context() as patch:
        if dry_run:
            patch.setattr(builtins, 'open', read_only_open)
            patch.setattr(io, 'open', read_only_open)
            patch.setattr(os, 'open', forbidden)
            patch.setattr(JsonlLedger, 'append', forbidden)
        code, report = recover_provider(budget.path, capsys, request_ids=request_ids,
                                        expect=stop, dry_run=dry_run)
    assert code == 0 and report['status'] == 'ok'
    assert report['acknowledged_ids'] == list(request_ids)
    assert report['unsettled_attempts_before'] == len(request_ids)
    assert report['unsettled_attempts_after'] == 0
    assert report['rows_before'] == len(before[budget.path].splitlines())
    assert report['rows_after'] == report['rows_before'] + (0 if dry_run else len(request_ids) + 1)
    assert report['recovery_row_durable'] is (not dry_run)
    assert report['settlement_rows_durable'] == ([] if dry_run else list(request_ids))
    if dry_run:
        assert {path: path.read_bytes() for path in tmp_path.rglob('*') if path.is_file()} == before
        appended = report['rows_to_append']
    else:
        assert budget.path.read_bytes().startswith(before[budget.path])
        appended = [json.loads(line) for line in budget.path.read_bytes()[len(before[budget.path]):].splitlines()]
    prefix, unknown = before[budget.path], dict(state['unknown_quantities'])
    for request_id, row in zip(request_ids, appended[:-1], strict=True):
        usage = tmp_path / request_id / 'usage.jsonl'
        unknown = {key: unknown.get(key, 0) + 1 for key in state['reported_usage']}
        assert row == {
            'event': 'settlement', 'request_id': request_id, 'timestamp': row['timestamp'],
            'acknowledged': 'provider_5xx_rejection', 'reason': 'provider back; bound kept',
            'usage_ledger': str(usage), 'usage_ledger_sha256': hashlib.sha256(before[usage]).hexdigest(),
            'usage': {key: None for key in state['reported_usage']},
            **{key: state[key] for key in STATE_FIELDS}, 'unknown_quantities': unknown,
            'stop_reason': stop,
        }
        prefix += canonical_bytes(row) + b'\n'
    assert appended[-1] == {
        'event': 'recovered', 'reason': 'provider back; bound kept', 'previous_stop_reason': stop,
        'timestamp': appended[-1]['timestamp'], 'through_rows': len(prefix.splitlines()),
        'ledger_prefix_sha256': hashlib.sha256(prefix).hexdigest(),
        **{key: state[key] for key in STATE_FIELDS}, 'unknown_quantities': unknown, 'stop_reason': None,
    }
    # Verify both the appended result and the dry-run proposal with the real loader.
    copy = tmp_path / 'recovered-copy.jsonl'
    copy.write_bytes(prefix + canonical_bytes(appended[-1]) + b'\n')
    summary = SharedTeacherBudget(copy if dry_run else budget.path, **budget.caps).summary()
    assert summary['unsettled_attempts'] == 0 and summary['stop_reason'] is None
    assert report['state_before'] == {key: state[key] for key in (*STATE_FIELDS, 'stop_reason')}
    assert report['state_after'] == {key: summary[key] for key in (*STATE_FIELDS, 'stop_reason')}


@pytest.mark.parametrize('error_type,http_status', [
    ('InternalServerError', None), ('ServiceUnavailableError', None),
    ('BadGatewayError', None), ('GatewayTimeoutError', None),
    ('APIStatusError', 500), ('APIStatusError', 502), ('APIStatusError', 503), ('APIStatusError', 504),
])
def test_provider_error_accepts_named_error_or_http_status(tmp_path, capsys, error_type, http_status):
    budget = provider_budget(tmp_path)
    JsonlLedger(tmp_path / 'provider/usage.jsonl').append(
        dict(call_id='provider', status='failed', error_type=error_type, http_status=http_status, usage=None))
    code, report = recover_provider(budget.path, capsys, dry_run=True)
    assert code == 0 and report['rows_to_append'][0]['acknowledged'] == 'provider_5xx_rejection'


@pytest.mark.parametrize('failure,message', [
    ('4xx', 'provider 5xx failure'),
    ('usage', 'receipt with usage'),
    ('earlier_usage', 'receipt with usage'),
    ('missing_usage', 'null usage'),
    ('unfinished', 'provider 5xx failure'),
    ('other_unsettled', 'Other unsettled reservations'),
    ('duplicate_flag', 'Duplicate --acknowledge-provider-error'),
    ('teacher_quota', 'requires --expect-stop <phase>_transport_failed'),
    ('teacher_halted', 'requires --expect-stop <phase>_transport_failed'),
    ('stop_mismatch', 'does not match --expect-stop'),
])
@pytest.mark.parametrize('dry_run', [False, True])
def test_provider_acknowledgement_refuses_without_writes(tmp_path, capsys, failure, message, dry_run):
    ids, expect = ['provider'], 'review_transport_failed'
    budget = provider_budget(tmp_path, ('provider', 'other') if failure == 'other_unsettled' else ids)
    usage = tmp_path / 'provider/usage.jsonl'
    receipts = [json.loads(line) for line in usage.read_bytes().splitlines()]
    if failure == '4xx':
        receipts[-1].update(error_type='BadRequestError', http_status=400)
    elif failure == 'usage':
        receipts[-1]['usage'] = {}
    elif failure == 'earlier_usage':
        receipts.insert(1, dict(call_id='provider', status='success', usage={'input_tokens': 1}))
    elif failure == 'missing_usage':
        del receipts[-1]['usage']
    elif failure == 'unfinished':
        receipts[-1]['status'] = 'started'
    elif failure == 'duplicate_flag':
        ids *= 2
    elif failure in ('teacher_quota', 'teacher_halted'):
        expect = failure
        budget.halt(expect)
    elif failure == 'stop_mismatch':
        budget.halt('label_transport_failed')
    usage.write_bytes(b''.join(canonical_bytes(row) + b'\n' for row in receipts))
    before = {path: path.read_bytes() for path in tmp_path.rglob('*') if path.is_file()}
    code, report = recover_provider(budget.path, capsys, request_ids=ids, expect=expect, dry_run=dry_run)
    assert code != 0 and report['status'] == 'error'
    assert report['error_type'] == 'ValueError' and message in report['message']
    assert {path: path.read_bytes() for path in tmp_path.rglob('*') if path.is_file()} == before


@pytest.mark.parametrize('stop', [
    '_transport_failed', 'Review_transport_failed', 'review-2_transport_failed',
    'review_transport_failed_extra', 'review_transport_failed\n', '',
])
def test_invalid_transport_stop_pattern_refused_without_writes(tmp_path, capsys, stop):
    budget = provider_budget(tmp_path, stop=stop)
    before = budget.path.read_bytes()
    code, report = recover_provider(budget.path, capsys, expect=stop)
    assert code != 0 and '--expect-stop must be' in report['message']
    assert budget.path.read_bytes() == before
