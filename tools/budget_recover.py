"""Clear an explicitly expected teacher halt without refunding shared usage."""
import argparse
from contextlib import contextmanager
import fcntl
import hashlib
import json
from pathlib import Path

from repair.ledger import teacher_token_usage
from repair.takeover import SharedTeacherBudget
from sed.provenance import utc_now
from sed.storage import canonical_bytes, locked


_STATE_FIELDS = ('caps', 'tokens_or_unsettled_bounds', 'reported_usage',
                 'unknown_quantities', 'reserved_attempts')


class _Parser(argparse.ArgumentParser):
    def error(self, message):
        raise ValueError(message)


@contextmanager
def _accounting_lock(path, read_only):
    lock_path = str(path) + '.accounting.lock'
    if read_only:
        # SharedTeacherBudget creates this lock. A dry run must not create or
        # open any run artifact for writing, including the lock file itself.
        with open(lock_path, 'rb') as handle:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            yield
    else:
        with locked(lock_path):
            yield


def _quota_settlements(request_ids, budget, rows, reason):
    if len(set(request_ids)) != len(request_ids):
        raise ValueError('Duplicate --acknowledge-quota-rejected ID')
    evidence = []
    settled = {row['request_id'] for row in rows if row['event'] == 'settlement'}
    for request_id in request_ids:
        reservations = [row for row in rows if row['event'] == 'reservation'
                        and row['request_id'] == request_id]
        if len(reservations) != 1:
            raise ValueError('Acknowledged ID must have exactly one reservation')
        if request_id in settled:
            raise ValueError('Acknowledged ID is already settled')
        receipt_path = reservations[0].get('usage_ledger')
        if not isinstance(receipt_path, str) or not receipt_path or not Path(receipt_path).is_file():
            raise ValueError('Acknowledged reservation usage ledger is missing')
        content = Path(receipt_path).read_bytes()
        if not content or not content.endswith(b'\n'):
            raise ValueError('Damaged usage ledger')
        try:
            receipts = [json.loads(line) for line in content.splitlines()]
        except (ValueError, UnicodeError):
            raise ValueError('Invalid usage ledger') from None
        if any(not isinstance(row, dict) for row in receipts):
            raise ValueError('Invalid usage ledger')
        receipts = [row for row in receipts if row.get('call_id') == request_id]
        # Reject contradictory usage anywhere in this call's history, including
        # a completed/rejected receipt preceding the final quota error.
        if any(row.get('usage') is not None for row in receipts):
            raise ValueError('Acknowledged call has a receipt with usage')
        if (not receipts or receipts[-1].get('status') != 'failed'
                or receipts[-1].get('error_type') != 'TeacherQuotaError'
                or 'usage' not in receipts[-1]):
            raise ValueError('Acknowledged call must end with TeacherQuotaError and null usage')
        evidence.append(dict(request_id=request_id, usage_ledger=receipt_path,
                             usage_ledger_sha256=hashlib.sha256(content).hexdigest()))
    reservations = {row['request_id'] for row in rows if row['event'] == 'reservation'}
    if reservations - settled - set(request_ids):
        raise ValueError('Other unsettled reservations must also be acknowledged')

    # GlobalTeacherBudget.settle requires numeric input/output usage to refund
    # bounds. Reuse its quantity schema/counting, but retain every bound here.
    usage = teacher_token_usage({})
    settlements = []
    for receipt in evidence:
        for key in usage:
            budget.unknown_quantities[key] = budget.unknown_quantities.get(key, 0) + 1
        settlements.append({
            'event': 'settlement', **receipt, 'timestamp': utc_now(),
            'acknowledged': 'teacher_quota_rejection', 'reason': reason,
            'usage': dict(usage), **budget.summary(),
        })
    return settlements


def _recover(args, report):
    if len(args.reason) > 200:
        raise ValueError('--reason must be at most 200 characters')
    if any(char in args.reason for char in '\n\r\v\f\x1c\x1d\x1e\x85\u2028\u2029'):
        raise ValueError('--reason must not contain newlines')
    acknowledge = args.acknowledge_quota_rejected
    if acknowledge and args.expect_stop != 'teacher_quota':
        raise ValueError('--acknowledge-quota-rejected requires --expect-stop teacher_quota')

    path = Path(args.ledger)
    report.update(ledger=str(path), dry_run=args.dry_run, recovery_row_durable=False)
    with _accounting_lock(path, read_only=args.dry_run and bool(acknowledge)):
        if not path.exists():
            raise ValueError('Ledger does not exist; recovery requires an existing ledger')
        prefix = path.read_bytes()
        if not prefix:
            raise ValueError('Ledger is empty; recovery requires an existing checkpoint')

        # __init__ would reacquire the non-reentrant accounting lock. Use the
        # same locked loader as reserve(), and let it reject a torn tail before
        # attempting to decode that tail to obtain the caps.
        shared = object.__new__(SharedTeacherBudget)
        shared.path = path
        shared.caps = json.loads(prefix.splitlines()[-1])['caps'] if prefix.endswith(b'\n') else {}
        budget, rows = shared._load()
        caps = shared.caps
        # Preserve the constructor's cap validation when bypassing __init__.
        if (set(caps) != {'input_tokens', 'output_tokens'}
                or any(type(value) is not int or value < 0 for value in caps.values())):
            raise ValueError('Token caps must be nonnegative integers')

        state = rows[-1]
        previous = state['stop_reason']
        if previous != args.expect_stop:
            raise ValueError('Ledger stop_reason does not match --expect-stop; recovery refused')
        copied = {key: state[key] for key in _STATE_FIELDS}
        reservations = {row['request_id'] for row in rows if row['event'] == 'reservation'}
        settled = {row['request_id'] for row in rows if row['event'] == 'settlement'}
        report.update(rows_before=len(rows), rows_after=len(rows),
                      previous_stop_reason=previous, **copied,
                      unsettled_attempts=len(reservations - settled))
        settlements = []
        if acknowledge:
            before = {**copied, 'stop_reason': previous}
            settlements = _quota_settlements(acknowledge, budget, rows, args.reason)
            after = budget.summary()
            copied = {key: after[key] for key in _STATE_FIELDS}
            report.update(acknowledged_ids=acknowledge, settlement_rows_durable=[],
                          unsettled_attempts_before=len(reservations - settled),
                          unsettled_attempts_after=0, state_before=before,
                          state_after={**copied, 'stop_reason': None})
        if not args.dry_run or acknowledge:
            prefix_hash = hashlib.sha256(prefix)
            for row in settlements:
                prefix_hash.update(canonical_bytes(row) + b'\n')
            recovered = {
                'event': 'recovered', 'reason': args.reason,
                'previous_stop_reason': previous, 'timestamp': utc_now(),
                'through_rows': len(rows) + len(settlements),
                'ledger_prefix_sha256': prefix_hash.hexdigest(),
                **copied, 'stop_reason': None,
            }
            if acknowledge and args.dry_run:
                report['rows_to_append'] = [*settlements, recovered]
            if not args.dry_run:
                for row in settlements:
                    budget.ledger.append(row)
                    report['settlement_rows_durable'].append(row['request_id'])
                    report['rows_after'] += 1
                budget.ledger.append(recovered)
                report.update(rows_after=len(rows) + len(settlements) + 1, recovery_row_durable=True)

    if not args.dry_run:
        # The public API takes its own accounting lock. A concurrent writer can
        # cause verification to fail; the durable recovery must still be reported.
        summary = SharedTeacherBudget(path, **caps).summary()
        if (summary['stop_reason'] is not None
                or any(summary[key] != copied[key] for key in _STATE_FIELDS)
                or acknowledge and summary['unsettled_attempts'] != 0):
            raise ValueError('Post-append verification failed; recovery row is already durable')
    report['status'] = 'ok'


def main(argv=None):
    report = {}
    try:
        parser = _Parser(description=__doc__)
        parser.add_argument('--ledger', required=True)
        parser.add_argument('--expect-stop', required=True, choices=('teacher_quota', 'teacher_halted'))
        parser.add_argument('--reason', required=True)
        parser.add_argument('--dry-run', action='store_true')
        parser.add_argument('--acknowledge-quota-rejected', action='append', default=[],
                            metavar='REQUEST_ID', help='Resolve a quota-rejected reservation without refunding its bound')
        args = parser.parse_args(argv)
        _recover(args, report)
    except Exception as exc:
        report.update(status='error', error_type=type(exc).__name__)
        # Only our own and the accounting API's plain ValueErrors have safe
        # messages. Never expose arbitrary exception text or ledger contents.
        if type(exc) is ValueError:
            report['message'] = str(exc)
        if report.get('recovery_row_durable'):
            report['note'] = 'Recovery row is already durable; post-append verification failed.'
        elif report.get('settlement_rows_durable'):
            report['note'] = 'Settlement rows are already durable; recovery did not complete.'
        print(json.dumps(report, sort_keys=True))
        return 1
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
