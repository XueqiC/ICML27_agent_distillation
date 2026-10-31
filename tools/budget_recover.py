"""Clear an explicitly expected teacher halt without refunding shared usage."""
import argparse
import hashlib
import json
from pathlib import Path

from repair.takeover import SharedTeacherBudget
from sed.provenance import utc_now
from sed.storage import locked


_STATE_FIELDS = ('caps', 'tokens_or_unsettled_bounds', 'reported_usage',
                 'unknown_quantities', 'reserved_attempts')


class _Parser(argparse.ArgumentParser):
    def error(self, message):
        raise ValueError(message)


def _recover(args, report):
    if len(args.reason) > 200:
        raise ValueError('--reason must be at most 200 characters')
    if any(char in args.reason for char in '\n\r\v\f\x1c\x1d\x1e\x85\u2028\u2029'):
        raise ValueError('--reason must not contain newlines')

    path = Path(args.ledger)
    report.update(ledger=str(path), dry_run=args.dry_run, recovery_row_durable=False)
    with locked(str(path) + '.accounting.lock'):
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
        if not args.dry_run:
            budget.ledger.append({
                'event': 'recovered', 'reason': args.reason,
                'previous_stop_reason': previous, 'timestamp': utc_now(),
                'through_rows': len(rows),
                'ledger_prefix_sha256': hashlib.sha256(prefix).hexdigest(),
                **copied, 'stop_reason': None,
            })
            report.update(rows_after=len(rows) + 1, recovery_row_durable=True)

    if not args.dry_run:
        # The public API takes its own accounting lock. A concurrent writer can
        # cause verification to fail; the durable recovery must still be reported.
        summary = SharedTeacherBudget(path, **caps).summary()
        if summary['stop_reason'] is not None or any(summary[key] != copied[key] for key in _STATE_FIELDS):
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
        print(json.dumps(report, sort_keys=True))
        return 1
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
