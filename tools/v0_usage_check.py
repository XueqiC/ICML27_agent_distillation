"""Read-only reconciliation of V0 usage receipts and budget settlements.

Run with --review-dir ROUND/review --budget ROUND/teacher_budget.jsonl.
Count each transport call once, including rejected outputs. State links,
recovered request copies and cumulative budget snapshots are not new usage.
"""
import argparse
from collections import Counter
import json
from pathlib import Path

from teachers.openai_teacher import has_known_usage

TOKEN_KEYS = ('input_tokens', 'output_tokens', 'reasoning_tokens')
TERMINAL = {'success', 'rejected', 'failed', 'incomplete'}


def rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines()]


def totals(usages):
    return {key: sum(usage.get(key, 0) or 0 for usage in usages) for key in TOKEN_KEYS}


def audit_usage(review_dir, budget):
    files = sorted(Path(review_dir).rglob('usage.jsonl'))
    if not files:
        raise ValueError('No usage ledgers found')
    receipts, starts, statuses = {}, set(), Counter()
    duplicates = 0
    for path in files:
        for row in rows(path):
            statuses[row['status']] += 1
            if row['status'] == 'started':
                starts.add(row['call_id'])
            if row['status'] not in TERMINAL:
                continue
            call = row['call_id']
            if call in receipts:
                if any(receipts[call].get(k) != row.get(k) for k in
                       ('status', 'usage', 'reported_usage', 'error_type')):
                    raise ValueError('Conflicting terminal usage receipts')
                duplicates += 1
            receipts[call] = row
    events = rows(budget)
    reservations, settlements = {}, {}
    for row in events:
        if row['event'] == 'reservation':
            if row['request_id'] in reservations:
                raise ValueError('Duplicate budget reservation')
            reservations[row['request_id']] = row
        elif row['event'] == 'settlement':
            call = row['request_id']
            if call not in reservations or call in settlements:
                raise ValueError('Unreserved or duplicate budget settlement')
            settlements[call] = row['usage']
    if not (starts | receipts.keys()) <= reservations.keys():
        raise ValueError('Usage receipt lacks a budget reservation')
    for call, row in receipts.items():
        usage = row.get('usage') or row.get('reported_usage')
        if has_known_usage(usage):
            if call not in settlements or totals([usage]) != totals([settlements[call]]):
                raise ValueError('Usage receipt differs from budget settlement')
        elif call in settlements:
            raise ValueError('Settlement lacks a reported usage receipt')
    all_settled = totals(list(settlements.values()))
    if all_settled != {key: events[-1]['reported_usage'][key] for key in TOKEN_KEYS}:
        raise ValueError('Settlements differ from final cumulative budget snapshot')
    calls = starts | receipts.keys()
    selected = calls & settlements.keys()
    return dict(usage_files=len(files), attempts=len(calls), settled_attempts=len(selected),
        unsettled_attempts=len(calls - settlements.keys()), interrupted_attempts=len(starts - receipts.keys()),
        duplicate_receipts_ignored=duplicates, row_statuses=dict(statuses),
        unknown_usage_attempts=sum(not has_known_usage(r.get('usage')) for r in receipts.values()),
        settled_usage=totals([settlements[c] for c in selected]),
        all_budget_settled_usage=all_settled, budget_snapshot_matches=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--review-dir', required=True)
    parser.add_argument('--budget', required=True)
    args = parser.parse_args(argv)
    print(json.dumps(audit_usage(args.review_dir, args.budget), indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
