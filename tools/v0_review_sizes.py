"""Read-only comparison of V0 review request bytes; no environment/model calls.

Use recorded public trajectories in requests.json when available, otherwise the
selected failures in attempt/records.json. The old encoder/template is loaded
from the specified local git revision, without creating a checkout or staging.
Print aggregates only; never write to either input directory.
"""
import argparse
import json
from pathlib import Path
import statistics
import subprocess
import sys
from types import ModuleType

from repair.records import TrajectoryRecord
from repair.v0_execution import selected_failures
from repair.v0_review import WholePrompt, public_execution, review_prompt
from sed.storage import canonical_bytes, file_hash


def recorded_trajectory(public):
    """Rebuild only the public inputs consumed by TransformFactTable.

    Old rows recorded pre-action command lists. A row's post-action list is the
    next row's pre-action list; the last post-action list was never serialized
    and is not consumed by either review encoder.
    """
    rows = public['steps']
    initial = dict(visible=dict(observation=public['initial_observation'],
        admissible_commands=rows[0]['admissible_commands'] if rows else []))
    steps = [dict(action_raw=row['reply'], action_parsed=row['action'],
        action_outcome=row['action_outcome'], visible=None if row['observation'] is None else
        dict(observation=row['observation'],
             admissible_commands=rows[i + 1]['admissible_commands'] if i + 1 < len(rows) else []))
        for i, row in enumerate(rows)]
    return TrajectoryRecord(task_id='recorded-public-review', initial=initial, steps=steps,
        success=public['final_outcome']['won'], termination_reason=public['final_outcome']['termination'])


def baseline_encoder(revision):
    source = subprocess.run(['git', 'show', f'{revision}:repair/v0_review.py'],
        cwd=Path(__file__).resolve().parents[1], check=True, capture_output=True, text=True).stdout
    module = ModuleType('_v0_review_size_baseline')
    sys.modules[module.__name__] = module  # Required for dataclass annotation lookup.
    exec(compile(source, f'{revision}:repair/v0_review.py', 'exec'), module.__dict__)
    return module


def compare(requests_path, attempts_path, *, baseline='6e33968', expected_failures=146):
    old = baseline_encoder(baseline)
    sizes = []
    if requests_path.is_file():
        source, kind = requests_path, 'recorded_public_trajectories'
        for row in json.loads(source.read_text()):
            envelope = row['request']
            payload = envelope['payload']
            original = old.WholePrompt(payload, envelope['retry_reason'])
            assert original.to_json() == envelope, 'Baseline template/envelope differs'

            def compact(public):
                trace = recorded_trajectory(public)
                assert old.public_execution(trace) == public, 'Recorded public trajectory differs on replay'
                return public_execution(trace)

            updated = dict(payload, trajectory=compact(payload['trajectory']), previous_rounds=[
                dict(round=r['round'], executions=[compact(e) for e in r['executions']])
                for r in payload['previous_rounds']])
            current = WholePrompt(updated, envelope['retry_reason'])
            old_bytes = len(canonical_bytes(original.messages()))
            assert old_bytes == row['request_bytes'], 'Recorded request byte count differs'
            sizes.append((old_bytes, len(canonical_bytes(current.messages()))))
    else:
        source, kind = attempts_path, 'attempt_records'
        attempts = json.loads(source.read_text())
        if any(row['heldout_probe'] for row in attempts):
            raise ValueError('V0 probes cannot be used for trajectory review')
        for row in selected_failures(attempts):
            trace = TrajectoryRecord.from_json(row['execution'])
            sizes.append(tuple(len(canonical_bytes(encoder(trace, row['prefix_length']).messages()))
                               for encoder in (old.review_prompt, review_prompt)))
    assert len(sizes) == expected_failures, f'Expected {expected_failures} failures, found {len(sizes)}'
    old_sizes, new_sizes = zip(*sizes)
    def summary(values):
        return dict(mean=statistics.mean(values), max=max(values), total=sum(values))
    root = Path(__file__).resolve().parents[1]
    return dict(source=str(source.resolve()), source_sha256=file_hash(source), source_kind=kind,
        baseline_revision=baseline, failures=len(sizes), teacher_calls=0, environment_calls=0,
        measurement='UTF-8 canonical JSON of system+user messages; first review round only',
        old_request_bytes=summary(old_sizes), new_request_bytes=summary(new_sizes),
        new_over_old=sum(new_sizes) / sum(old_sizes),
        max_per_request_new_over_old=max(new / old for old, new in sizes),
        code_sha256={name: file_hash(root / name) for name in ('repair/v0_review.py', 'tools/v0_review_sizes.py')})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--requests', type=Path, required=True)
    parser.add_argument('--attempts', type=Path, required=True)
    parser.add_argument('--baseline', default='6e33968')
    parser.add_argument('--expected-failures', type=int, default=146)
    args = parser.parse_args()
    print(json.dumps(compare(args.requests, args.attempts, baseline=args.baseline,
        expected_failures=args.expected_failures), indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
