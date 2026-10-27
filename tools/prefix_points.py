"""Build dp/1 diagnostic prefix specs from frozen TRAIN coverage, offline.

Usage: python -m tools.prefix_points build --coverage DIR --splits FILE
       --set U_dev --output spec.json

Writes spec.json, spec.summary.json and spec.summary.md with exclusive create.
Success strata count all verified coverage attempts (greedy and sampled). The
fixed selection order is greedy, sampled seed 0, 1, 2. No teacher or environment
calls, training exports, or state restoration are performed.
"""
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path

from repair.artifacts import read_json
from repair.prefixes import (RULE_VERSION, ATTEMPT_FIELDS, checked_episode, coverage_rows,
                             coverage_stratum, decision_point, load_original)
from sed.config import load_config
from sed.manifest import authorize_task, load_frozen, resolve_task
from sed.provenance import safe_config, versions
from sed.storage import atomic_write, digest, file_hash, write_json
from tools.cg_probe import markdown_table


ORDER = [('greedy', None), ('sampled', 0), ('sampled', 1), ('sampled', 2)]


def checked_set(path, name, frozen):
    document = read_json(path)
    if document.get('content_sha256') != digest({k: v for k, v in document.items() if k != 'content_sha256'}):
        raise ValueError('Method split content hash differs')
    listing = document['sets'][name]
    ids, groups = listing['task_ids'], listing['group_ids']
    if (listing.get('list_sha256') != digest(ids)
            or ('group_list_sha256' in listing and listing['group_list_sha256'] != digest(groups))
            or len(ids) != len(set(ids)) or len(groups) != len(set(groups)) or len(ids) != len(groups)
            or set(groups) != {Path(t).parent.as_posix() for t in ids}
            or listing.get('purpose') == 'evaluation_only'):
        raise ValueError('Method split task/group listing differs or is evaluation-only')
    if name == 'U_dev' and (listing.get('purpose') != 'diagnosis_only' or listing.get('training_eligible') is not False):
        raise ValueError('U_dev must remain diagnosis-only')
    if frozen is not None:
        manifest, freeze = frozen
        for value in (document, listing):
            if value.get('manifest_sha256') != digest(manifest) or value.get('split_freeze_sha256') != digest(freeze):
                raise ValueError('Method splits differ from frozen manifest')
        for tid in ids:
            task = resolve_task(manifest, freeze, tid)
            authorize_task(task, 'generation')
            if task['source_split'] != 'train':
                raise ValueError('Decision points require TRAIN tasks')
    return ids


def build(args):
    config = load_config(args.config)
    frozen = None if args.dry_run else load_frozen(config['paths']['manifest'], config['paths']['split_freeze'])
    ids = checked_set(args.splits, args.task_set, frozen)
    root, output = Path(args.coverage).resolve(), Path(args.output)
    summary_path, table_path = output.with_suffix('.summary.json'), output.with_suffix('.summary.md')
    if any(p.exists() for p in (output, summary_path, table_path)):
        raise FileExistsError('Prefix spec and summary outputs must be new')
    coverage = coverage_rows(root / 'failures.json', root)
    rows, hashes = coverage
    if not (root / 'episode_index.jsonl').is_file():
        raise ValueError('Building strata requires the full coverage episode index')
    hashes = dict(hashes)
    by_task, allowed = defaultdict(list), set(ids)
    for row in rows:
        if row['task_id'] in allowed:
            by_task[row['task_id']].append(row)
    spec, excluded = [], []
    for tid in ids:
        attempts = by_task[tid]
        if not attempts:
            excluded.append(dict(task_id=tid, reason='no_coverage_rollout'))
            continue
        cells = [(r['decoding'], r['sampling_seed']) for r in attempts]
        if len(cells) != len(set(cells)) or any(c not in ORDER for c in cells):
            raise ValueError('Coverage must follow greedy, sampled seeds 0, 1, 2 without duplicates')
        checked = []
        for row in sorted(attempts, key=lambda r: ORDER.index((r['decoding'], r['sampling_seed']))):
            ep, task, source = checked_episode(row, root, config, frozen=frozen)
            hashes.update(source['evidence_sha256'])
            checked.append((row, ep, task))
        k, n = sum(ep['success'] for _, ep, _ in checked), len(checked)
        stratum = coverage_stratum(k, n)
        if stratum == 'excluded':
            excluded.append(dict(task_id=tid, reason='no_failed_rollout', coverage_successes=k, coverage_rollouts=n))
            continue
        row, ep, task = next(item for item in checked if item[1]['success'] is False)
        point = decision_point(ep['initial'], ep['steps'])
        group = row['group_id'].removeprefix('train/')
        entry = dict(case_id=group + '/' + RULE_VERSION, group=group, task_id=tid,
            task_type=task.get('task_type', row.get('task_type', group.split('-')[0])),
            source=dict(failures_json=str(root / 'failures.json'), coverage_root=str(root), episode_dir=row['episode_dir']),
            **{key: row[key] for key in ATTEMPT_FIELDS}, **point,
            coverage_successes=k, coverage_rollouts=n, stratum=stratum)
        # A bad first prefix is an error; never silently choose a later rollout.
        load_original(entry, config, dry_run=args.dry_run, _coverage=coverage, _frozen=frozen)
        spec.append(entry)
    for path in (args.splits, args.config, __file__, 'repair/prefixes.py', 'repair/facts.py'):
        hashes[str(Path(path).resolve())] = file_hash(path)
    if frozen is not None:
        for key in ('manifest', 'split_freeze'):
            path = Path(config['paths'][key]).resolve()
            hashes[str(path)] = file_hash(path)
    counts = Counter((e['decision_point'], e['stratum'], e['task_type']) for e in spec)
    table = [dict(decision_point=label, stratum=stratum, task_type=kind, count=count)
             for (label, stratum, kind), count in sorted(counts.items())]
    summary = dict(schema_version=1, decision_rule_version=RULE_VERSION, task_set=args.task_set,
        dry_run=args.dry_run, spec_sha256=digest(spec), input_sha256=hashes,
        config=safe_config(config), dependencies=versions(),
        selection_order=[dict(decoding=d, sampling_seed=s) for d, s in ORDER],
        stratum_denominator='all verified coverage rollouts for the task, including greedy',
        unique_instances=len(spec), task_groups=len({e['group'] for e in spec}),
        counts=table, excluded=excluded, new_teacher_calls=0, new_environment_steps=0, training_exposure=0,
        interpretation='Frozen TRAIN diagnostic prefixes only; synthetic tests establish no scientific finding.')
    write_json(output, spec, exclusive=True)
    write_json(summary_path, summary, exclusive=True)
    text = '# dp/1 prefix points\n\n' + summary['interpretation'] + '\n\n'
    text += 'Strata count all verified attempts, including greedy.\n\n'
    text += markdown_table(table, ['decision_point', 'stratum', 'task_type', 'count']) + '\n'
    atomic_write(table_path, text.encode(), exclusive=True)
    print(json.dumps(dict(spec=str(output), summary=str(summary_path), table=str(table_path), cases=len(spec))))
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    command = sub.add_parser('build')
    command.add_argument('--coverage', required=True)
    command.add_argument('--splits', default='data/manifest/method_v1_splits.json')
    command.add_argument('--set', dest='task_set', default='U_dev', choices=['U_dev', 'V2_pool'])
    command.add_argument('--output', required=True)
    command.add_argument('--config', default='configs/base_e4b.yaml')
    command.add_argument('--dry-run', action='store_true', help='Synthetic fixtures only: bypass frozen-manifest resolution')
    return build(parser.parse_args(argv))


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(json.dumps(dict(status='error', error_type=type(exc).__name__)))
        raise SystemExit(1) from None
