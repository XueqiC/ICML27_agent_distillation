"""Teacher-free, from-start V2 E-sel / §6(c) evaluation on frozen repair sets."""
import argparse
from copy import deepcopy
import json
from pathlib import Path
import sys
from types import SimpleNamespace

from eval.run_suite import run_suite, task_result
from repair.adapter_eval import NoTeacherBudget, assert_no_teacher_calls, leakage_check, serving_spec
from repair.artifacts import checked_document, provenance, read_json, run_manifest, seal, write_sealed
from repair.joint import usage_summary
from sed.config import load_config
from sed.manifest import authorize_task, load_frozen, resolve_task
from sed.provenance import code_hashes, safe_config, versions
from sed.storage import canonical_bytes, digest, file_hash, locked, write_json
from students.vllm_student import context_options
from tools.repair_coverage import checked_repair_splits

SETS = {'D_unseen_quick': 24, 'D_probe': 60}
THETA0 = 'e4b-tsft-k20-e32-lr5e-5'
RECEIPT = 'v2_completion.json'


def evaluation_tasks(path, manifest, freeze, set_name):
    """Validate frozen listings without selecting, sorting or expanding tasks."""
    if set_name not in SETS:
        raise ValueError('V2 task evaluation allows only D_unseen_quick and D_probe')
    splits = checked_repair_splits(path, manifest, freeze)
    resolved = {}
    for name, size in SETS.items():
        listing = splits['sets'][name]
        ids = listing['task_ids']
        if (len(ids) != size or len(set(ids)) != size or listing['list_sha256'] != digest(ids)
                or listing['manifest_sha256'] != digest(manifest)
                or listing['split_freeze_sha256'] != digest(freeze)):
            raise ValueError(f'{name} requires exactly {size} frozen tasks with matching hashes')
        tasks = [resolve_task(manifest, freeze, tid) for tid in ids]
        for task in tasks:
            authorize_task(task, 'evaluation')
            if task['split'] != 'train' or task['source_split'] != 'train' or task['sealed']:
                raise ValueError('Repair evaluation requires unsealed TRAIN tasks')
        if (len({t['group_id'] for t in tasks}) != size
                or listing['group_ids'] != [t['group_id'] for t in tasks]):
            raise ValueError('Frozen repair task groups differ')
        resolved[name] = tasks
    if not set(splits['sets']['D_unseen_quick']['task_ids']) < set(splits['sets']['D_probe']['task_ids']):
        raise ValueError('D_unseen_quick must be a subset of D_probe')
    return splits, splits['sets'][set_name], resolved[set_name]


def assert_teacher_free(output):
    """Apply E0.6 accounting to suite ledgers, including interrupted attempts."""
    output = Path(output)
    for pattern in ('**/teacher_usage.jsonl', '**/usage.jsonl'):
        for path in output.glob(pattern):
            assert_no_teacher_calls([dict(teacher_usage=usage_summary(path), mode='U',
                                         guidance_injections=[], steps=[])])
            # Cache hits also indicate a teacher policy, even with zero billed tokens.
            if path.read_text().strip():
                raise AssertionError('Teacher usage found in V2 task evaluation')
    for path in output.glob('**/teacher_report.json'):
        report = read_json(path)
        if any(report.get(k, 0) for k in ('transport_failures', 'transport_failed_attempts',
               'rejections', 'parser_rejections', 'truncated_outputs', 'unknown_outcomes')):
            raise AssertionError('Teacher usage found in V2 task evaluation')


def artifact_hashes(directory):
    return {str(p.relative_to(directory)): file_hash(p) for p in sorted(directory.rglob('*'))
            if p.is_file() and p.name != RECEIPT and not p.name.endswith('.lock')}


def decoding_rows(directory, tasks):
    rows = []
    for i, task in enumerate(tasks):
        episode_dir = directory / 'episodes' / f"{i:03d}-{digest(task['task_id'])[:12]}"
        row = task_result(task, episode_dir)
        episode_path = episode_dir / 'episode.json'
        if episode_path.exists():
            episode = read_json(episode_path)
            if episode['task_id'] != task['task_id']:
                raise ValueError('Suite episode task identity differs')
            if episode.get('complete') is not True:
                row['status'] = 'incomplete'
        rows.append(row)
    return rows


def totals(rows, decodings):
    by_decoding = {}
    for name in decodings:
        values = [r['success'][name] for r in rows]
        known = [v for v in values if v is not None]
        by_decoding[name] = dict(successes=sum(v is True for v in known), valid_episodes=len(known),
            unknown_or_error_episodes=len(values) - len(known),
            success_rate=sum(known) / len(known) if known else None)
    means = [r['sampled_success_mean'] for r in rows if r['sampled_success_mean'] is not None]
    return dict(unique_instances=len(rows), task_groups=len({r['group_id'] for r in rows}),
        greedy_successes=by_decoding.get('greedy', {}).get('successes'),
        sampled_success_rate=sum(means) / len(means) if means else None,
        sampled_valid_tasks=len(means), per_decoding=by_decoding,
        unknown_or_error_episodes={k: v['unknown_or_error_episodes'] for k, v in by_decoding.items()})


def summarize(identity, results, tasks):
    decodings = identity['decodings']
    sampled = [name for name, spec in decodings.items() if spec['decoding'] == 'sampled']
    per_task = []
    for i, task in enumerate(tasks):
        success = {}
        for name in decodings:
            row = results[name][i]
            if any(row[k] != task[k] for k in ('task_id', 'group_id', 'task_type')):
                raise ValueError('Suite result task identity differs')
            success[name] = row['success'] if row['status'] == 'ok' and type(row['success']) is bool else None
        valid = [success[name] for name in sampled if success[name] is not None]
        per_task.append(dict(task_id=task['task_id'], group_id=task['group_id'], task_type=task['task_type'],
            success=success, sampled_valid_seeds=len(valid),
            sampled_success_mean=sum(valid) / len(valid) if valid else None))
    result = dict(schema_version=1, set=identity['set'], list_sha256=identity['listing']['list_sha256'],
        evaluated_adapter=identity['evaluated_adapter'], decodings=decodings, per_task=per_task,
        totals=totals(per_task, decodings), teacher_calls=0, training_exposure=0, model_updates=0,
        global_budget=NoTeacherBudget().summary(),
        interpretation='From-start, no-hint evaluation. Unknown/error episodes are null, never failures. '
                      'Sampled success averages valid seeds per task, then tasks with at least one valid seed.')
    if identity['set'] == 'D_probe':
        quick = set(identity['quick_listing']['task_ids'])
        result['subsets'] = {
            'non_quick_36': dict(label='Main §6(c) comparison: D_probe minus D_unseen_quick',
                task_ids=[r['task_id'] for r in per_task if r['task_id'] not in quick],
                totals=totals([r for r in per_task if r['task_id'] not in quick], decodings)),
            'quick_24': dict(label='Checkpoint-selection tasks; descriptive only',
                task_ids=[r['task_id'] for r in per_task if r['task_id'] in quick],
                totals=totals([r for r in per_task if r['task_id'] in quick], decodings))}
        result['interpretation'] += ' All 60 tasks are descriptive; the non-quick 36 are the main comparison.'
    return result


def run(args, *, suite_runner=None):
    if type(args.workers) is not int or args.workers < 1:
        raise ValueError('workers must be positive')
    if (any(type(s) is not int for s in args.sampled_seeds)
            or len(set(args.sampled_seeds)) != len(args.sampled_seeds)):
        raise ValueError('Sampled seeds must be unique integers')
    if not args.greedy and not args.sampled_seeds:
        raise ValueError('Select --greedy and/or --sampled-seeds')
    config = load_config(args.config)
    if args.data_root:
        config['environment']['data_root'] = args.data_root
    if not config['environment']['data_root']:
        raise ValueError('Evaluation requires data-root/ALFWORLD_DATA_ROOT')
    manifest, freeze = load_frozen(config['paths']['manifest'], config['paths']['split_freeze'])
    splits, listing, tasks = evaluation_tasks(args.repair_splits, manifest, freeze, args.set)
    spec, adapter, adapter_inputs = serving_spec(SimpleNamespace(**vars(args), dry_run=False), config)
    # serving_spec has now loaded provenance; match run_suite's effective spec.
    spec = {**spec, **context_options(config)}
    theta0 = adapter['name'] == THETA0 and adapter['training'] is None
    adapter.update(arm='theta0' if theta0 else adapter['training']['arm'] if adapter['training'] else None,
                   arm_index=0 if theta0 else None, is_theta0=theta0)
    leakage = leakage_check([dict(spec=dict(task_id=t['task_id'])) for t in tasks], adapter)
    decodings = {}
    if args.greedy:
        decodings['greedy'] = dict(spec, decoding='greedy')
    for seed in args.sampled_seeds:
        decodings[f'sampled-s{seed}'] = dict(spec, decoding='sampled', sampling_seed=seed)
    output = Path(args.output)
    for name in ('raw', 'verified'):
        if output.resolve().is_relative_to(Path(config['paths'][name]).resolve()):
            raise ValueError('Evaluation output must be outside training data destinations')
    inputs = [__file__, args.config, args.repair_splits, config['paths']['manifest'],
              config['paths']['split_freeze'], *adapter_inputs]
    root = Path(__file__).resolve().parents[1]
    identity = dict(set=args.set, listing=listing, quick_listing=splits['sets']['D_unseen_quick'],
        repair_splits_sha256=splits['content_sha256'], evaluated_adapter=adapter, leakage_check=leakage,
        config=safe_config(config), decodings=decodings, workers=args.workers,
        input_sha256={str(p): file_hash(p) for p in inputs}, dependencies=versions(),
        code_sha256={**code_hashes(root), **{str(p.relative_to(root)): file_hash(p)
                      for p in (root / 'repair').glob('*.py')}},
        from_start=True, teacher_calls_allowed=0, evaluation_only=True,
        student_sampling=dict(temperature=.7, top_p=.95, n=1,
            seed_rule='sha256([sampling_seed, task_id, zero_based_turn])[:8]'))
    run_sha = digest(identity)
    if args.resume:
        previous = checked_document(output / 'plan.json')
        if previous['identity'] != identity or previous['run_sha256'] != run_sha:
            raise ValueError('Resume requires identical plan, adapter, inputs, config, code and dependencies')
        if (output / 'tasks.json').read_bytes() != canonical_bytes(listing) + b'\n':
            raise ValueError('Resume task listing differs')
    else:
        output.mkdir(parents=True, exist_ok=False)
        write_json(output / 'tasks.json', listing, exclusive=True)
        write_sealed(output / 'plan.json', dict(identity=identity, run_sha256=run_sha))
        prov = provenance(output, config, command='v2_task_eval run', inputs=inputs,
                          details=dict(run_sha256=run_sha, evaluation_only=True, from_start=True))
        write_json(output / 'run_manifest.json', run_manifest(prov, checkpoint=adapter, tokenizer=None,
            tasks=tasks, seeds=dict(environment=config['environment']['seed'], sampling=args.sampled_seeds),
            run_sha256=run_sha, evaluation_only=True, training_exposure=0, teacher_calls=0,
            teacher_alias=config['models']['teacher']['provider_model_id'],
            teacher_returned_model_id=None), exclusive=True)
    suite_runner = suite_runner or run_suite
    with locked(output / 'run.lock'):
        assert_teacher_free(output)
        # Validate ALL existing decodings before doing any new execution.
        for name, current_spec in decodings.items():
            directory = output / name
            if directory.exists():
                if not (directory / RECEIPT).is_file():
                    raise ValueError(f'Incomplete decoding {name}; resume only fills missing directories')
                receipt = checked_document(directory / RECEIPT)
                if (receipt['run_sha256'] != run_sha or receipt['spec'] != current_spec
                        or receipt['artifacts'] != artifact_hashes(directory)):
                    raise ValueError('Resume decoding identity or artifacts differ')
            elif (output / 'summary.json').exists():
                raise ValueError('Completed summary has a missing decoding')
        results = {}
        for name, current_spec in decodings.items():
            directory = output / name
            if not directory.exists():
                suite_runner(config=deepcopy(config), tasks_path=output / 'tasks.json', output=directory,
                    policy='student', spec=dict(current_spec), workers=args.workers, inputs=inputs)
                assert_teacher_free(directory)
                suite_summary = read_json(directory / 'summary.json')
                if (suite_summary['policy'] != 'student' or suite_summary['student'] != current_spec
                        or suite_summary['task_list_sha256'] != listing['list_sha256']):
                    raise ValueError('Suite result differs from planned evaluation')
                decoding_rows(directory, tasks)  # Check task identities before sealing completion.
                write_sealed(directory / RECEIPT, dict(run_sha256=run_sha, spec=current_spec,
                                                      artifacts=artifact_hashes(directory)))
            results[name] = decoding_rows(directory, tasks)
        assert_teacher_free(output)
        result = dict(summarize(identity, results, tasks), run_sha256=run_sha)
        summary_path = output / 'summary.json'
        if summary_path.exists():
            if checked_document(summary_path) != seal(result):
                raise ValueError('Resume summary differs from completed decodings')
        else:
            write_sealed(summary_path, result)
    print(json.dumps(dict(set=args.set, totals=result['totals'], output=str(output))))
    return int(any(result['totals']['unknown_or_error_episodes'].values()))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    p = sub.add_parser('run')
    p.add_argument('--set', choices=tuple(SETS), required=True)
    p.add_argument('--repair-splits', default='data/manifest/repair_v01_splits.json')
    p.add_argument('--greedy', action='store_true')
    p.add_argument('--sampled-seeds', nargs='*', type=int, default=[])
    p.add_argument('--config', default='configs/base_e4b.yaml')
    p.add_argument('--data-root')
    for name in ('endpoint', 'model-name', 'server-provenance', 'lora-check', 'output'):
        p.add_argument('--' + name, required=True)
    p.add_argument('--workers', type=int, default=1)
    p.add_argument('--resume', action='store_true')
    return run(parser.parse_args(argv))


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(json.dumps(dict(status='error', error_type=type(exc).__name__)))
        if isinstance(exc, (ValueError, AssertionError)):
            origin = exc.__traceback__
            while origin.tb_next is not None:
                origin = origin.tb_next
            # Only this tool's validation messages are safe to print.
            if origin.tb_frame.f_globals is globals():
                print(str(exc), file=sys.stderr)
        raise SystemExit(1) from None
