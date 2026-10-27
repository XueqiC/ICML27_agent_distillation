"""Teacher takeover diagnostics. No training exports or student/model serving."""
import argparse
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from functools import partial
import json
import multiprocessing
from pathlib import Path
import statistics

from repair.artifacts import provenance, read_json, run_manifest
from repair.records import TrajectoryRecord
from repair.takeover import (SharedTeacherBudget, RecordedEnvironment, execute_takeover,
                             make_teacher_policy, run_key, usage_summary)
from sed.config import load_config
from sed.provenance import code_hashes, safe_config, versions
from sed.react import PROMPT_TEMPLATE, PROMPT_VERSION
from sed.storage import atomic_write, digest, file_hash, locked, write_json
from tools.cg_probe import load_cases, markdown_table, select_groups


def spec_from_cg(args):
    config = load_config(args.config)
    groups = select_groups(','.join(args.groups))
    cases = load_cases(args.failures, args.coverage_root, groups, config, dry_run=args.dry_run)
    spec = []
    for case in cases:
        original = case['original']
        if not 0 <= args.branch_index < min(len(original.steps), original.max_steps):
            raise ValueError('Branch must leave an executable continuation')
        episode = next(Path(p).parent for p in case['source']['evidence_sha256'] if Path(p).name == 'episode.json')
        spec.append({'case_id': original.task_id, 'group': case['group'], 'task_id': original.task_id,
            'source': {'failures_json': str(Path(args.failures).resolve()),
                       'coverage_root': str(Path(args.coverage_root).resolve()),
                       'episode_dir': str(episode.relative_to(Path(args.coverage_root).resolve()))},
            'branch_index': args.branch_index})
    write_json(args.output, spec, exclusive=True)
    return 0


def load_spec(path, config, *, dry_run=False):
    spec = read_json(path)
    if not isinstance(spec, list) or not spec:
        raise ValueError('Prefix spec must be a nonempty list')
    ids, cases, inputs, cache = set(), [], {str(Path(path).resolve())}, {}
    for entry in spec:
        if set(entry) != {'case_id', 'group', 'task_id', 'source', 'branch_index'}:
            raise ValueError('Invalid prefix spec entry fields')
        if not isinstance(entry['case_id'], str) or not entry['case_id'] or entry['case_id'] in ids:
            raise ValueError('Case ids must be unique nonempty strings')
        ids.add(entry['case_id'])
        source = entry['source']
        if set(source) != {'failures_json', 'coverage_root', 'episode_dir'}:
            raise ValueError('Invalid source fields')
        def resolve(value):
            p = Path(value)
            return p.resolve() if p.is_absolute() else (Path(path).resolve().parent / p).resolve()
        failures, root = resolve(source['failures_json']), resolve(source['coverage_root'])
        key = (str(failures), str(root), entry['group'])
        if key not in cache:
            cache[key] = load_cases(failures, root, [entry['group']], config, dry_run=dry_run)[0]
        loaded = cache[key]
        original = loaded['original']
        episode = (root / source['episode_dir']).resolve()
        if not episode.is_relative_to(root) or str(episode / 'episode.json') not in loaded['source']['evidence_sha256']:
            raise ValueError('Spec episode differs from validated CG source')
        n = entry['branch_index']
        if original.task_id != entry['task_id'] or type(n) is not int or not 0 <= n < min(len(original.steps), original.max_steps):
            raise ValueError('Spec task or branch differs from source')
        if any(s.get('terminated') or s.get('truncated') for s in original.steps[:n]):
            raise ValueError('Branch prefix is terminal')
        inputs.update([str(failures), str(failures.parent / 'summary.json'), *loaded['source']['evidence_sha256']])
        # load_cases adds an instruction for its structured guidance envelope.
        # Collection and CG student ReAct turns already carry the goal in the
        # initial observation; do not prepend that CG-only duplicate instruction.
        task = {k: v for k, v in loaded['task'].items() if k != 'instruction'}
        cases.append({'spec': entry, **loaded, 'task': task})
    return cases, sorted(inputs)


def _job(job):
    case, config, output, identity, mode, run, dry_run, caps = job
    budget = SharedTeacherBudget(Path(output) / 'global_teacher_budget.jsonl', **caps)
    original = TrajectoryRecord.from_json(case['original'])
    n = 0 if mode == 'from_start' else case['spec']['branch_index']
    if dry_run:
        original = replace(original, initial={**original.initial, 'audit_sha256': digest({'won': False})},
            steps=[{**s, 'audit_sha256': digest({'won': False})} for s in original.steps])
        env_factory = partial(RecordedEnvironment, original, n)
    else:
        from harness.alfworld_env import Environment
        env_factory = partial(Environment, config['environment']['data_root'], split='train',
                              seed=original.env_seed, max_episode_steps=original.max_steps)
    policy_factory = partial(make_teacher_policy, config, case['task'], output=output, budget=budget, dry_run=dry_run)
    return execute_takeover(case['spec'], original, mode=mode, run=run, env_factory=env_factory,
        policy_factory=policy_factory, output=output, run_sha256=identity, budget=budget)


def _reconcile(output, budget):
    # Settlement is idempotent. Durable teacher receipts recover a crash between
    # usage append and settlement. Failed or interrupted requests retain bounds.
    for path in sorted((output / 'calls').glob('*/*/usage.jsonl')):
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        for row in rows:
            if row['status'] not in ('started', 'cache_hit'):
                budget.settle(row['call_id'], row.get('usage'))
                if row['status'] in ('failed', 'incomplete') and not row.get('will_retry'):
                    budget.halt('teacher_halted')


def run(args):
    if args.workers < 1 or args.runs < 1:
        raise ValueError('workers/runs must be positive')
    if args.canary and args.dry_run:
        raise ValueError('Canary requires the real seam; cannot combine with dry-run')
    config = load_config(args.config)
    if args.data_root:
        config['environment']['data_root'] = args.data_root
    cases, inputs = load_spec(args.spec, config, dry_run=args.dry_run)
    if args.canary:
        cases, args.runs, args.workers = cases[:1], 1, 1
    if not args.dry_run and not config['environment']['data_root']:
        raise ValueError('Real takeover requires data-root')
    caps = {'input_tokens': args.max_teacher_input_tokens, 'output_tokens': args.max_teacher_output_tokens}
    if args.canary:
        caps = {'input_tokens': min(caps['input_tokens'], 100_000),
                'output_tokens': min(caps['output_tokens'], 8192)}
    inputs += [str(Path(args.config).resolve()), __file__]
    if not args.dry_run:
        inputs += [config['paths'][k] for k in ('manifest', 'split_freeze')]
    root = Path(__file__).resolve().parents[1]
    identity = {'config': safe_config(config), 'cases': [c['spec'] for c in cases],
        'originals_sha256': digest([c['original'].to_json() for c in cases]),
        'input_sha256': {str(p): file_hash(p) for p in inputs},
        'code_sha256': {**code_hashes(root), **{str(p.relative_to(root)): file_hash(p) for p in (root / 'repair').glob('*.py')}},
        'dependencies': versions(), 'dry_run': args.dry_run, 'canary': args.canary,
        'mode': 'from_start' if args.from_start else 'takeover', 'runs': args.runs, 'caps': caps,
        'prompt_version': PROMPT_VERSION, 'prompt_template_sha256': digest(PROMPT_TEMPLATE)}
    run_sha = digest(identity)
    output = Path(args.output)
    if args.resume:
        if read_json(output / 'plan.json')['identity'] != identity:
            raise ValueError('Resume requires identical spec, evidence, config, caps, code and dependencies')
    else:
        output.mkdir(parents=True, exist_ok=False)
        prov = provenance(output, config, command='teacher_takeover run', inputs=inputs,
                          details={'run_sha256': run_sha, 'evaluation_only': True, 'dry_run': args.dry_run})
        write_json(output / 'plan.json', {'identity': identity, 'run_sha256': run_sha}, exclusive=True)
        write_json(output / 'run_manifest.json', run_manifest(prov, checkpoint=config['models']['student'], tokenizer=None,
            tasks=[c['task'] for c in cases], seeds={'environment': config['environment']['seed']},
            run_sha256=run_sha, teacher_alias=config['models']['teacher']['provider_model_id'],
            prompt_version=PROMPT_VERSION, prompt_template_sha256=digest(PROMPT_TEMPLATE),
            evaluation_only=True, training_exposure=0, dry_run=args.dry_run,
            source_evidence=[c['source'] for c in cases]), exclusive=True)
    with locked(output / 'run.lock'):
        budget = SharedTeacherBudget(output / 'global_teacher_budget.jsonl', **caps)
        _reconcile(output, budget)
        jobs = [({**c, 'original': c['original'].to_json()}, config, str(output), run_sha,
                 identity['mode'], r, args.dry_run, caps) for c in cases for r in range(args.runs)]
        if args.workers == 1:
            rows = [_job(j) for j in jobs]
        else:
            with ProcessPoolExecutor(max_workers=args.workers, mp_context=multiprocessing.get_context('spawn')) as pool:
                rows = list(pool.map(_job, jobs))
        report = {'expected': len(jobs), 'recorded': len(rows), 'void': sum(r['void'] for r in rows),
                  'global_budget': budget.summary(), 'run_sha256': run_sha, 'dry_run': args.dry_run}
        write_json(output / 'completion.json', report)
        manifest = read_json(output / 'run_manifest.json')
        # Include orphan attempts as well as atomic per-run results.
        manifest['teacher_attempts'] = [dict(call, usage_ledger=str(p.relative_to(output)))
            for p in sorted((output / 'calls').glob('*/*/usage.jsonl')) for call in usage_summary(p)['calls']]
        manifest['global_budget'] = budget.summary()
        write_json(output / 'run_manifest.json', manifest)
    print(json.dumps(report))
    return int(report['void'] > 0 or report['global_budget']['stop_reason'] is not None)


def summarize_records(plan, rows):
    identity = plan['identity']
    expected = {run_key(c['case_id'], identity['mode'], r): (c['case_id'], identity['mode'], r)
                for c in identity['cases'] for r in range(identity['runs'])}
    seen = set()
    for row in rows:
        key = row['run_key']
        if (key in seen or expected.get(key) != (row['case_id'], row['mode'], row['run']) or
                row['run_sha256'] != plan['run_sha256']):
            raise ValueError('Unexpected, duplicate or foreign takeover record')
        seen.add(key)
    per_case = []
    for case in identity['cases']:
        current = [r for r in rows if r['case_id'] == case['case_id']]
        valid = [r for r in current if not r['void']]
        successful = [r for r in valid if r['success'] is True]
        per_case.append({'case_id': case['case_id'], 'mode': identity['mode'], 'expected_runs': identity['runs'],
            'runs': len(valid), 'recorded': len(current), 'void': sum(r['void'] for r in current),
            'successes': len(successful), 'found': sum(r['metrics']['found'] for r in valid),
            'picked': sum(r['metrics']['picked'] for r in valid),
            'median_steps_to_success': statistics.median(r['metrics']['steps_to_success'] for r in successful) if successful else None,
            'feasible': bool(successful) if successful or len(valid) == identity['runs'] else None})
    return {'schema_version': 1, 'run_sha256': plan['run_sha256'], 'dry_run': identity['dry_run'],
            'per_case': per_case, 'missing_runs': sorted(set(expected) - seen),
            'teacher_usage': {k: sum(r['teacher_usage']['totals'][k] for r in rows)
                              for k in ('input_tokens', 'output_tokens', 'reasoning_tokens')},
            'unknown_usage_attempts': sum(r['teacher_usage']['unknown_usage_attempts'] for r in rows),
            'limitations': 'TRAIN diagnostic; synthetic runs establish no feasibility. Unknown/void runs are not failures.'}


def summarize(args):
    root, output = Path(args.rollouts), Path(args.output)
    plan = read_json(root / 'plan.json')
    result = summarize_records(plan, [read_json(p) for p in sorted((root / 'records').glob('*.json'))])
    budget = SharedTeacherBudget(root / 'global_teacher_budget.jsonl', **plan['identity']['caps'])
    result['global_budget'] = budget.summary()
    all_usage = [usage_summary(p) for p in sorted((root / 'calls').glob('*/*/usage.jsonl'))]
    result['all_attempt_usage_including_interrupted_runs'] = {
        k: sum(u['totals'][k] for u in all_usage) for k in ('input_tokens', 'output_tokens', 'reasoning_tokens')}
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / 'summary.json', result, exclusive=True)
    text = '# Teacher takeover diagnostic\n\n' + result['limitations'] + '\n\n'
    text += markdown_table(result['per_case'], ['case_id', 'mode', 'expected_runs', 'runs', 'void', 'successes',
                                               'found', 'picked', 'median_steps_to_success', 'feasible'])
    text += '\n\nAll-attempt usage: ' + json.dumps(result['all_attempt_usage_including_interrupted_runs'], sort_keys=True)
    text += '\n\nGlobal accounting: ' + json.dumps(result['global_budget'], sort_keys=True) + '\n'
    atomic_write(output / 'summary.md', text.encode(), exclusive=True)
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    s = sub.add_parser('spec-from-cg')
    s.add_argument('--failures', required=True)
    s.add_argument('--coverage-root', required=True)
    s.add_argument('--groups', nargs='+', required=True, help='Comma-separated or space-separated CG group names')
    s.add_argument('--branch-index', type=int, default=10)
    r = sub.add_parser('run')
    r.add_argument('--spec', required=True)
    r.add_argument('--runs', type=int, default=4)
    r.add_argument('--workers', type=int, default=1)
    r.add_argument('--max-teacher-input-tokens', type=int, default=5_000_000)
    r.add_argument('--max-teacher-output-tokens', type=int, default=500_000)
    r.add_argument('--from-start', action='store_true')
    r.add_argument('--canary', action='store_true')
    r.add_argument('--resume', action='store_true')
    r.add_argument('--data-root')
    for p in (s, r):
        p.add_argument('--config', default='configs/base_e4b.yaml')
        p.add_argument('--dry-run', action='store_true')
        p.add_argument('--output', required=True)
    a = sub.add_parser('summarize')
    a.add_argument('--rollouts', required=True)
    a.add_argument('--output', required=True)
    args = parser.parse_args(argv)
    return {'spec-from-cg': spec_from_cg, 'run': run, 'summarize': summarize}[args.command](args)


if __name__ == '__main__':
    # Never emit raw SDK exception text from an operational CLI failure.
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(json.dumps({'status': 'error', 'error_type': type(exc).__name__}))
        raise SystemExit(1) from None
