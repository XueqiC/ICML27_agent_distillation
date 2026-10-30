"""Run/resume/summarize E0.3 joint guidance with one shared teacher budget."""
import argparse
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from functools import partial
import json
import multiprocessing
from pathlib import Path
import statistics

from repair.artifacts import provenance, read_json, run_manifest
from repair.guidance import cluster_bootstrap
from repair.joint import (JointPrompt, FakeEnvironment, FakeStudent, execute_joint, make_reviewer,
                          run_key, TEMPLATE_VERSION, OUTPUT_CAP, usage_summary)
from repair.records import TrajectoryRecord
from repair.takeover import SharedTeacherBudget
from sed.config import load_config
from sed.provenance import code_hashes, safe_config, versions
from sed.storage import atomic_write, digest, file_hash, locked, write_json
from tools.cg_probe import _student, serving_spec, markdown_table
from tools.teacher_takeover import load_spec, _reconcile


def _job(job):
    case, config, spec, output, identity, mode, decoding, seed, dry, caps, ledger_path, from_start, review_template, *evaluation = job
    if evaluation:
        from repair.adapter_eval import NoTeacherBudget
        budget = NoTeacherBudget(caps)
    else:
        budget = SharedTeacherBudget(ledger_path, **caps)
    original = TrajectoryRecord.from_json(case['original'])
    n = 0 if from_start else case['spec']['branch_index']
    if dry:
        original = replace(original, initial={**original.initial, 'audit_sha256': digest({'won': False})},
            steps=[{**s, 'audit_sha256': digest({'won': False})} for s in original.steps])
        env_factory, student_factory = partial(FakeEnvironment, original, n), FakeStudent
    else:
        from harness.alfworld_env import Environment
        env_factory = partial(Environment, config['environment']['data_root'], split='train',
                              seed=original.env_seed, max_episode_steps=original.max_steps)
        student_factory = partial(_student, spec=spec, decoding=decoding, seed=seed, task_id=original.task_id)
    return execute_joint(case['spec'], original, mode=mode, decoding=decoding, seed=seed,
        env_factory=env_factory, student_factory=student_factory,
        reviewer_factory=partial(make_reviewer, config, case['task'], output=output, budget=budget, dry_run=dry,
                                 review_template=review_template),
        output=output, run_sha256=identity, budget=budget, from_start=from_start, review_template=review_template,
        **({'evaluated_adapter': evaluation[0]} if evaluation else {}))


def run(args):
    eval_adapter = getattr(args, 'eval_adapter', False)
    if eval_adapter and args.modes != ['U']:
        raise ValueError('--eval-adapter requires --modes U; G/G1 are forbidden')
    if not eval_adapter and not args.budget_ledger:
        raise ValueError('--budget-ledger is required unless --eval-adapter is set')
    prompt = JointPrompt({}, template_version=args.review_template)
    if args.workers < 1 or len(set(args.seeds)) != len(args.seeds) or len(set(args.modes)) != len(args.modes):
        raise ValueError('Positive workers and unique modes/seeds required')
    config = load_config(args.config)
    if args.data_root:
        config['environment']['data_root'] = args.data_root
    cases, inputs = load_spec(args.spec, config, dry_run=args.dry_run)
    if not args.dry_run and not all((args.endpoint, args.server_provenance, args.lora_check,
                                    args.model_name, config['environment']['data_root'])):
        raise ValueError('Real runs require endpoint, server-provenance, lora-check, model-name, data-root')
    if eval_adapter:
        from repair.adapter_eval import (NoTeacherBudget, PREFIX_VERIFICATION, assert_no_teacher_calls,
                                         leakage_check, serving_spec as eval_serving_spec)
        spec, evaluated_adapter, adapter_inputs = eval_serving_spec(args, config)
        leakage = leakage_check(cases, evaluated_adapter)
        inputs += adapter_inputs
    else:
        spec = {} if args.dry_run else serving_spec(args, config, cases)
    caps = dict(input_tokens=args.max_teacher_input_tokens, output_tokens=args.max_teacher_output_tokens)
    inputs += [str(Path(args.config).resolve()), __file__, 'tools/cg_probe.py', 'tools/teacher_takeover.py']
    if not args.dry_run:
        inputs += [args.server_provenance, args.lora_check, config['paths']['manifest'], config['paths']['split_freeze']]
    root, output = Path(__file__).resolve().parents[1], Path(args.output)
    identity = dict(config=safe_config(config), serving=spec, cases=[c['spec'] for c in cases],
        originals_sha256=digest([c['original'].to_json() for c in cases]),
        input_sha256={str(p): file_hash(p) for p in inputs},
        code_sha256={**code_hashes(root), **{str(p.relative_to(root)): file_hash(p) for p in (root / 'repair').glob('*.py')}},
        dependencies=versions(), dry_run=args.dry_run, from_start=args.from_start, modes=args.modes, seeds=args.seeds,
        caps=caps, budget_ledger=str(Path(args.budget_ledger).resolve()) if args.budget_ledger else None,
        template_version=prompt.template_version, template_sha256=prompt.template_sha256,
        output_cap_including_reasoning=OUTPUT_CAP, H=40,
        student_sampling={'temperature': .7, 'top_p': .95,
                          'seed_rule': 'sha256([sampling_seed, task_id, zero_based_turn])[:8]; regeneration uses same turn'})
    if eval_adapter:
        identity.update(eval_adapter=True, evaluated_adapter=evaluated_adapter, leakage_check=leakage,
                        prefix_verification=PREFIX_VERIFICATION, teacher_calls_allowed=0)
    run_sha = digest(identity)
    # Check the external shared caps before creating a new output directory.
    budget = NoTeacherBudget(caps) if eval_adapter else SharedTeacherBudget(identity['budget_ledger'], **caps)
    if args.resume:
        previous = read_json(output / 'plan.json')['identity']
        if previous['template_version'] != args.review_template:
            raise ValueError('A joint run directory cannot mix review template versions')
        if previous != identity:
            raise ValueError('Resume requires identical evidence, config, code, dependencies, seeds and caps')
    else:
        output.mkdir(parents=True, exist_ok=False)
        prov = provenance(output, config, command='joint_guidance run', inputs=inputs,
                          details={'run_sha256': run_sha, 'evaluation_only': True, 'dry_run': args.dry_run})
        write_json(output / 'plan.json', {'identity': identity, 'run_sha256': run_sha}, exclusive=True)
        write_json(output / 'run_manifest.json', run_manifest(prov, checkpoint=spec, tokenizer=None,
            tasks=[c['task'] for c in cases], seeds={'environment': config['environment']['seed'], 'sampling': args.seeds},
            run_sha256=run_sha, teacher_alias=config['models']['teacher']['provider_model_id'],
            evaluation_only=True, dry_run=args.dry_run, training_exposure=0,
            source_evidence=[c['source'] for c in cases], template_version=prompt.template_version,
            template_sha256=prompt.template_sha256), exclusive=True)
    with locked(output / 'run.lock'):
        if eval_adapter:
            assert_no_teacher_calls([], output)
        else:
            _reconcile(output, budget)
        jobs = [({**c, 'original': c['original'].to_json()}, config, spec, str(output), run_sha,
                 mode, decoding, seed, args.dry_run, caps, identity['budget_ledger'], args.from_start, args.review_template)
                for c in cases for mode in args.modes
                for decoding, seed in [('greedy', None), *[('sampled', s) for s in args.seeds]]]
        if eval_adapter:
            jobs = [(*job, evaluated_adapter) for job in jobs]
        if args.workers == 1:
            rows = [_job(job) for job in jobs]
        else:
            with ProcessPoolExecutor(max_workers=args.workers, mp_context=multiprocessing.get_context('spawn')) as pool:
                rows = list(pool.map(_job, jobs))
        if eval_adapter:
            assert_no_teacher_calls(rows, output)
        report = dict(expected=len(jobs), recorded=len(rows), void=sum(r['void'] for r in rows),
                      global_budget=budget.summary(), run_sha256=run_sha, dry_run=args.dry_run)
        write_json(output / 'completion.json', report)
        manifest = read_json(output / 'run_manifest.json')
        manifest.update(global_budget=budget.summary(), teacher_attempts=[dict(c, usage_ledger=str(p.relative_to(output)))
            for p in sorted((output / 'calls').glob('*/*/usage.jsonl')) for c in usage_summary(p)['calls']])
        write_json(output / 'run_manifest.json', manifest)
    print(json.dumps(report))
    return int(report['void'] > 0 or report['global_budget']['stop_reason'] is not None)


def metric_table(rows, expected):
    valid = [r for r in rows if not r['void']]
    complete = len(valid) == expected and expected > 0
    metrics = [r['metrics'] for r in valid]
    result = dict(expected=expected, recorded=len(rows), valid=len(valid), void=len(rows) - len(valid),
                  missing=expected - len(rows))
    for name in ('success', 'picked', 'found'):
        result[name + '_count'] = sum(m[name] is True for m in metrics)
        result[name + '_rate'] = (result[name + '_count'] / expected
            if complete and all(type(m[name]) is bool for m in metrics) else None)
    for name, numerator, denominator in [('teacher_step_share', 'teacher_executed_steps', 'steps_used_after_branch'),
                                         ('follow_rate', 'guided_steps_followed', 'guides')]:
        count = sum(m[denominator] for m in metrics)
        result[numerator] = sum(m[numerator] for m in metrics)
        result[denominator] = count
        result[name] = sum(m[numerator] for m in metrics) / count if complete and count else None
    result['not_found_count'] = sum(m['found'] is False for m in metrics)
    result['discovery_times'] = [dict(seed=r['seed'], found=r['metrics']['found'],
        steps_to_find=r['metrics']['steps_to_find'], steps_remaining_when_found=r['metrics']['steps_remaining_when_found']) for r in valid]
    return result


def summarize_records(plan, rows):
    identity = plan['identity']
    trials = [('greedy', None), *[('sampled', s) for s in identity['seeds']]]
    expected = {run_key(c['case_id'], m, d, s): (c['case_id'], m, d, s)
                for c in identity['cases'] for m in identity['modes'] for d, s in trials}
    seen = set()
    for row in rows:
        key = row['run_key']
        if (key in seen or expected.get(key) != (row['case_id'], row['mode'], row['decoding'], row['seed'])
                or row['run_sha256'] != plan['run_sha256']):
            raise ValueError('Duplicate, unexpected or foreign joint record')
        seen.add(key)
    per_case = []
    for case in identity['cases']:
        for mode in identity['modes']:
            for decoding, count in [('greedy', 1), ('sampled', len(identity['seeds']))]:
                current = [r for r in rows if (r['case_id'], r['mode'], r['decoding']) == (case['case_id'], mode, decoding)]
                per_case.append(dict(case_id=case['case_id'], mode=mode, decoding=decoding, **metric_table(current, count)))
    aggregate = []
    rate_keys = ('success_rate', 'picked_rate', 'found_rate', 'teacher_step_share', 'follow_rate')
    for mode in identity['modes']:
        for decoding in ('greedy', 'sampled'):
            cells = [r for r in per_case if (r['mode'], r['decoding']) == (mode, decoding)]
            current = [r for r in rows if (r['mode'], r['decoding']) == (mode, decoding)]
            result = dict(mode=mode, decoding=decoding, cases=len(cells),
                          **metric_table(current, sum(c['expected'] for c in cells)))
            for metric in rate_keys:
                values = [r[metric] for r in cells]
                # Primary rates average cases; teacher share/follow pool actual
                # suffix turns/guided attempts, so zero-guidance cases add zero.
                known = [v for v in values if v is not None]
                if metric in ('success_rate', 'picked_rate', 'found_rate'):
                    result[metric] = statistics.mean(values) if values and all(v is not None for v in values) else None
                result[metric + '_defined_cases'] = len(known)
            aggregate.append(result)
    pairs, case_pairs = [], []
    for decoding in ('greedy', 'sampled'):
        for left, right in [('G', 'U'), ('G1', 'U'), ('G', 'G1')]:
            if not {left, right} <= set(identity['modes']):
                continue
            for metric in rate_keys:
                differences, included, excluded = [], [], []
                for case in identity['cases']:
                    cells = {r['mode']: r for r in per_case if r['case_id'] == case['case_id'] and r['decoding'] == decoding}
                    x, y = cells[left][metric], cells[right][metric]
                    delta = x - y if x is not None and y is not None else None
                    case_pairs.append(dict(case_id=case['case_id'], decoding=decoding,
                                           comparison=left + '-' + right, metric=metric, difference=delta))
                    if delta is None:
                        excluded.append(case['case_id'])
                    else:
                        differences.append(delta)
                        included.append(case['case_id'])
                pairs.append(dict(decoding=decoding, comparison=left + '-' + right, metric=metric,
                    difference=statistics.mean(differences) if differences else None,
                    ci95=cluster_bootstrap(differences), included_cases=included, excluded_cases=excluded))
    by_cell = {(r['case_id'], r['mode'], r['decoding'], r['seed']): r for r in rows}
    action_changes = []
    for row in rows:
        baseline = by_cell.get((row['case_id'], 'U', row['decoding'], row['seed']))
        if not baseline or row['void'] or baseline['void']:
            continue
        a = {s['step']: s['executed_action'] for s in row['steps'] if s.get('execution_recorded')}
        b = {s['step']: s['executed_action'] for s in baseline['steps'] if s.get('execution_recorded')}
        shared = a.keys() & b.keys()
        action_changes.append(dict(case_id=row['case_id'], mode=row['mode'], decoding=row['decoding'], seed=row['seed'],
            comparable_turns=len(shared), changed_actions=sum(a[i] != b[i] for i in shared),
            action_change_rate=sum(a[i] != b[i] for i in shared) / len(shared) if shared else None))
    return dict(schema_version=1, run_sha256=plan['run_sha256'], dry_run=identity['dry_run'],
        per_case=per_case, aggregate=aggregate, paired_differences=pairs, per_case_paired_differences=case_pairs,
        action_changes_vs_U=action_changes, missing_records=sorted(expected.keys() - seen),
        void_records=[r['run_key'] for r in rows if r['void']],
        bootstrap={'unit': 'case', 'resamples': 10000, 'numpy_seed': 0},
        limitations='TRAIN diagnostic only. Greedy and sampled separate. Void/missing rates are unknown; no-guidance follow rates are undefined. Complete paired cases only; synthetic runs establish no scientific result.')


def summarize(args):
    root, output = Path(args.rollouts), Path(args.output)
    plan = read_json(root / 'plan.json')
    rows = [read_json(p) for p in sorted((root / 'records').glob('*.json'))]
    result = summarize_records(plan, rows)
    if plan['identity'].get('eval_adapter'):
        from repair.adapter_eval import NoTeacherBudget, assert_no_teacher_calls
        assert_no_teacher_calls(rows, root)
        result['global_budget'] = NoTeacherBudget(plan['identity']['caps']).summary()
    else:
        result['global_budget'] = SharedTeacherBudget(plan['identity']['budget_ledger'], **plan['identity']['caps']).summary()
    usages = [usage_summary(p) for p in sorted((root / 'calls').glob('*/*/usage.jsonl'))]
    result['all_attempt_usage'] = {k: sum(u['totals'][k] for u in usages) for k in ('input_tokens', 'output_tokens', 'reasoning_tokens')}
    result['unknown_usage_attempts'] = sum(u['unknown_usage_attempts'] for u in usages)
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / 'summary.json', result, exclusive=True)
    columns = ['mode', 'decoding', 'success_rate', 'picked_rate', 'found_rate', 'teacher_step_share', 'follow_rate']
    text = '# Joint guidance diagnostic\n\n' + result['limitations'] + '\n\n'
    text += markdown_table(result['aggregate'], columns) + '\n\n'
    text += markdown_table(result['per_case'], ['case_id', *columns, 'expected', 'valid', 'void', 'missing']) + '\n\n'
    text += markdown_table(result['paired_differences'], ['decoding', 'comparison', 'metric', 'difference', 'ci95', 'excluded_cases'])
    text += '\n\nCase-cluster bootstrap: 10,000 resamples, numpy seed 0.\n'
    atomic_write(output / 'summary.md', text.encode(), exclusive=True)
    return 0


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest='command', required=True)
    r = sub.add_parser('run')
    r.add_argument('--spec', required=True)
    r.add_argument('--from-start', action='store_true', help='Use the spec task identities but reset to turn zero')
    r.add_argument('--eval-adapter', action='store_true',
                   help='Evaluate a LoRA-gated adapter without hints or teacher calls; requires --modes U')
    r.add_argument('--review-template', choices=['joint-review/1', 'joint-review/2'], default=TEMPLATE_VERSION)
    r.add_argument('--modes', nargs='+', choices=['U', 'G', 'G1'], default=['U', 'G', 'G1'])
    r.add_argument('--seeds', nargs='*', type=int, default=list(range(8)), help='Sampled seeds; greedy always runs')
    r.add_argument('--workers', type=int, default=1)
    r.add_argument('--budget-ledger', help='Required unless --eval-adapter; untouched in adapter evaluation')
    r.add_argument('--max-teacher-input-tokens', type=int, default=5_000_000)
    r.add_argument('--max-teacher-output-tokens', type=int, default=500_000)
    r.add_argument('--resume', action='store_true')
    r.add_argument('--dry-run', action='store_true')
    r.add_argument('--config', default='configs/base_e4b.yaml')
    for name in ('endpoint', 'server-provenance', 'lora-check', 'model-name', 'data-root'):
        r.add_argument('--' + name)
    s = sub.add_parser('summarize')
    s.add_argument('--rollouts', required=True)
    for parser in (r, s):
        parser.add_argument('--output', required=True)
    args = p.parse_args(argv)
    if args.command == 'run' and not args.eval_adapter and not args.budget_ledger:
        r.error('the following arguments are required: --budget-ledger')
    return {'run': run, 'summarize': summarize}[args.command](args)


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(json.dumps({'status': 'error', 'error_type': type(exc).__name__}))
        raise SystemExit(1) from None
