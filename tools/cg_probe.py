"""Pre-registered condition-guidance probe; no training or E1 repair inputs."""
import argparse
from concurrent.futures import ProcessPoolExecutor
from copy import deepcopy
from dataclasses import replace
from functools import partial
import csv
import json
import multiprocessing
from pathlib import Path
import statistics

from repair.artifacts import provenance, read_json, run_manifest
from repair.ledger import summarize as summarize_budget
from repair.guidance import (GROUPS, BRANCH_INDEX, OUTPUT_CAP, GuidancePrompt, GuidanceSession,
    GlobalTeacherBudget, BudgetLedger, TrajectoryRecord, DryTeacher, DryEnvironment, DryPolicy,
    approve_hints, approved_cases, branch_texts, cluster_bootstrap, decision_rule,
    execute_rollout, guidance_prompt, make_guidance_teacher, prefix_sha, receptacles,
    rollout_key, target_pattern, validate_guidance)
from sed.config import load_config
from sed.manifest import authorize_task, load_frozen, resolve_task
from sed.provenance import code_hashes
from sed.react import action_outcome
from sed.storage import atomic_write, digest, file_hash, locked, write_json
from sed.visibility import visible_observation


def select_groups(text):
    groups = text.split(',')
    if not groups or len(set(groups)) != len(groups) or any(g not in GROUPS for g in groups):
        raise ValueError('--tasks must be a unique nonempty subset of the seven preregistered group names')
    return groups


def load_cases(failures_path, coverage_root, groups, config, *, dry_run=False):
    """Import runner transitions with the collector's checks/accounting semantics.

    Unlike repair_collect's full replay preflight, this CPU-only import does not
    open games. Real replay is mandatory in every rollout, before any student call.
    """
    root = Path(coverage_root).resolve()
    failures = read_json(failures_path)
    rows = failures.get('failures', failures.get('failure_list'))
    if not isinstance(rows, list) or digest(rows) != failures.get('list_sha256', failures.get('failure_list_sha256')):
        raise ValueError('Coverage failure-list hash differs')
    summary = read_json(Path(failures_path).parent / 'summary.json')
    if (summary['failure_list'] != rows or summary['failure_list_sha256'] != digest(rows)
            or not summary['branch_decision_actionable']):
        raise ValueError('Failure list differs from actionable coverage summary')
    frozen = None if dry_run else load_frozen(config['paths']['manifest'], config['paths']['split_freeze'])
    result = []
    for group in groups:
        matches = [r for r in rows if r.get('group_id') == 'train/' + group]
        if len(matches) != 1:
            raise ValueError('Requested group missing or duplicated in failures: ' + group)
        row = matches[0]
        if row.get('attempt') != 1 or row.get('decoding') != 'greedy' or row.get('sampling_seed') is not None:
            raise ValueError('CG cases require attempt-01 greedy failures')
        if 'attempt-01-greedy' not in Path(row['episode_dir']).parts:
            raise ValueError('CG episode must be from attempt-01-greedy')
        if Path(row['task_id']).parent.as_posix() != 'train/' + group:
            raise ValueError('Failure task/group mismatch')
        directory = (root / row['episode_dir']).resolve()
        if not directory.is_relative_to(root):
            raise ValueError('Coverage episode escapes coverage-root')
        hashes = {}
        for name, key in [('episode.json', 'episode_sha256'), ('events.jsonl', 'events_sha256'),
                          ('provenance.json', 'provenance_sha256')]:
            path = directory / name
            if not path.resolve().is_relative_to(root) or file_hash(path) != row[key]:
                raise ValueError('Coverage episode evidence changed')
            hashes[str(path)] = row[key]
        episode = read_json(directory / 'episode.json')
        if (episode['task_id'] != row['task_id'] or episode.get('episode_id') != row['episode_id']
                or episode['success'] is not False or episode.get('split') != 'train' or episode.get('sealed')
                or not episode['complete'] or episode['status'] != 'ok'
                or episode['seed'] != config['environment']['seed']
                or episode['max_episode_steps'] != 40 or config['environment']['max_episode_steps'] != 40):
            raise ValueError('Frozen failure identity/outcome/config differs')
        task = (resolve_task(*frozen, row['task_id']) if frozen else
                {'task_id': row['task_id'], 'split': 'train', 'sealed': False, 'group_id': 'train/' + group})
        authorize_task(task, 'generation')
        if frozen and (task['source_split'] != 'train' or episode['game_sha256'] != task['game_sha256']):
            raise ValueError('Frozen task/game identity differs')
        steps = []
        for step in episode['steps']:
            steps.append({**step, 'feedback_source': 'analytic_verifier' if step.get('action_outcome') == 'parser_error'
                          else 'real_environment'})
        reason = ('CONTEXT_LIMIT' if episode.get('termination_reason') == 'context_overflow' else
                  'STEP_LIMIT' if episode.get('truncated') else 'FAILED')
        used = sum(s.get('environment_steps', 1) for s in steps)
        original = TrajectoryRecord(row['task_id'], episode['initial'], steps, False, reason,
            token_costs={k: episode.get('metrics', {}).get(k) for k in
                         ('prompt_tokens', 'completion_tokens', 'thinking_tokens', 'thinking_chars')},
            environment_steps=used, environment_steps_confirmed=used,
            game_sha256=episode['game_sha256'], env_seed=episode['seed'], max_steps=40,
            error_type='context_overflow' if reason == 'CONTEXT_LIMIT' else None)
        if len(steps) < 11:
            raise ValueError('Original trajectory needs at least 11 steps')
        previous = original.initial['visible']
        target = group.split('-')[1].lower()
        for step in original.steps[:10]:
            action, outcome = action_outcome(step['action_raw'], previous['admissible_commands'])
            if (outcome == 'parser_error' or step.get('action_outcome') != outcome
                    or step.get('action_parsed') != action or step['visible'] is None
                    or step.get('terminated') or step.get('truncated')):
                raise ValueError('Invalid, parser-error, or terminal prefix')
            previous = step['visible']
        if any(target_pattern(target).search(s['visible']['observation']) for s in
               [original.initial, *original.steps[:10]]):
            raise ValueError('Target already observed before branch')
        receptacles(original.initial['visible']['observation'])
        task = {**task, 'instruction': guidance_prompt(original).payload['task'],
                'parent_seed_ids': [row['episode_id']]}
        result.append({'task': task, 'original': original, 'group': group, 'target': target,
            'source': {'episode_id': row['episode_id'], 'evidence_sha256': hashes,
                       'environment_revision': episode.get('environment_revision'),
                       'provenance': read_json(directory / 'provenance.json')}})
    return result


def input_paths(args, config):
    paths = [__file__, args.config, args.failures, Path(args.failures).parent / 'summary.json']
    if not args.dry_run:
        paths.extend(config['paths'][k] for k in ('manifest', 'split_freeze'))
    return paths


def write_tsv(path, rows, fields):
    with Path(path).open('x', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter='\t', extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)


def hints(args):
    config = load_config(args.config)
    cases = load_cases(args.failures, args.coverage_root, select_groups(args.tasks), config, dry_run=args.dry_run)
    if args.branch_index != BRANCH_INDEX:
        raise ValueError('Pre-registered branch index is 10')
    if not (0 <= args.max_teacher_input_tokens <= 200000 and 0 <= args.max_teacher_output_tokens <= 60000):
        raise ValueError('CG global caps cannot exceed 200000 input / 60000 output')
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    prov = provenance(output, config, command='cg_probe hints', inputs=input_paths(args, config),
        details={'dry_run': args.dry_run, 'template_sha256': GuidancePrompt({}).template_sha256,
                 'output_cap_including_reasoning': OUTPUT_CAP})
    budget = GlobalTeacherBudget(output / 'global_teacher_budget.jsonl',
        input_tokens=args.max_teacher_input_tokens, output_tokens=args.max_teacher_output_tokens)
    ledger = BudgetLedger(output / 'budget.jsonl')
    teacher = DryTeacher() if args.dry_run else make_guidance_teacher(config, output / 'teacher', run_id=prov['run_id'])
    rows, reviews = [], []
    for case in cases:
        original, task = case['original'], case['task']
        value, attempts = GuidanceSession(teacher, task, ledger, budget).request(original)
        row = {'task_id': original.task_id, 'group': case['group'], 'target': case['target'],
            'branch_index': 10, 'prefix_sha': prefix_sha(original),
            'admissible_now': original.steps[9]['visible']['admissible_commands'],
            'teacher_json': value, 'teacher_raw_json': next((a['raw_output'] for a in reversed(attempts) if 'raw_output' in a), None),
            'validation_log': attempts, 'texts': branch_texts(value),
            'status': 'valid_pending_review' if value else 'missing', 'source': case['source'],
            'cost_record': summarize_budget([r for r in ledger.rows() if r['task_id'] == original.task_id]),
            'global_budget_after_case': budget.summary()}
        rows.append(row)
        reviews.append({'task_id': original.task_id, 'plan': value['student_plan'] if value else '',
            'conditions': json.dumps(value['conditions'], ensure_ascii=False) if value else '',
            'C': row['texts']['C'], 'D': row['texts']['D'],
            'validator_result': 'valid' if value else json.dumps(attempts[-1].get('reason')),
            'decision': '', 'note': ''})
        # Durable partial evidence survives a later transport failure or interruption.
        write_json(output / 'cases' / (digest(original.task_id) + '.json'), row, exclusive=True)
    result = {'schema_version': 1, 'kind': 'condition_guidance', 'dry_run': args.dry_run,
        'template_sha256': GuidancePrompt({}).template_sha256, 'cases': rows,
        'global_budget': budget.summary(), 'budget': ledger.summary(),
        'unique_instances': len(rows), 'task_groups': len({r['group'] for r in rows}),
        'training_exposure': 0}
    write_json(output / 'hints.json', result, exclusive=True)
    write_tsv(output / 'review.tsv', reviews,
              ['task_id', 'plan', 'conditions', 'C', 'D', 'validator_result', 'decision', 'note'])
    models = [{k: a['metadata'].get(k) for k in ('returned_model_id', 'timestamp', 'usage')}
              for r in rows for a in r['validation_log'] if 'metadata' in a]
    write_json(output / 'run_manifest.json', run_manifest(prov, checkpoint=None, tokenizer=None,
        tasks=[c['task'] for c in cases], seeds={'environment': config['environment']['seed']},
        teacher_alias='gpt-6-luna', teacher_results=models, global_budget=budget.summary(),
        template_sha256=result['template_sha256'], dry_run=args.dry_run,
        hints_sha256=file_hash(output / 'hints.json'), training_exposure=0), exclusive=True)
    print(json.dumps({'hints': str(output / 'hints.json'), 'missing': sum(r['status'] == 'missing' for r in rows),
                      'budget': budget.summary()}))
    return int(budget.stop_reason is not None)


def serving_spec(args, config, cases):
    from students.vllm_student import Student, context_options
    from tools.check_lora_serving import validate_report
    server, report = read_json(args.server_provenance), read_json(args.lora_check)
    config['server_provenance'], config['lora_equivalence'] = server, report
    model = config['models']['student']
    spec = dict(endpoint=args.endpoint, model=args.model_name, revision=model['revision'],
                enable_thinking=False, max_tokens=1024, **context_options(config))
    Student(log_dir=Path(args.output) / 'unused', **spec)
    validate_report(report, server, spec['model'], spec['endpoint'])
    adapter = report['identity']['adapter']
    if (adapter['base_model'] != model['requested_repo'] or adapter['revision'] != model['revision']
            or args.model_name != 'e4b-tsft-k20-e32-lr5e-5'):
        raise ValueError('CG requires the configured theta0 adapter e4b-tsft-k20-e32-lr5e-5')
    # Equality of weights/config binds theta0 across relocated serving hosts.
    for case in cases:
        old = case['source']['provenance']['config']['server_provenance']['config']['server']['lora_adapters']
        matches = [a for a in old if a['name'] == args.model_name]
        if len(matches) != 1 or any(matches[0][k] != adapter[k] for k in
                ('base_model', 'revision', 'adapter_weights_sha256', 'adapter_config_sha256')):
            raise ValueError('Served theta0 differs from original coverage adapter')
    return spec


def _student(log_dir, *, spec, decoding, seed, task_id):
    from students.sampled_student import SampledStudent
    from students.vllm_student import Student
    if decoding == 'sampled':
        return SampledStudent(**spec, sampling_seed=seed, task_id=task_id, log_dir=log_dir)
    return Student(**spec, log_dir=log_dir)


def _job(job):
    case, original_json, branch, decoding, seed, config, spec, output, identity, dry_run = job
    original = TrajectoryRecord.from_json(original_json)
    if dry_run:
        # A visibly synthetic fixture, not a claim to reproduce original audit state.
        original = replace(original,
            initial={**original.initial, 'audit_sha256': digest({'won': False})},
            steps=[{**s, 'audit_sha256': digest({'won': False})} for s in original.steps])
        env_factory, policy_factory = partial(DryEnvironment, original), DryPolicy
    else:
        from harness.alfworld_env import Environment
        env_factory = partial(Environment, config['environment']['data_root'], split='train',
            seed=original.env_seed, max_episode_steps=40)
        policy_factory = partial(_student, spec=spec, decoding=decoding, seed=seed, task_id=original.task_id)
    return execute_rollout(case, original, branch=branch, decoding=decoding, seed=seed,
        env_factory=env_factory, policy_factory=policy_factory, output=output, run_sha256=identity)


def rollouts(args):
    if args.workers < 1 or (args.limit_cases is not None and args.limit_cases < 1):
        raise ValueError('workers and limit-cases must be positive')
    hints_document, cases = approved_cases(args.hints, args.approval)
    if hints_document['dry_run'] != args.dry_run:
        raise ValueError('Synthetic hints can only be used with synthetic rollouts')
    if args.limit_cases:
        cases = cases[:args.limit_cases]
    config = load_config(args.config)
    if args.data_root:
        config['environment']['data_root'] = args.data_root
    originals = load_cases(args.failures, args.coverage_root, [c['group'] for c in cases], config, dry_run=args.dry_run)
    for case, source in zip(cases, originals):
        original = source['original']
        if (case['task_id'] != original.task_id or case['branch_index'] != 10 or case['target'] != source['target']
                or case['prefix_sha'] != prefix_sha(original)
                or case['admissible_now'] != original.steps[9]['visible']['admissible_commands']
                or case['source'] != source['source']):
            raise ValueError('Hints differ from coverage evidence')
        if case['teacher_json'] is not None:
            validate_guidance(json.dumps(case['teacher_json']), original)
        if case['texts'] != branch_texts(case['teacher_json']):
            raise ValueError('Branch text differs from validated teacher JSON')
    if not args.dry_run and not all((args.endpoint, args.server_provenance, args.lora_check, args.model_name,
                                     config['environment']['data_root'])):
        raise ValueError('Real rollouts require endpoint, server-provenance, lora-check, model-name and data-root')
    spec = {} if args.dry_run else serving_spec(args, config, originals)
    inputs = [*input_paths(args, config), args.hints, args.approval]
    if not args.dry_run:
        inputs.extend([args.server_provenance, args.lora_check])
    identities = {'hints_sha256': file_hash(args.hints), 'approval_sha256': file_hash(args.approval),
        'failures_sha256': file_hash(args.failures), 'config_sha256': digest(config), 'spec': spec,
        'dry_run': args.dry_run, 'case_ids': [c['task_id'] for c in cases],
        'originals_sha256': digest([c['original'].to_json() for c in originals]),
        'code_sha256': {**code_hashes(Path(__file__).resolve().parents[1]),
                       **{str(p): file_hash(p) for p in sorted(Path('repair').glob('*.py'))}},
        'input_sha256': {str(p): file_hash(p) for p in inputs},
        'branch_index': 10, 'H': 40, 'sampling_seeds': list(range(8)), 'temperature': .7, 'top_p': .95}
    run_sha = digest(identities)
    output = Path(args.output)
    if args.resume:
        if not output.is_dir() or read_json(output / 'plan.json')['identity'] != identities:
            raise ValueError('Resume requires an existing run with identical inputs, config, and code')
    else:
        output.mkdir(parents=True, exist_ok=False)
        prov = provenance(output, config, command='cg_probe rollouts', inputs=inputs,
                          details={'run_sha256': run_sha, 'evaluation_only': True, 'dry_run': args.dry_run})
        write_json(output / 'run_manifest.json', run_manifest(prov, checkpoint=spec, tokenizer=None,
            tasks=[c['task'] for c in originals], seeds={'sampling': list(range(8)), 'environment': config['environment']['seed']},
            run_sha256=run_sha, evaluation_only=True, dry_run=args.dry_run, training_exposure=0, teacher_calls=0), exclusive=True)
        write_json(output / 'plan.json', {'identity': identities, 'run_sha256': run_sha,
            'cases': [{k: c[k] for k in ('task_id', 'group', 'branches', 'cd_status')} for c in cases]}, exclusive=True)
    jobs = [(case, source['original'].to_json(), branch, decoding, seed, config, spec, str(output), run_sha, args.dry_run)
            for case, source in zip(cases, originals) for branch in case['branches']
            for decoding, seed in [('greedy', None), *[('sampled', s) for s in range(8)]]]
    # Workers skip immutable existing records under a per-key lock. Different
    # launches cannot execute a shared key concurrently.
    with locked(output / 'run.lock'):
        if args.workers == 1:
            rows = [_job(j) for j in jobs]
        else:
            with ProcessPoolExecutor(max_workers=args.workers, mp_context=multiprocessing.get_context('spawn')) as pool:
                rows = list(pool.map(_job, jobs))
        report = {'expected': len(jobs), 'recorded': len(rows), 'void': sum(r['void'] for r in rows),
                  'dry_run': args.dry_run, 'run_sha256': run_sha}
        write_json(output / 'completion.json', report)
    print(json.dumps(report))
    return int(report['void'] > 0)


def metric_table(rows, expected):
    valid = [r for r in rows if not r['void']]
    metrics = [r['metrics'] for r in valid]
    complete = len(valid) == expected and expected > 0
    def rate(key):
        values = [m[key] for m in metrics]
        return sum(v is True for v in values) / expected if complete and all(type(v) is bool for v in values) else None
    found_steps = [m['steps_to_find'] for m in metrics if m['found']]
    suggestions = [m['suggestion_reveals_target'] for m in metrics if m['suggestion_reveals_target'] is not None]
    return {'expected': expected, 'recorded': len(rows), 'valid': len(valid), 'void': len(rows) - len(valid),
        'missing': expected - len(rows), 'found_rate': rate('found'),
        'median_steps_to_find': statistics.median(found_steps) if found_steps else None,
        'picked_rate': rate('picked'), 'success_rate': rate('success'), 'follow_rate': rate('followed_suggestion'),
        'suggestion_executed': len(suggestions),
        'suggestion_reveals_target_rate': sum(suggestions) / len(suggestions) if suggestions else None,
        'inadmissible_count': sum(m['inadmissible_count'] for m in metrics),
        'revisits_of_cleared': sum(m['revisits_of_cleared'] for m in metrics)}


def summarize_records(plan, rows):
    expected = {rollout_key(c['task_id'], b, d, s): (c['task_id'], b, d, s)
                for c in plan['cases'] for b in c['branches']
                for d, s in [('greedy', None), *[('sampled', s) for s in range(8)]]}
    indexed = {}
    for row in rows:
        key = row['rollout_key']
        if (key not in expected or key in indexed or row['run_sha256'] != plan['run_sha256']
                or expected[key] != (row['case'], row['branch'], row['decoding'], row['seed'])):
            raise ValueError('Unexpected, duplicate, or foreign rollout record')
        indexed[key] = row
    per_case = []
    for case in plan['cases']:
        for branch in 'ABCD':
            for decoding, count in [('greedy', 1), ('sampled', 8)]:
                current = [r for r in rows if (r['case'], r['branch'], r['decoding']) == (case['task_id'], branch, decoding)]
                per_case.append({'task_id': case['task_id'], 'group': case['group'], 'branch': branch,
                    'decoding': decoding, 'cd_status': case['cd_status'],
                    'scheduled': branch in case['branches'], **metric_table(current, count)})
    aggregate = []
    for branch in 'ABCD':
        for decoding in ('greedy', 'sampled'):
            current = [r for r in rows if (r['branch'], r['decoding']) == (branch, decoding)]
            cases = [c for c in per_case if (c['branch'], c['decoding']) == (branch, decoding)]
            table = metric_table(current, len(cases) * (8 if decoding == 'sampled' else 1))
            # Primary rates are explicitly the mean of per-case rates. Missing
            # cells remain unknown, rather than failures or smaller denominators.
            for key in ('found_rate', 'picked_rate', 'success_rate', 'follow_rate'):
                values = [c[key] for c in cases]
                table[key] = statistics.mean(values) if values and all(v is not None for v in values) else None
            aggregate.append({'branch': branch, 'decoding': decoding, **table})
    pairs = []
    for metric in ('found_rate', 'picked_rate', 'success_rate'):
        for left, right in [('C', 'B'), ('D', 'C'), ('B', 'A'), ('C', 'A'), ('D', 'A'), ('D', 'B')]:
            # D-B is needed for section 8's conclusion mapping; marked secondary.
            xs, ys, included, excluded = [], [], [], []
            for case in plan['cases']:
                entries = {c['branch']: c for c in per_case if c['task_id'] == case['task_id'] and c['decoding'] == 'sampled'}
                x, y = entries[left][metric], entries[right][metric]
                if x is None or y is None:
                    excluded.append(case['task_id'])
                else:
                    xs.append(x)
                    ys.append(y)
                    included.append(case['task_id'])
            differences = [x - y for x, y in zip(xs, ys)]
            pairs.append({'comparison': left + '-' + right, 'metric': metric,
                'difference': statistics.mean(differences) if differences else None,
                'ci95': cluster_bootstrap(differences), 'cases': included, 'excluded_cases': excluded,
                'cases_x_gt_y': sum(x > y for x, y in zip(xs, ys)),
                'cases_x_lt_y': sum(x < y for x, y in zip(xs, ys)),
                'decision': decision_rule(xs, ys, left, right), 'additional_mapping_comparison': (left, right) == ('D', 'B')})
    decisions = {p['comparison']: p for p in pairs if p['metric'] == 'found_rate'}
    full = (len(plan['cases']) == 7 and all(not p['excluded_cases'] for p in decisions.values())
            and not plan['identity']['dry_run'])
    better = lambda pair: decisions[pair]['decision'] != 'no clear difference'
    mapping = ('Supports the hypothesis that identifying missing conditions helps the student use existing ability.'
               if better('C-B') else
               'Prioritize learning information-gathering action selection.' if better('D-C') else
               'Report first; a teacher closed-loop follow-up requires user authorization and is outside this probe.'
               if not better('D-B') else 'No preregistered conclusion mapping applies.') if full else 'Incomplete or synthetic probe; no mechanism conclusion.'
    diagnostics = []
    by_cell = {(r['case'], r['branch'], r['decoding'], r['seed']): r for r in rows}
    for row in rows:
        base = by_cell.get((row['case'], 'A', row['decoding'], row['seed']))
        diagnostics.append({'case': row['case'], 'branch': row['branch'], 'decoding': row['decoding'], 'seed': row['seed'],
            'first_action_differs_from_A': (row['metrics']['first_post_branch_action'] != base['metrics']['first_post_branch_action'])
                if base and not row['void'] and not base['void'] else None,
            'metrics': row['metrics'], 'void_reason': row['void_reason']})
    return {'schema_version': 1, 'run_sha256': plan['run_sha256'], 'dry_run': plan['identity']['dry_run'],
        'aggregate': aggregate, 'per_case': per_case, 'sampled_pairs': pairs, 'diagnostics': diagnostics,
        'decision_scope': 'full_preregistered_probe' if full else 'available_complete_pairs_only_diagnostic',
        'conclusion_mapping': mapping, 'bootstrap': {'unit': 'case', 'resamples': 10000, 'numpy_seed': 0},
        'expected_rollouts': len(expected), 'missing_rollouts': sorted(set(expected) - set(indexed)),
        'void_rollouts': [r['rollout_key'] for r in rows if r['void']],
        'limitations': 'Seven TRAIN search failures; C/B length confound; one injection; greedy is separate; no significance claim.'}


def markdown_table(rows, columns):
    def cell(value):
        if value is None:
            return 'unknown'
        if isinstance(value, float):
            return f'{value:.4f}'
        return str(value).replace('|', '\\|').replace('\n', ' ')
    return '\n'.join(['| ' + ' | '.join(columns) + ' |', '| ' + ' | '.join(['---'] * len(columns)) + ' |',
                      *['| ' + ' | '.join(cell(r.get(k)) for k in columns) + ' |' for r in rows]])


def summarize(args):
    root = Path(args.rollouts)
    plan = read_json(root / 'plan.json')
    rows = [read_json(p) for p in sorted((root / 'records').glob('*.json'))]
    result = summarize_records(plan, rows)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / 'summary.json', result, exclusive=True)
    fields = ['task_id', 'branch', 'decoding', 'cd_status', 'scheduled', 'expected', 'recorded', 'valid', 'void', 'missing',
              'found_rate', 'median_steps_to_find', 'picked_rate', 'success_rate', 'follow_rate']
    write_tsv(output / 'per_case.tsv', result['per_case'], fields)
    columns = ['branch', 'decoding', 'valid', 'expected', 'found_rate', 'median_steps_to_find', 'picked_rate', 'success_rate', 'follow_rate']
    text = '# Condition-guidance probe\n\n' + result['conclusion_mapping'] + '\n\n' + result['limitations']
    for decoding in ('sampled', 'greedy'):
        text += '\n\n## ' + decoding + '\n\n' + markdown_table([r for r in result['aggregate'] if r['decoding'] == decoding], columns)
    text += '\n\n## Sampled paired comparisons\n\nScope: ' + result['decision_scope'] + '. Case cluster bootstrap: 10,000 resamples, numpy seed 0.\n\n'
    text += markdown_table(result['sampled_pairs'], ['comparison', 'metric', 'difference', 'ci95', 'cases_x_gt_y', 'cases_x_lt_y', 'decision'])
    text += '\n\n## Per case\n\n' + markdown_table(result['per_case'], fields)
    text += '\n\nUnknown rates indicate missing/void rollouts or unavailable C/D; they are never scored as failures.\n'
    atomic_write(output / 'summary.md', text.encode(), exclusive=True)
    print('\n'.join(p['comparison'] + ': ' + p['decision'] for p in result['sampled_pairs'] if p['metric'] == 'found_rate'))
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    h = sub.add_parser('hints')
    h.add_argument('--tasks', required=True, help='Comma-separated preregistered group names; subsets allowed')
    h.add_argument('--branch-index', type=int, default=10)
    h.add_argument('--max-teacher-input-tokens', type=int, default=200000)
    h.add_argument('--max-teacher-output-tokens', type=int, default=60000)
    r = sub.add_parser('rollouts')
    r.add_argument('--hints', required=True)
    r.add_argument('--approval', required=True)
    for name in ('endpoint', 'server-provenance', 'lora-check', 'model-name', 'data-root'):
        r.add_argument('--' + name)
    r.add_argument('--workers', type=int, default=8)
    r.add_argument('--limit-cases', type=int)
    r.add_argument('--resume', action='store_true', help='Resume only the identical frozen run; retain old call logs')
    for p in (h, r):
        p.add_argument('--failures', required=True)
        p.add_argument('--coverage-root', required=True)
        p.add_argument('--config', required=True)
        p.add_argument('--output', required=True)
        p.add_argument('--dry-run', action='store_true')
    a = sub.add_parser('approve')
    a.add_argument('--hints', required=True)
    a.add_argument('--decisions', required=True)
    s = sub.add_parser('summarize')
    s.add_argument('--rollouts', required=True)
    s.add_argument('--output', required=True)
    args = parser.parse_args(argv)
    if args.command == 'approve':
        print(approve_hints(args.hints, args.decisions))
        return 0
    return {'hints': hints, 'rollouts': rollouts, 'summarize': summarize}[args.command](args)


if __name__ == '__main__':
    raise SystemExit(main())
