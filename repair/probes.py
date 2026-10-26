"""Direct behavior observations on a served adapter; no teacher or training."""
from pathlib import Path
import csv

from repair.align import canonical_action
from repair.artifacts import provenance, run_manifest, write_sealed, checkpoint_files
from repair.export import export_branch, require_approved
from repair.branches import collection_branch, validate_base_serving, ADAPTER_KEYS
from repair.execute import execute_actions
from repair.ledger import BudgetLedger
from repair.minimality import same_reset
from repair.records import RepairRecord, TrajectoryRecord
from sed.actions import parse_action, ActionParseError
from sed.storage import digest, write_json, file_hash

SAMPLING_SEEDS = tuple(range(8))


def probe_point(case):
    repair = RepairRecord.from_json(case['repair'])
    original, near = repair.original_trajectory, repair.executed_trajectory
    if repair.repair_status != 'VERIFIED' or near is None or original.success is not False:
        raise ValueError('Probe requires a verified repair of a real failure')
    if original.task_id != case['task_id'] or near.task_id != original.task_id:
        raise ValueError('Probe case task differs from execution evidence')
    if any(s['action_parsed'] is None for s in original.steps):
        raise ValueError('Repair probe editing protocol excludes parser-feedback originals')
    point = 0
    while (point < min(len(original.steps), len(near.steps)) and
           canonical_action(original.steps[point]['action_parsed']) == canonical_action(near.steps[point]['action_parsed'])):
        point += 1
    if point >= len(near.steps) or point >= original.max_steps:
        raise ValueError('No witnessed next decision at a changed action')
    candidates = [near, *[TrajectoryRecord.from_json(r['execution']) for r in repair.verification_results],
                  *[TrajectoryRecord.from_json(r['execution']) for r in case.get('candidates', []) +
                    case.get('far_candidates', []) if r.get('execution')]]
    keys = ('action_parsed', 'visible', 'reward', 'terminated', 'truncated', 'audit_sha256')
    witnesses, excluded = {}, []
    for candidate in candidates:
        valid = (candidate.success is True and same_reset(original, candidate) and
                 candidate.max_steps == original.max_steps and len(candidate.steps) > point and
                 candidate.steps[point]['action_parsed'] is not None and
                 all(all(a.get(k) == b.get(k) for k in keys)
                     for a, b in zip(original.steps[:point], candidate.steps[:point])))
        if valid:
            action = canonical_action(candidate.steps[point]['action_parsed'])
            witnesses.setdefault(action, []).append(candidate.trajectory_id)
        else:
            excluded.append(candidate.trajectory_id)
    if not witnesses:
        raise ValueError('No successful candidate preserves the original prefix')
    return original, point, {a: sorted(set(ids)) for a, ids in sorted(witnesses.items())}, sorted(set(excluded))


def probe_case(case, *, env_factory, policy_factory, output, context=None):
    """policy_factory(seed, task_id, log_dir) -> (public_history, visible) -> reply."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    context = context or {}
    ledger = BudgetLedger(output / 'budget.jsonl', context=context)
    original, point, witnesses, excluded = probe_point(case)
    prefix = [s['action_raw'] for s in original.steps[:point]]
    samples = []
    def observe(history, visible):
        for seed in SAMPLING_SEEDS:
            row = {**context, 'seed': seed, 'request_seed': int(digest([seed, original.task_id, point])[:8], 16)}
            try:
                reply = policy_factory(seed, original.task_id, output / f'decision-{seed}')(history, visible)
                try:
                    action = canonical_action(parse_action(reply))
                except ActionParseError:
                    action = None
                row.update(reply=reply, action=action, hit=action in witnesses, status='ok')
            except Exception as exc:
                row.update(action=None, hit=None, status='error', error_type=type(exc).__name__)
            samples.append(row)
    replay = execute_actions(env_factory, original.task_id, prefix, original.max_steps,
        prefix_reference=original, observe_prefix=observe, ledger=ledger, interaction_kind='evaluation_steps')
    usable = replay.termination_reason != 'ENV_ERROR' and len(samples) == 8 and all(r['status'] == 'ok' for r in samples)
    continuations = []
    for seed in SAMPLING_SEEDS[:2]:
        result = execute_actions(env_factory, original.task_id, prefix, original.max_steps,
            prefix_reference=original, continuation=policy_factory(seed, original.task_id, output / f'continuation-{seed}'),
            ledger=ledger, interaction_kind='evaluation_steps')
        # A location in observed execution, never a claim about a hidden cause.
        rejected = next((i for i, s in enumerate(result.steps) if i >= point and
                         s.get('action_outcome') == 'parser_error'), None)
        continuations.append({**context, 'seed': seed, 'success': result.success,
            'new_failure_position': None if result.success is not False else rejected if rejected is not None else len(result.steps),
            'failure_position_rule': 'first parser error after prefix, else terminal boundary (zero-based); causal attribution unknown',
            'prefix_turns': point, 'remaining_H': original.max_steps - point, 'execution': result.to_json()})
    return {**context, 'task_id': original.task_id, 'decision_index': point, 'H': original.max_steps,
        'A_plus': sorted(witnesses), 'witnesses': witnesses, 'excluded_witnesses': excluded,
        'decision_replay': replay.to_json(), 'samples': samples,
        'm': sum(r['hit'] for r in samples) / 8 if usable else None,
        'continuations': continuations, 'budget': ledger.summary()}


def served_policy_factory(spec):
    from students.sampled_student import SampledStudent
    def factory(seed, task_id, log_dir):
        policy = SampledStudent(**spec, sampling_seed=seed, task_id=task_id, log_dir=log_dir)
        return lambda history, visible: policy.generate(history, {'task_id': task_id}, visible).raw_text
    return factory


def quick_eval(config, splits, output, spec, *, workers=1, suite_runner=None):
    from eval.run_suite import run_suite, task_result, frozen_tasks
    from sed.manifest import load_frozen
    suite_runner = suite_runner or run_suite
    listing = splits['sets']['D_unseen_quick']
    if len(listing['task_ids']) != 24:
        raise ValueError('Independent quick evaluation requires exactly 24 frozen tasks')
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / 'tasks.json', listing, exclusive=True)
    manifest, freeze = load_frozen(config['paths']['manifest'], config['paths']['split_freeze'])
    _, tasks = frozen_tasks(output / 'tasks.json', manifest, freeze)
    summary = suite_runner(config=config, tasks_path=output / 'tasks.json', output=output / 'eval',
        policy='student', spec={**spec, 'decoding': 'greedy'}, workers=workers)
    rows = [task_result(t, output / 'eval' / 'episodes' / f"{i:03d}-{digest(t['task_id'])[:12]}")
            for i, t in enumerate(tasks)]
    return {'summary': summary, 'tasks': rows, 'list_sha256': listing['list_sha256']}


def validate_serving(config, spec, *, base_model=False, expected=None):
    from tools.check_lora_serving import validate_report
    server = config.get('server_provenance', {})
    if server.get('config', {}).get('server', {}).get('batch_invariant') is not True:
        raise ValueError('Repair probes require the grid BATCH_INVARIANT=1 server')
    if base_model:
        return validate_base_serving(config, spec, expected=expected)
    report = config.get('lora_equivalence', {})
    validate_report(report, server, spec['model'], spec['endpoint'])
    # The report's identity intentionally excludes filesystem paths. The
    # validated server row supplies the actual artifact path on this host.
    adapter = next(a for a in server['config']['server']['lora_adapters'] if a['name'] == spec['model'])
    if (adapter['base_model'] != config['models']['student']['requested_repo'] or
            adapter['revision'] != config['models']['student']['revision']):
        raise ValueError('Served adapter differs from configured student')
    return adapter


def validate_training_checkpoint(trained, export, collection_path, export_path, *, arm, checkpoint, adapter):
    """Cross-branch receipt contract; paths may relocate, artifact bytes may not."""
    context = export_branch(export)
    if (trained.get('format') != 'repair_e2_training_run' or trained.get('schema_version') != 1
            or trained.get('arm') != arm or trained.get('dry_run') is not False
            or export.get('ready_for_training') is not True or export.get('dry_run') is not False
            or export.get('canary') is not False or export.get('training_blockers') != []
            or trained.get('training_collection_sha256') != file_hash(export_path)
            or export.get('source_collection_sha256') != file_hash(collection_path)
            or trained.get('source_collection_sha256') != file_hash(collection_path)
            or trained.get('theta0') != export.get('theta0')):
        raise ValueError('Training receipt/export/collection/theta0 mismatch')
    if context['branch'] != 'P0_K20' and any(trained.get(k) != v for k, v in context.items()):
        raise ValueError('Training receipt branch differs from export')
    if len(export['D_repair_train']) != 4 or set(export['D_repair_train']) & set(export['D_repair_probe']):
        raise ValueError('Training receipt requires four disjoint approved pairs')
    for name in ('N', 'F'):
        entries = export['arms'][name]
        if [e['task_id'] for e in entries] != export['D_repair_train']:
            raise ValueError('Training export task order differs')
        for entry in entries:
            require_approved(entry)
    matches = [c for c in trained['checkpoints'] if c['update'] == checkpoint]
    if len(matches) != 1:
        raise ValueError('Missing or duplicate fixed checkpoint')
    point = matches[0]
    # Use the actual served adapter path, not the training host's historical path.
    if point['status'] != 'saved' or checkpoint_files(adapter['path']) != point['sha256']:
        raise ValueError('Checkpoint artifact changed or was never saved')
    for key, name in [('adapter_weights_sha256', 'adapter_model.safetensors'),
                      ('adapter_config_sha256', 'adapter_config.json')]:
        if adapter[key] != point['sha256'][name]:
            raise ValueError('Served adapter differs from requested fixed checkpoint')
    return point


def run_probes(collection, splits, config, output, *, arm, checkpoint, spec, env_factory,
               policy_factory=None, quick_runner=quick_eval, workers=1, inputs=()):
    if (arm, checkpoint) not in {('shared', 0), *[(a, c) for a in ('N', 'F') for c in (1, 5, 20)]}:
        raise ValueError('Use shared checkpoint 0 or N/F checkpoints 1/5/20')
    context = collection_branch(collection)
    base_zero = checkpoint == 0 and context['theta0_kind'] == 'base_model'
    adapter = (validate_serving(config, spec, base_model=True, expected=collection['checkpoint'])
               if base_zero else validate_serving(config, spec))
    if checkpoint == 0 and not base_zero:
        expected = collection['checkpoint']['identity']
        for key in ADAPTER_KEYS:
            if adapter.get(key) != expected.get(key):
                raise ValueError('Checkpoint zero is not collection theta0')
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    prov = provenance(output, config, command='repair_probes', inputs=inputs, details=context)
    factory = policy_factory or served_policy_factory(spec)
    rows = []
    for tid in collection['D_repair_probe']:
        case = next((c for c in collection['cases'] if c['task_id'] == tid), None)
        try:
            if case is None:
                raise ValueError('Probe case was not measured')
            rows.append(probe_case(case, env_factory=env_factory, policy_factory=factory,
                                   output=output / 'cases' / digest(tid)[:12], context=context))
        except (ValueError, KeyError) as exc:
            rows.append({**context, 'task_id': tid, 'm': None, 'continuations': [], 'status': 'unavailable',
                         'error_type': type(exc).__name__})
    quick = quick_runner(config, splits, output / 'quick', spec, workers=workers) if checkpoint in (0, 20) else None
    result = {'schema_version': 1, **context, 'arm': arm, 'checkpoint': checkpoint,
        'adapter': None if base_zero else adapter, 'served_checkpoint': adapter, 'cases': rows, 'quick_eval': quick,
        'seeds': list(SAMPLING_SEEDS), 'temperature': .7, 'top_p': .95, 'H': config['environment']['max_episode_steps'],
        'collection_sha256': digest(collection), 'split_sha256': splits['content_sha256'],
        'model_updates': 0, 'teacher_calls': 0}
    write_sealed(output / 'checkpoint.json', result)
    write_json(output / 'manifest.json', run_manifest(prov, checkpoint=adapter,
        tokenizer=collection['tokenizer'], tasks=collection['D_repair_probe'], seeds={'sampling': list(SAMPLING_SEEDS)},
        **context, evaluation_only=True, result_sha256=digest(result)), exclusive=True)
    return result


def combined_table(results, output):
    if not results:
        raise ValueError('Need checkpoint results')
    by_key = {(r['arm'], r['checkpoint']): r for r in results}
    if len(by_key) != len(results) or ('shared', 0) not in by_key:
        raise ValueError('Need one shared theta0; duplicate checkpoints forbidden')
    for key in ('collection_sha256', 'split_sha256', 'seeds', 'H', 'temperature', 'top_p'):
        if any(r[key] != results[0][key] for r in results):
            raise ValueError('Paired probe conditions differ')
    base = by_key['shared', 0]
    context = collection_branch(base)
    if any(collection_branch(r) != context for r in results):
        raise ValueError('Cannot combine different pilot branches')
    ids = [c['task_id'] for c in base['cases']]
    def mean_m(r):
        if r is None or [c['task_id'] for c in r['cases']] != ids or any(c['m'] is None for c in r['cases']):
            return None
        return sum(c['m'] for c in r['cases']) / len(ids) if ids else None
    m0 = mean_m(base)
    rows, paired, per_probe = [], [], []
    for checkpoint in (0, 1, 5, 20):
        values = {}
        for arm in ('N', 'F'):
            r = base if checkpoint == 0 else by_key.get((arm, checkpoint))
            m = mean_m(r)
            cs = [c for case in r['cases'] for c in case['continuations']] if r else []
            values[arm] = {'m': m, 'delta_m': m - m0 if m is not None and m0 is not None else None,
                'continuation_success': sum(c['success'] is True for c in cs),
                'continuation_known': sum(type(c['success']) is bool for c in cs),
                'continuation_expected': 2 * len(ids),
                'continuation_success_rate': (sum(c['success'] is True for c in cs) / len(cs)
                    if cs and len(cs) == 2 * len(ids) and all(type(c['success']) is bool for c in cs) else None)}
        nd, fd = values['N']['delta_m'], values['F']['delta_m']
        rows.append({'checkpoint': checkpoint, 'N': values['N'], 'F': values['F'],
                     'delta_m_N_minus_delta_m_F': nd - fd if nd is not None and fd is not None else None})
        for tid in ids:
            values_i = {}
            base_case = next(c for c in base['cases'] if c['task_id'] == tid)
            for arm in ('N', 'F'):
                r = base if checkpoint == 0 else by_key.get((arm, checkpoint))
                case = next((c for c in r['cases'] if c['task_id'] == tid), None) if r else None
                if case and any(case.get(k) != base_case.get(k) for k in ('decision_index', 'H', 'A_plus', 'witnesses')):
                    raise ValueError('Probe history or successful witnesses differ')
                m = case['m'] if case else None
                values_i[arm] = m - base_case['m'] if m is not None and base_case['m'] is not None else None
            nd, fd = values_i['N'], values_i['F']
            per_probe.append({'task_id': tid, 'checkpoint': checkpoint, 'delta_m_N': nd, 'delta_m_F': fd,
                'delta_m_N_minus_delta_m_F': nd - fd if nd is not None and fd is not None else None})
    quick_maps = {}
    for key in [('shared', 0), ('N', 20), ('F', 20)]:
        r = by_key.get(key)
        q = r.get('quick_eval') if r else None
        if q:
            tids = [t['task_id'] for t in q['tasks']]
            if len(tids) != len(set(tids)) or ('shared' in quick_maps and set(tids) != set(quick_maps['shared'])):
                raise ValueError('Quick evaluation task identities differ or are duplicated')
            quick_maps[key[0]] = {t['task_id']: t['success'] if t['status'] == 'ok' else 'unknown' for t in q['tasks']}
    quick_counts = {}
    if 'shared' in quick_maps:
        for tid, baseline in quick_maps['shared'].items():
            paired.append({'task_id': tid, 'theta0': baseline, 'N20': quick_maps.get('N', {}).get(tid, 'unknown'),
                           'F20': quick_maps.get('F', {}).get(tid, 'unknown')})
        for left, right in [('theta0', 'N20'), ('theta0', 'F20'), ('N20', 'F20')]:
            quick_counts[left + '_vs_' + right] = {
                'both_success': sum(p[left] is True and p[right] is True for p in paired),
                'left_only': sum(p[left] is True and p[right] is False for p in paired),
                'right_only': sum(p[left] is False and p[right] is True for p in paired),
                'both_failure': sum(p[left] is False and p[right] is False for p in paired),
                'unknown': sum(type(p[left]) is not bool or type(p[right]) is not bool for p in paired)}
    result = {'schema_version': 1, **context, 'checkpoints': rows, 'per_probe': per_probe,
              'source_checkpoint_sha256': [digest(r) for r in results],
              'quick_paired': paired, 'quick_paired_counts': quick_counts,
              'missing_checkpoints': [[a, c] for a in ('N', 'F') for c in (1, 5, 20) if (a, c) not in by_key]}
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / 'combined.json', result, exclusive=True)
    with (output / 'combined.tsv').open('x', newline='') as f:
        fields = ['checkpoint', 'delta_m_N_minus_delta_m_F', *[a + '_' + k for a in ('N', 'F') for k in rows[0][a]]]
        w = csv.DictWriter(f, fieldnames=fields, delimiter='\t')
        w.writeheader()
        for row in rows:
            w.writerow({k: row[k] for k in fields[:2]} | {a + '_' + k: v for a in ('N', 'F') for k, v in row[a].items()})
    for name, records in [('per_probe', per_probe), ('quick_paired', paired), ('quick_counts', [
            {'comparison': k, **v} for k, v in quick_counts.items()])]:
        with (output / (name + '.tsv')).open('x', newline='') as f:
            if records:
                writer = csv.DictWriter(f, fieldnames=list(records[0]), delimiter='\t')
                writer.writeheader()
                writer.writerows(records)
    return result
