"""Teacher-free P0 evaluation and exact paired statistics.

Five random directions yield five paired binary tests, not 5*n independent
examples. Random control point estimates are means; the conservative p-value is
the maximum of the five exact tests. These choices are frozen before outcomes.
"""
from collections import defaultdict
import json
import math
import operator
from pathlib import Path

from repair.p0_agent import load_submission
from repair.p0_introspect import ToolSession, tensor_hash
from repair.p0_pairs import collateral_sample, render_state, provenance, CATEGORIES
from repair.v0_common import SyntheticManifest, read
from sed.storage import digest, write_json, file_hash, locked, atomic_write, canonical_bytes

CONDITIONS = ('none', 'teacher', 'random_0', 'random_1', 'random_2', 'random_3', 'random_4', 'fixed')
STATISTICAL_PLAN = dict(mcnemar='two-sided exact conditional binomial, unadjusted pooled/per-category',
    random_seeds=[0, 1, 2, 3, 4], random_aggregate='arithmetic mean point estimate; max of 5 exact McNemar p-values; superiority must hold for each seed',
    fixed_layers=list(range(20, 42)), fixed_targets='same output projection targets as teacher',
    collateral_seed=0, collateral_decisions=200, collateral_pooled='maximum across five category-specific edits',
    k20_pooled='each category-specific edit must not decrease success vs none',
    unknown_policy='retain unknown; report complete-pair n; cannot declare P0 established with missing outcomes',
    threshold=0.05, layer_indexing='zero based', experiment_unit='repair attempt/decision pair',
    decision_flip_denominator='test pairs whose unedited theta0 HF greedy action reproduces a_f; measured after evaluation barrier',
    primary_rollout_denominator='all test pairs, regardless of theta0 reproduction',
    independence='Instance-disjoint diagnosis/test groups; repeated pairs within instances are not independent tasks')


def mcnemar_exact(left, right):
    """Two-sided exact test; unknown pairs are explicit, never silently false."""
    if len(left) != len(right):
        raise ValueError('Paired outcomes must have equal lengths')
    b = c = n = unknown = 0
    for a, z in zip(left, right):
        if type(a) is not bool or type(z) is not bool:
            unknown += 1
            continue
        n += 1
        b += a and not z
        c += z and not a
    discordant = b + c
    p = min(1.0, 2 * sum(math.comb(discordant, k) for k in range(min(b, c) + 1)) / 2 ** discordant) if discordant else 1.0
    return dict(n=n, unknown=unknown, left_only=b, right_only=c, p_value=p, method='McNemar two-sided exact')


def rate(values):
    known = [v for v in values if type(v) is bool]
    return dict(value=sum(known) / len(known) if known else None, n=len(known), unknown=len(values) - len(known))


def compare(rows):
    by_pair = defaultdict(dict)
    for row in rows:
        if row['condition'] not in CONDITIONS or row['condition'] in by_pair[row['pair_id']]:
            raise ValueError('Unknown/duplicate paired condition')
        by_pair[row['pair_id']][row['condition']] = row
    if not by_pair or any(set(v) != set(CONDITIONS) for v in by_pair.values()):
        raise ValueError('Evaluation must include every condition on every pair')
    ordered = [by_pair[k] for k in sorted(by_pair)]
    reproduced = [r for r in ordered if r['none'].get('reproduction_matches') is True]
    reproduction = dict(rate=rate([r['none'].get('reproduction_matches') for r in ordered]),
        matched=len(reproduced), total=len(ordered),
        mismatches=[k for k in sorted(by_pair) if by_pair[k]['none'].get('reproduction_matches') is False],
        unavailable=[k for k in sorted(by_pair) if type(by_pair[k]['none'].get('reproduction_matches')) is not bool])
    metrics, tests = {}, {}
    for condition in CONDITIONS:
        metrics[condition] = dict(rollout_success=rate([r[condition]['success'] for r in ordered]),
            decision_flip_rate=dict(rate([r[condition]['decision_flip'] for r in reproduced]),
                                    denominator=len(reproduced)))
        if condition not in ('teacher',):
            tests[condition] = mcnemar_exact([r['teacher']['success'] for r in ordered],
                                             [r[condition]['success'] for r in ordered])
    random = {}
    for metric in ('rollout_success', 'decision_flip_rate'):
        values = [metrics[f'random_{s}'][metric]['value'] for s in range(5)]
        random[metric] = dict(value=sum(values) / 5 if None not in values else None,
            n=min(metrics[f'random_{s}'][metric]['n'] for s in range(5)),
            unknown=max(metrics[f'random_{s}'][metric]['unknown'] for s in range(5)))
        if metric == 'decision_flip_rate':
            random[metric]['denominator'] = len(reproduced)
    metrics['random_mean'] = random
    tests['random_mean'] = dict(p_value=max(tests[f'random_{s}']['p_value'] for s in range(5)),
        method='max of five seed-wise exact paired tests', per_seed=[tests[f'random_{s}'] for s in range(5)])
    return dict(metrics=metrics, mcnemar=tests, pairs=len(ordered), reproduction=reproduction)


def predictions(submission, comparison, collateral, k20):
    metrics, tests = comparison['metrics'], comparison['mcnemar']
    def complete_value(condition, metric):
        item = metrics[condition][metric]
        return item['value'] if item['unknown'] == 0 else None
    teacher = complete_value('teacher', 'rollout_success')
    values = dict(rollout_success=teacher, decision_flip_rate=complete_value('teacher', 'decision_flip_rate'),
                  collateral=collateral, k20_success=k20)
    for short, condition in [('none', 'none'), ('random', 'random_mean'), ('fixed', 'fixed')]:
        other = complete_value(condition, 'rollout_success')
        values['rollout_gain_vs_' + short] = teacher - other if teacher is not None and other is not None else None
        test = tests[condition]
        unknown = (any(t['unknown'] for t in test['per_seed']) if 'per_seed' in test else test['unknown'])
        values['mcnemar_p_vs_' + short] = test['p_value'] if not unknown else None
    ops = {'>': operator.gt, '>=': operator.ge, '<': operator.lt, '<=': operator.le}
    return [dict(p, observed=values[p['metric']], passed=(None if values[p['metric']] is None else
        ops[p['direction']](values[p['metric']], p['threshold']))) for p in submission['predictions']]


def model_identity(identity):
    """Hardware is execution provenance, not a different student checkpoint."""
    return {k: v for k, v in identity.items() if k != 'runtime_device'}


def evaluation_identity(pairs, submission_paths, student_identity):
    submissions = [load_submission(p, pairs_sha256=pairs['sha256']) for p in submission_paths]
    if len(submissions) != 5 or {s['category'] for s in submissions} != set(CATEGORIES):
        raise ValueError('All five category submissions must freeze before evaluation')
    for s in submissions:
        expected = sorted(p['pair_id'] for p in pairs['pairs']
                          if p['split'] == 'diagnosis' and p['category'] == s['category'])
        if s['diagnosis_pair_ids'] != expected or model_identity(s['student_identity']) != model_identity(student_identity):
            raise ValueError('Submission diagnosis set/student differs')
    identity = dict(pairs_sha256=pairs['sha256'], submissions={s['category']: s['sha256'] for s in submissions},
                    student=model_identity(student_identity), statistical_plan=STATISTICAL_PLAN)
    return {s['category']: s for s in submissions}, identity


def evaluation_barrier(directory, pairs, submission_paths, student_identity, *, shared=False):
    """Validate all FIVE freezes, then publish plan before any model test call."""
    directory = Path(directory)
    submissions, identity = evaluation_identity(pairs, submission_paths, student_identity)
    value = dict(identity=identity, sha256=digest(identity), provenance=provenance(identity))
    with locked(directory / 'evaluation.lock'):
        path = directory / 'evaluation_freeze.json'
        if shared and path.exists():
            existing = read(path)
            if (existing['identity'] != identity or existing['sha256'] != value['sha256'] or
                    any(existing['provenance'][k] != value['provenance'][k]
                        for k in ('code_sha256', 'dependencies'))):
                raise ValueError('Evaluation freeze differs across shards')
            value = existing
        else:
            write_json(path, value, exclusive=True)
    return submissions, value


def restore_direction(student, frozen):
    import torch
    vectors = {int(i): torch.tensor(d, dtype=torch.float32) for i, d in frozen['directions'].items()}
    ident = digest({str(i): tensor_hash(d) for i, d in vectors.items()})
    if ident != frozen['submission']['edit']['direction_id']:
        raise ValueError('Frozen direction content mismatch')
    student.directions[ident] = vectors
    return frozen['submission']['edit']


def control_edits(student, frozen, diagnosis):
    import torch
    teacher = restore_direction(student, frozen)
    edits = dict(none=None, teacher=teacher)
    for seed in range(5):
        generator = torch.Generator(device='cpu').manual_seed(seed)
        vectors = {}
        for layer in teacher['layers']:
            d = torch.randn(student.config.hidden_size, generator=generator, dtype=torch.float32)
            vectors[layer] = d / d.norm()
        ident = digest({str(i): tensor_hash(d) for i, d in vectors.items()})
        student.directions[ident] = vectors
        edits[f'random_{seed}'] = dict(teacher, direction_id=ident)
    layers = list(range(20, 42))
    if any(p['a_r'] is None for p in diagnosis):
        raise ValueError('Fixed control needs four resolved diagnosis pairs; no silent subset')
    positive = [(render_state(p, student.tokenizer), p['a_f']) for p in diagnosis]
    negative = [(render_state(p, student.tokenizer), p['a_r']) for p in diagnosis]
    direction = student.direction(positive, negative, layers, framing='original_thought')
    edits['fixed'] = dict(teacher, direction_id=direction['direction_id'], layers=layers)
    return edits


class HarnessEvaluation:
    """Authorized train-side games only. Shared environment and execute_actions."""
    def __init__(self, student, pairs, directory):
        self.student, self.pairs, self.directory = student, pairs, Path(directory)
        self.manifest = SyntheticManifest(pairs['manifest_path'])
        if self.manifest.sha256 != pairs['manifest_sha256']:
            raise ValueError('Synthetic manifest changed')
        self.sessions = {}
        self.collateral = collateral_sample(pairs['provenance']['effective_config']['round_dir'])
        self.collateral_states = [render_state(p, student.tokenizer) for p in self.collateral]
        self.baseline_collateral = {}
        self.seed_ids = sorted(self.manifest.value['seed_task_ids'])
        self.seed_sources = self.manifest.value['seed_sources']
        if len(self.seed_ids) != 20 or set(self.seed_sources) != set(self.seed_ids):
            raise ValueError('Expected exactly the original 20 trusted seed instances')
        self.collateral_by_id = {p['pair_id']: i for i, p in enumerate(self.collateral)}

    def environment(self, pair):
        return self.manifest.environment(pair['task_id'], seed=pair['audit']['env_seed'])

    def pair(self, pair, edit):
        key = pair['pair_id']
        if key not in self.sessions:
            self.sessions[key] = ToolSession(self.student, [pair], self.directory / 'pairs' / key,
                env_factory=self.environment, scope='evaluation')
        session = self.sessions[key]
        state = session.state(key)
        with self.student.edited(edit):
            greedy = self.student.greedy(state)
        result = session.call('rollout', dict(state=key, edit=edit))
        return dict(success=result['data']['success'], greedy=greedy['action'],
                    decision_flip=greedy['action'] != pair['a_f'],
                    reproduction_matches=greedy['action'] == pair['a_f'] if edit is None else None,
                    rollout=result)

    def collateral_decision(self, pair_id, edit):
        state = self.collateral_states[self.collateral_by_id[pair_id]]
        if pair_id not in self.baseline_collateral:
            self.baseline_collateral[pair_id] = self.student.greedy(state)['action']
        original = self.baseline_collateral[pair_id]
        with self.student.edited(edit):
            action = original if edit is None else self.student.greedy(state)['action']
        return dict(pair_id=pair_id, original=original, edited=action, changed=original != action)

    def k20_task(self, task_id, edit):
        from harness.alfworld_env import Environment
        from repair.execute import execute_actions
        from repair.ledger import BudgetLedger
        from sed.react import build_messages
        source = self.seed_sources[task_id]
        game = Path(source['game_path'])
        if not task_id.startswith('train/') or file_hash(game) != source['game_sha256']:
            raise ValueError('Seed game identity mismatch')
        root = game.parents[3]
        if root / task_id / 'game.tw-pddl' != game:
            raise ValueError('Seed game path mismatch')
        factory = lambda: Environment(root, seed=source['environment_seed'], max_episode_steps=40, split='train')
        ledger = BudgetLedger(self.directory / 'k20_environment.jsonl')
        with self.student.edited(edit):
            # A zero-action reset is measured, never a fabricated restore.
            reference = execute_actions(factory, task_id, [], 40, ledger=ledger, interaction_kind='evaluation_steps')
            if reference.termination_reason == 'ENV_ERROR':
                raise RuntimeError('K20 reset failed')
            def continuation(history, visible):
                text = self.student.tokenizer.apply_chat_template(build_messages(history, {'task_id': task_id}, visible),
                    tokenize=False, add_generation_prompt=True, enable_thinking=False)
                return self.student.greedy(dict(text=text))['reply']
            result = execute_actions(factory, task_id, [], 40, ledger=ledger,
                interaction_kind='evaluation_steps', continuation=continuation, prefix_reference=reference)
        path = self.directory / 'k20_executions' / (result.trajectory_id + '.json')
        write_json(path, result.to_json(), exclusive=True)
        return dict(task_id=task_id, success=result.success, termination=result.termination_reason,
                    execution_sha256=file_hash(path), environment_steps=result.environment_steps)


def parse_shard(value):
    try:
        parts = value.split('/')
        if len(parts) != 2 or any(not p.isascii() or not p.isdigit() for p in parts):
            raise ValueError
        index, count = map(int, parts)
        if not 0 <= index < count:
            raise ValueError
        return index, count
    except (AttributeError, ValueError):
        raise ValueError('--shard must be zero-based i/n with 0 <= i < n') from None


def shard_owner(unit, count):
    return int(digest(unit), 16) % count


def work_units(pairs, collateral_ids, seed_ids):
    """Stable identities, independent of process, GPU and output path."""
    units = []
    for category in CATEGORIES:
        for pair in sorted((p for p in pairs['pairs'] if p['split'] == 'test' and p['category'] == category),
                           key=lambda p: p['pair_id']):
            units.extend(dict(kind='pair', category=category, condition=c, key=pair['pair_id']) for c in CONDITIONS)
        for condition in CONDITIONS:
            for kind, keys in [('collateral', collateral_ids), ('k20', sorted(seed_ids))]:
                units.extend(dict(kind=kind, category=category, condition=condition, key=key) for key in keys)
    if len({digest(u) for u in units}) != len(units):
        raise ValueError('Duplicate evaluation work units')
    return units


def publish_shared(path, value):
    with locked(path.parent / 'evaluation.lock'):
        if path.exists():
            if read(path) != value:
                raise ValueError('Evaluation work plan differs across shards')
        else:
            write_json(path, value, exclusive=True)


def evaluate(student, pairs, submission_paths, directory, *, runtime_factory=HarnessEvaluation,
             edits_factory=control_edits, shard=None):
    index, count = (0, 1) if shard is None else parse_shard(shard) if isinstance(shard, str) else shard
    if type(index) is not int or type(count) is not int or not 0 <= index < count:
        raise ValueError('Invalid evaluation shard')
    directory = Path(directory)
    submissions, freeze = evaluation_barrier(directory, pairs, submission_paths, student.identity, shared=True)
    workdir = directory / 'shards' / f'{index:04d}-of-{count:04d}'
    # Exclusive ownership also protects all runtime caches, logs and executions.
    workdir.mkdir(parents=True, exist_ok=False)
    runtime = runtime_factory(student, pairs, workdir)
    collateral_ids = [p['pair_id'] for p in runtime.collateral]
    if len(collateral_ids) != 200 or len(set(collateral_ids)) != 200 or len(set(runtime.seed_ids)) != 20:
        raise ValueError('Expected 200 unique collateral decisions and 20 unique K20 tasks')
    units = work_units(pairs, collateral_ids, runtime.seed_ids)
    plan = dict(schema='p0-work-plan/1', freeze_sha256=freeze['sha256'], shards=count,
        units=units, collateral=runtime.collateral, seed_ids=runtime.seed_ids, seed_sources=runtime.seed_sources)
    publish_shared(directory / 'work_plan.json', plan)
    publish_shared(directory / 'collateral_freeze.json', dict(seed=0, decisions=runtime.collateral,
                                                           sha256=digest(runtime.collateral)))
    edits, conditions = {}, {}
    for category in CATEGORIES:
        diagnosis = [p for p in pairs['pairs'] if p['split'] == 'diagnosis' and p['category'] == category]
        edits[category] = edits_factory(student, submissions[category], diagnosis)
        conditions[category] = dict(edits=edits[category], directions={
            str(k): {str(i): d.tolist() for i, d in student.directions[k].items()}
            for k in {e['direction_id'] for e in edits[category].values() if e is not None}})
    publish_shared(directory / 'conditions.json', conditions)
    tests = {p['pair_id']: p for p in pairs['pairs'] if p['split'] == 'test'}
    records = []
    for unit in units:
        if shard_owner(unit, count) != index:
            continue
        edit = edits[unit['category']][unit['condition']]
        if unit['kind'] == 'pair':
            pair = tests[unit['key']]
            result = dict(pair_id=pair['pair_id'], category=unit['category'], condition=unit['condition'],
                task_id=pair['task_id'], group_id=pair['group_id'], alignment=pair['alignment'],
                **runtime.pair(pair, edit))
        elif unit['kind'] == 'collateral':
            result = runtime.collateral_decision(unit['key'], edit)
        else:
            result = runtime.k20_task(unit['key'], edit)
        record = dict(unit_id=digest(unit), result=result)
        write_json(workdir / 'units' / (record['unit_id'] + '.json'), record, exclusive=True)
        records.append(record)
    result = dict(schema='p0-shard/1', index=index, shards=count, work_plan_sha256=digest(plan),
        conditions_sha256=digest(conditions), model_identity=student.identity, records=records)
    write_json(workdir / 'result.json', result, exclusive=True)
    if shard is not None:
        return dict(shard=f'{index}/{count}', units=len(records), result=str(workdir / 'result.json'))
    return merge_evaluation(pairs, submission_paths, directory)


def merge_evaluation(pairs, submission_paths, directory):
    """CPU-only validation and aggregation; never construct a model/environment."""
    directory = Path(directory)
    freeze, plan = read(directory / 'evaluation_freeze.json'), read(directory / 'work_plan.json')
    submissions, identity = evaluation_identity(pairs, submission_paths, freeze['identity']['student'])
    if freeze['identity'] != identity or freeze['sha256'] != digest(identity) or plan['freeze_sha256'] != freeze['sha256']:
        raise ValueError('Evaluation freeze mismatch')
    current = provenance()
    if any(freeze['provenance'][k] != current[k] for k in ('code_sha256', 'dependencies')):
        raise ValueError('Evaluation code/dependencies changed')
    units = work_units(pairs, [p['pair_id'] for p in plan['collateral']], plan['seed_ids'])
    if units != plan['units'] or len(plan['collateral']) != 200 or len(set(plan['seed_ids'])) != 20:
        raise ValueError('Evaluation work unit manifest mismatch')
    expected = {digest(u): u for u in units}
    count, records, shards, identities = plan['shards'], {}, set(), {}
    condition_hash = file_hash(directory / 'conditions.json')
    conditions_digest = digest(read(directory / 'conditions.json'))
    for path in sorted((directory / 'shards').glob('*/result.json')):
        result = read(path)
        index = result['index']
        if index in shards:
            raise ValueError('Duplicate evaluation shard')
        if (type(index) is not int or not 0 <= index < count or result['shards'] != count or
                result['work_plan_sha256'] != digest(plan) or result['conditions_sha256'] != conditions_digest or
                model_identity(result['model_identity']) != identity['student']):
            raise ValueError('Evaluation shard identity mismatch')
        shards.add(index)
        identities[digest(result['model_identity'])] = result['model_identity']
        for row in result['records']:
            key = row['unit_id']
            if key in records:
                raise ValueError('Duplicate evaluation unit')
            if key not in expected or shard_owner(expected[key], count) != index:
                raise ValueError('Unexpected/misassigned evaluation unit')
            records[key] = row['result']
    if shards != set(range(count)) or set(records) != set(expected):
        raise ValueError('Missing evaluation shards or units')
    rows, side = [], defaultdict(list)
    for unit in units:
        result = records[digest(unit)]
        if unit['kind'] == 'pair':
            if any(result[k] != unit[u] for k, u in [('pair_id', 'key'), ('condition', 'condition'), ('category', 'category')]):
                raise ValueError('Pair result identity mismatch')
            rows.append(result)
        else:
            if result['pair_id' if unit['kind'] == 'collateral' else 'task_id'] != unit['key']:
                raise ValueError('Side result identity mismatch')
            side[unit['category'], unit['condition'], unit['kind']].append(result)
    collateral, k20 = {}, {}
    for category in CATEGORIES:
        collateral[category], k20[category] = {}, {}
        for condition in CONDITIONS:
            decisions = side[category, condition, 'collateral']
            seeds = side[category, condition, 'k20']
            collateral[category][condition] = dict(fraction=sum(r['changed'] for r in decisions) / 200,
                                                   n=200, decisions=decisions)
            k20[category][condition] = dict(rate=rate([r['success'] for r in seeds]), rows=seeds,
                cache_hit=False, executed_environment_steps=sum(r['environment_steps'] for r in seeds))
    # All coverage checks finish before publishing anything. Concurrent merges
    # serialize; exclusive report writes reject retries without overwriting.
    with locked(directory / 'evaluation.lock'):
        if file_hash(directory / 'conditions.json') != condition_hash:
            raise ValueError('Conditions changed during merge')
        atomic_write(directory / 'results.jsonl', b''.join(canonical_bytes(r) + b'\n' for r in rows), exclusive=True)
        return write_report(pairs, submissions, directory, freeze, rows, collateral, k20,
                            [identities[k] for k in sorted(identities)])


def write_report(pairs, submissions, directory, freeze, rows, collateral, k20, identities):
    category_reports = {}
    for category in CATEGORIES:
        comparison = compare([r for r in rows if r['category'] == category])
        prediction_results = predictions(submissions[category]['submission'], comparison,
            collateral[category]['teacher']['fraction'], k20[category]['teacher']['rate']['value']
            if k20[category]['teacher']['rate']['unknown'] == 0 else None)
        category_reports[category] = dict(**comparison, predictions=prediction_results)
    pooled = compare(rows)
    teacher_rate = pooled['metrics']['teacher']['rollout_success']['value']
    complete = all(type(r['success']) is bool for r in rows)
    complete = complete and all(k20[c][k]['rate']['unknown'] == 0 for c in CATEGORIES for k in CONDITIONS)
    superior = teacher_rate is not None and all(
        pooled['metrics'][c]['rollout_success']['value'] is not None
        and teacher_rate > pooled['metrics'][c]['rollout_success']['value']
        and pooled['mcnemar'][c]['p_value'] < .05 for c in ('none', 'random_mean', 'fixed'))
    superior = superior and all(t['left_only'] > t['right_only'] for t in pooled['mcnemar']['random_mean']['per_seed'])
    side_ok = all(collateral[c]['teacher']['fraction'] <= .05 and
        k20[c]['teacher']['rate']['value'] is not None and k20[c]['none']['rate']['value'] is not None and
        k20[c]['teacher']['rate']['value'] >= k20[c]['none']['rate']['value'] for c in CATEGORIES)
    all_predictions = [p for c in category_reports.values() for p in c['predictions']]
    report = dict(schema='p0-evaluation/1', freeze_sha256=freeze['sha256'], teacher_calls=0,
        model_identities=identities,
        pairs_sha256=pairs['sha256'], categories=category_reports, pooled=pooled,
        collateral=collateral, k20=k20, counts=pairs['counts'], statistical_plan=STATISTICAL_PLAN,
        complete=complete, preregistered_numeric_criterion=bool(complete and superior and side_ok),
        diagnosis_fidelity=rate([p['passed'] for p in all_predictions]),
        reproduction=dict(pooled['reproduction'], checked=sum(r['condition'] == 'none' for r in rows)),
        inference_limitation='Pair-level analysis; repeated pairs within task instances are not independent tasks.',
        provenance=provenance(dict(evaluation_freeze=freeze['sha256'])))
    write_json(directory / 'report.json', report, exclusive=True)
    lines = ['# P0 evaluation', '', '| Category | N | None | Teacher | Random (5-seed mean) | Fixed |',
             '|---|---:|---:|---:|---:|---:|']
    for name, c in [*category_reports.items(), ('pooled', pooled)]:
        vals = [c['metrics'][k]['rollout_success']['value'] for k in ('none', 'teacher', 'random_mean', 'fixed')]
        lines.append(f"| {name} | {c['pairs']} | " + ' | '.join('unknown' if v is None else f'{v:.3f}' for v in vals) + ' |')
    lines += ['', 'Rollout success uses all test pairs. Decision flip uses only pairs where theta0 reproduces a_f.',
              '', '| Category | Reproduced / total | Reproduction rate | Unknown |', '|---|---:|---:|---:|']
    for name, c in [*category_reports.items(), ('pooled', pooled)]:
        r = c['reproduction']
        value = r['rate']['value']
        display = 'unknown' if value is None else f'{value:.3f}'
        lines.append(f"| {name} | {r['matched']} / {r['total']} | {display} | {r['rate']['unknown']} |")
    lines += ['', '| Category | Condition | Decision flip | Collateral / 200 | K20 success |',
              '|---|---|---:|---:|---:|']
    def fmt(v):
        return 'unknown' if v is None else f'{v:.3f}'
    for category in CATEGORIES:
        for condition in ('none', 'teacher', 'random_mean', 'fixed'):
            if condition == 'random_mean':
                side = sum(collateral[category][f'random_{s}']['fraction'] for s in range(5)) / 5
                k_values = [k20[category][f'random_{s}']['rate']['value'] for s in range(5)]
                k = sum(k_values) / 5 if None not in k_values else None
            else:
                side, k = collateral[category][condition]['fraction'], k20[category][condition]['rate']['value']
            flip = category_reports[category]['metrics'][condition]['decision_flip_rate']['value']
            lines.append(f'| {category} | {condition} | {fmt(flip)} | {fmt(side)} | {fmt(k)} |')
    lines += ['', '| Category | McNemar vs none | vs random (max seed p) | vs fixed |',
              '|---|---:|---:|---:|']
    for name, c in [*category_reports.items(), ('pooled', pooled)]:
        lines.append('| ' + name + ' | ' + ' | '.join(f"{c['mcnemar'][k]['p_value']:.6g}"
                      for k in ('none', 'random_mean', 'fixed')) + ' |')
    lines += ['', '| Category | Prediction | Observed | Pass |', '|---|---|---:|---|']
    for category in CATEGORIES:
        for p in category_reports[category]['predictions']:
            lines.append(f"| {category} | {p['metric']} {p['direction']} {p['threshold']} | {fmt(p['observed'])} | {p['passed']} |")
    lines += ['', f"Numeric criterion met: {report['preregistered_numeric_criterion']}; complete: {complete}.",
              report['inference_limitation'], 'Unknown outcomes and all prediction results are retained in report.json.',
              'Random-control p-value is the maximum of five exact paired tests; no fivefold replication of pairs.']
    lines += ['', 'Model identities (including execution GPU):', '```json',
              json.dumps(identities, indent=2, sort_keys=True), '```']
    atomic_write(directory / 'report.md', ('\n'.join(lines) + '\n').encode(), exclusive=True)
    return report
