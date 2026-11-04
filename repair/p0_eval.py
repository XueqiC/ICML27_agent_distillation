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
from sed.storage import digest, write_json, file_hash, JsonlLedger

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


def evaluation_barrier(directory, pairs, submission_paths, student_identity):
    """Validate all FIVE freezes, then publish plan before any model test call."""
    directory = Path(directory)
    submissions = [load_submission(p, pairs_sha256=pairs['sha256']) for p in submission_paths]
    if len(submissions) != 5 or {s['category'] for s in submissions} != set(CATEGORIES):
        raise ValueError('All five category submissions must freeze before evaluation')
    for s in submissions:
        expected = sorted(p['pair_id'] for p in pairs['pairs']
                          if p['split'] == 'diagnosis' and p['category'] == s['category'])
        if s['diagnosis_pair_ids'] != expected or s['student_identity'] != student_identity:
            raise ValueError('Submission diagnosis set/student differs')
    identity = dict(pairs_sha256=pairs['sha256'], submissions={s['category']: s['sha256'] for s in submissions},
                    student=student_identity, statistical_plan=STATISTICAL_PLAN)
    value = dict(identity=identity, sha256=digest(identity), provenance=provenance(identity))
    write_json(directory / 'evaluation_freeze.json', value, exclusive=True)
    return {s['category']: s for s in submissions}, value


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
        self.baseline_collateral = None
        self.k20_cache = {}

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

    def collateral_run(self, edit):
        if self.baseline_collateral is None:
            self.baseline_collateral = [self.student.greedy(s)['action'] for s in self.collateral_states]
        with self.student.edited(edit):
            actions = self.baseline_collateral if edit is None else [self.student.greedy(s)['action'] for s in self.collateral_states]
        changes = [a != b for a, b in zip(self.baseline_collateral, actions)]
        return dict(fraction=sum(changes) / 200, n=200, decisions=[dict(pair_id=p['pair_id'],
            original=a, edited=b, changed=c) for p, a, b, c in zip(self.collateral, self.baseline_collateral, actions, changes)])

    def k20_run(self, edit):
        from harness.alfworld_env import Environment
        from repair.execute import execute_actions
        from repair.ledger import BudgetLedger
        from sed.react import build_messages
        rows = []
        sources = self.manifest.value['seed_sources']
        seed_ids = self.manifest.value['seed_task_ids']
        if len(seed_ids) != 20 or set(sources) != set(seed_ids):
            raise ValueError('Expected exactly the original 20 trusted seed instances')
        cache_key = digest(dict(student=self.student.identity, edit=edit,
            framing='free_greedy', sources={t: {k: sources[t][k] for k in ('game_sha256', 'environment_seed')} for t in seed_ids}))
        if cache_key in self.k20_cache:
            return dict(self.k20_cache[cache_key], cache_hit=True, executed_environment_steps=0)
        with self.student.edited(edit):
            for task_id in sorted(seed_ids):
                source = sources[task_id]
                game = Path(source['game_path'])
                if not task_id.startswith('train/') or file_hash(game) != source['game_sha256']:
                    raise ValueError('Seed game identity mismatch')
                root = game.parents[3]
                if root / task_id / 'game.tw-pddl' != game:
                    raise ValueError('Seed game path mismatch')
                factory = lambda: Environment(root, seed=source['environment_seed'], max_episode_steps=40, split='train')
                ledger = BudgetLedger(self.directory / 'k20_environment.jsonl')
                # A zero-action reset is a measured reference, not a fabricated restore.
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
                write_json(path, result.to_json())
                rows.append(dict(task_id=task_id, success=result.success, termination=result.termination_reason,
                                 execution_sha256=file_hash(path), environment_steps=result.environment_steps))
        result = dict(rate=rate([r['success'] for r in rows]), rows=rows, cache_hit=False,
                      executed_environment_steps=sum(r['environment_steps'] for r in rows))
        if all(type(r['success']) is bool for r in rows):
            self.k20_cache[cache_key] = result
        return result


def evaluate(student, pairs, submission_paths, directory, *, runtime_factory=HarnessEvaluation,
             edits_factory=control_edits):
    directory = Path(directory)
    submissions, freeze = evaluation_barrier(directory, pairs, submission_paths, student.identity)
    runtime = runtime_factory(student, pairs, directory)
    rows, collateral, k20, category_reports = [], {}, {}, {}
    ledger = JsonlLedger(directory / 'results.jsonl')
    # Collateral draw is frozen before observing any actions/outcomes.
    if hasattr(runtime, 'collateral'):
        write_json(directory / 'collateral_freeze.json', dict(seed=0, decisions=runtime.collateral,
                   sha256=digest(runtime.collateral)), exclusive=True)
    for category in CATEGORIES:
        diagnosis = [p for p in pairs['pairs'] if p['split'] == 'diagnosis' and p['category'] == category]
        tests = [p for p in pairs['pairs'] if p['split'] == 'test' and p['category'] == category]
        edits = edits_factory(student, submissions[category], diagnosis)
        write_json(directory / category / 'conditions.json', dict(edits=edits,
            directions={str(k): {str(i): d.tolist() for i, d in student.directions[k].items()}
                        for k in {e['direction_id'] for e in edits.values() if e is not None}}), exclusive=True)
        for pair in sorted(tests, key=lambda p: p['pair_id']):
            for condition in CONDITIONS:
                result = runtime.pair(pair, edits[condition])
                row = dict(pair_id=pair['pair_id'], category=category, condition=condition,
                    task_id=pair['task_id'], group_id=pair['group_id'], alignment=pair['alignment'], **result)
                rows.append(row); ledger.append(row)
        collateral[category], k20[category] = {}, {}
        for condition in CONDITIONS:
            collateral[category][condition] = runtime.collateral_run(edits[condition])
            k20[category][condition] = runtime.k20_run(edits[condition])
            write_json(directory / category / (condition + '_side_metrics.json'),
                       dict(collateral=collateral[category][condition], k20=k20[category][condition]), exclusive=True)
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
    (directory / 'report.md').write_text('\n'.join(lines) + '\n')
    return report
