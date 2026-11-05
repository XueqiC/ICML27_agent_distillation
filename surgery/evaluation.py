"""Frozen condition evaluation: stable shards, verified resume and CPU reports."""
from copy import deepcopy
from pathlib import Path

from repair.p0_eval import parse_shard, shard_owner, mcnemar_exact, rate, publish_shared, HarnessEvaluation
from repair.p0_introspect import write_receipt
from repair.p0_pairs import collateral_sample, render_state
from repair.v0_common import SyntheticManifest
from sed.storage import digest, file_hash, write_json, atomic_write
from surgery.common import Relocation, checked, read, seal, provenance
from surgery.conditions import condition_set, restore
from surgery.pairs import collateral_b
from surgery.runtime import identity_key
from surgery.session import Session


class Manifest(SyntheticManifest):
    def __init__(self, path, relocation=None):
        self.relocation = relocation or Relocation()
        super().__init__(self.relocation.path(path))

    def asset(self, task, key):
        # Original manifest bytes remain untouched for authorize's hash check.
        task = deepcopy(task)
        task['asset_root'] = str(self.relocation.path(task.get('asset_root', str(self.path.parent))))
        return super().asset(task, key)


def seed_sources(manifest, relocation, data_root=None):
    result = deepcopy(manifest.value['seed_sources'])
    for tid, source in result.items():
        source['game_path'] = str(Path(data_root) / tid / 'game.tw-pddl' if data_root else relocation.path(source['game_path']))
    return result


def environment_factory(pairs, *, manifest=None, data_root=None):
    if pairs['setting'] == 'A':
        if manifest is None:
            raise ValueError('A diagnosis requires its frozen synthetic manifest')
        return lambda p: manifest.environment(p['task_id'], seed=p['audit']['env_seed'])
    if data_root is None:
        raise ValueError('B diagnosis requires --alfworld-data')
    from harness.alfworld_env import Environment
    def environment(pair):
        game = Path(data_root) / pair['task_id'] / 'game.tw-pddl'
        if file_hash(game) != pair['audit']['original']['game_sha256']:
            raise ValueError('K20 seed game changed')
        return Environment(data_root, seed=pair['audit']['env_seed'], max_episode_steps=40, split='train')
    return environment


def make_plan(pairs, conditions, directory, *, round_dir, manifest_path=None, relocation=None,
              data_root=None, shards=1, trial=False, trial_limit=None):
    # This gate runs BEFORE reading held-out/probe task IDs or any test state.
    conditions = condition_set(conditions, pairs, trial=trial)
    if type(shards) is not int or shards < 1 or (trial_limit is not None and (not trial or trial_limit < 1)):
        raise ValueError('Invalid shard/trial plan')
    relocation = relocation or Relocation()
    manifest = Manifest(manifest_path or Path(round_dir) / 'synth/manifest.json', relocation)
    probes = sorted(tid for tid, row in manifest.tasks.items() if row['heldout_probe'])
    seeds = sorted(manifest.seeds)
    if len(probes) != 71 or len(seeds) != 20:
        raise ValueError('M2 requires exactly 71 round-1 probes and 20 original K20 tasks')
    if pairs['setting'] == 'A' and manifest.sha256 != pairs['manifest_sha256']:
        raise ValueError('M2 round differs from the frozen setting-A source')
    if pairs['setting'] == 'B' and {p['parent_seed_task_id'] for p in pairs['pairs'] + pairs['matched_pool']} != set(seeds):
        raise ValueError('M2 original K20 differs from setting-B verified trajectories')
    collateral = collateral_sample(relocation.path(round_dir)) if pairs['setting'] == 'A' else collateral_b(pairs)
    test = [p['pair_id'] for p in pairs['pairs'] if p['split'] == 'test'] if pairs['setting'] == 'A' else []
    if trial_limit is not None:
        test, probes, seeds, collateral = test[:trial_limit], probes[:trial_limit], seeds[:trial_limit], collateral[:trial_limit]
    keys = dict(M1=test, M2_probe=probes, M2_k20=seeds, M3=[p['pair_id'] for p in collateral])
    units = [dict(condition=c, metric=m, key=k) for c in sorted(conditions) for m, ids in keys.items() for k in ids]
    value = seal(dict(schema='p1-evaluation-plan/1', pairs_sha256=pairs['sha256'], setting=pairs['setting'],
        conditions={n: c['sha256'] for n, c in conditions.items()}, shards=shards, trial=trial, units=units,
        collateral=collateral, seed_sources=seed_sources(manifest, relocation, data_root),
        manifest_path=str(manifest.path), manifest_sha256=manifest.sha256,
        code_sha256=provenance()['code_sha256'], dependencies=provenance()['dependencies'],
        k20_label='training tasks' if pairs['setting'] == 'B' else 'original K20 tasks',
        environment=dict(H=40, decoding='greedy', seed=0), data_manifest_split_freeze=pairs['split_sha256']))
    publish_shared(Path(directory) / 'plan.json', value)
    return value


class Runtime(HarnessEvaluation):
    def __init__(self, student, pairs, directory, plan, relocation=None):
        self.student, self.pairs, self.directory = student, pairs, Path(directory)
        self.relocation = relocation or Relocation()
        self.manifest = Manifest(plan['manifest_path'], self.relocation)
        if self.manifest.sha256 != plan['manifest_sha256']:
            raise ValueError('Evaluation manifest bytes changed')
        self.seed_sources = {k: dict(v, game_path=str(self.relocation.path(v['game_path']))) for k, v in plan['seed_sources'].items()}
        self.collateral = {p['pair_id']: p for p in plan['collateral']}
        self.by_pair = {p['pair_id']: p for p in pairs['pairs'] if p['split'] == 'test'}

    def execution_receipts(self, root):
        return {str(p.relative_to(self.directory)): file_hash(p) for p in sorted(Path(root).rglob('*.json'))
                if p.parent.name in ('executions', 'k20_executions', 'probe_executions')}

    def evaluate(self, unit, edits):
        metric, key = unit['metric'], unit['key']
        if metric == 'M1':
            pair = self.by_pair[key]
            session = Session(self.student, [pair], self.directory / 'pairs' / key,
                scope='evaluation', env_factory=self.environment)
            state = session.state(key)
            with self.student.edited(edits):
                greedy = self.student.greedy(state)
            result = session.rollout(state, edits)
            value = dict(success=result['success'], greedy=greedy['action'], a_f=pair['a_f'], rollout=result)
            root = session.directory
        elif metric == 'M3':
            with self.student.edited(edits) as edit_receipt:
                greedy = self.student.greedy(render_state(self.collateral[key], self.student.tokenizer))
            return dict(greedy=greedy['action'], usage=greedy, edit_receipt=edit_receipt)
        elif metric == 'M2_k20':
            value = self.k20_task(key, edits)  # Native P0 initial-state rollout.
            root = self.directory / 'k20_executions'
        elif metric == 'M2_probe':
            value = self.probe(key, edits)
            root = self.directory / 'probe_executions'
        else:
            raise ValueError('Unknown metric')
        value['artifacts'] = self.execution_receipts(root)
        return value

    def probe(self, tid, edits):
        from repair.execute import execute_actions
        from repair.ledger import BudgetLedger
        from sed.react import build_messages
        factory = lambda: self.manifest.environment(tid, probes=True, seed=0)
        self.manifest.authorize(tid, probes=True)
        ledger = BudgetLedger(self.directory / 'probe_environment.jsonl')
        with self.student.edited(edits) as edit_receipt:
            reference = execute_actions(factory, tid, [], 40, ledger=ledger, interaction_kind='evaluation_steps')
            if reference.termination_reason == 'ENV_ERROR':
                raise RuntimeError('Probe reset failed')
            def continuation(history, visible):
                text = self.student.tokenizer.apply_chat_template(build_messages(history, {'task_id': tid}, visible),
                    tokenize=False, add_generation_prompt=True, enable_thinking=False)
                return self.student.greedy(dict(text=text))['reply']
            result = execute_actions(factory, tid, [], 40, ledger=ledger, interaction_kind='evaluation_steps',
                                     continuation=continuation, prefix_reference=reference)
        path = self.directory / 'probe_executions' / (result.trajectory_id + '.json')
        write_receipt(path, result.to_json())
        return dict(task_id=tid, success=result.success, termination=result.termination_reason,
                    execution_sha256=file_hash(path), environment_steps=result.environment_steps,
                    edit_receipt=edit_receipt)


def verify_unit(path, unit, plan, directory):
    row = checked(read(path), 'p1-unit/1')
    if row['unit'] != unit or row['plan_sha256'] != plan['sha256'] or row['condition_sha256'] != plan['conditions'][unit['condition']]:
        raise ValueError('Resume unit identity mismatch')
    root = Path(directory) / 'runs' / unit['condition'] / ('shard-' + str(shard_owner(unit, plan['shards'])))
    for relative, sha in row['result'].get('artifacts', {}).items():
        target = (root / relative).resolve()
        if not target.is_relative_to(root.resolve()) or file_hash(target) != sha:
            raise ValueError('Execution receipt missing or changed')
    return row


def evaluate_shard(pairs, conditions, directory, plan, *, shard, runtime_factory, resume=False):
    directory = Path(directory)
    checked(plan, 'p1-evaluation-plan/1')
    if shard[1] != plan['shards'] or not 0 <= shard[0] < shard[1]:
        raise ValueError('Shard plan mismatch')
    values = condition_set(conditions, pairs, trial=plan['trial'])
    if {n: v['sha256'] for n, v in values.items()} != plan['conditions']:
        raise ValueError('Conditions changed after plan freeze')
    if plan['code_sha256'] != provenance()['code_sha256'] or plan['dependencies'] != provenance()['dependencies']:
        raise ValueError('Evaluation code/dependencies changed since plan freeze')
    units = [u for u in plan['units'] if shard_owner(u, shard[1]) == shard[0]]
    done = 0
    for name in sorted(values):
        assigned = [u for u in units if u['condition'] == name]
        pending = []
        for u in assigned:
            path = directory / 'units' / (digest(u) + '.json')
            if path.exists():
                if not resume:
                    raise FileExistsError('Use --resume to verify and skip existing units')
                verify_unit(path, u, plan, directory)
                done += 1
            else:
                pending.append((u, path))
        if not pending:
            continue
        run_dir = directory / 'runs' / name / ('shard-' + str(shard[0]))
        runtime = runtime_factory(values[name], run_dir)
        try:
            if identity_key(runtime.student.identity) != identity_key(values[name]['student_identity']):
                raise ValueError('Loaded runtime differs from frozen condition')
            restore(runtime.student, values[name])
            for unit, path in pending:
                before = deepcopy(runtime.student.usage)
                result = runtime.evaluate(unit, values[name]['edits'])
                row = seal(dict(schema='p1-unit/1', unit=unit, result=result, plan_sha256=plan['sha256'],
                    condition_sha256=values[name]['sha256'], runtime_identity=runtime.student.identity,
                    edit_receipt=values[name]['edit_receipt'],
                    student_usage={k: runtime.student.usage[k] - v for k, v in before.items()},
                    provenance=provenance(dict(shard=list(shard))), training_exposure=0, teacher_calls=0))
                write_receipt(path, row)
                done += 1
        finally:
            if hasattr(runtime, 'close'):
                runtime.close()
    return dict(shard=list(shard), completed=done, assigned=len(units), plan_sha256=plan['sha256'])


def metric_rate(values):
    # P0's rate preserves unknowns. Explicit fields simplify the condition table.
    values = list(values)
    known = [v for v in values if type(v) is bool]
    return dict(value=sum(known) / len(known) if known else None, numerator=sum(known),
                denominator=len(known), total=len(values), unknown=len(values) - len(known))


def import_m4(values, conditions, relocation=None):
    """CPU-only ingestion, bound to the exported condition and historical suite."""
    relocation = relocation or Relocation()
    conditions = {c['name']: c for c in conditions}
    result = {}
    for value in values:
        name, sep, path = value.partition('=')
        if not sep or name not in conditions or name in result:
            raise ValueError('--m4-result needs a unique CONDITION=JOB_DIR')
        root = relocation.path(path)
        launch = checked(read(root / 'p1_launch.json'), 'p1-m4-launch/1')
        summary = read(root / 'eval/summary.json')
        server = read(root / 'server/provenance.json')['config']['server']
        if (launch['condition_sha256'] != conditions[name]['sha256'] or summary['episodes'] != 140
                or launch['conditions'] != {n: c['sha256'] for n, c in conditions.items()}
                or summary['task_list_sha256'] != launch['task_list_sha256']
                or not summary['batch_invariant'] or summary['policy'] != 'student'
                or summary['student'].get('decoding', 'greedy') != 'greedy'
                or not server['merged_model'] or not server['batch_invariant']
                or relocation.path(server['model_path']) != relocation.path(launch['exported'])):
            raise ValueError('M4 suite differs from frozen standalone condition/protocol')
        from surgery.export import verify_export
        exported = verify_export(relocation.path(launch['exported']))
        if exported['sha256'] != launch['export_manifest_sha256']:
            raise ValueError('M4 exported weights changed')
        result[name] = dict(value=summary['success_rate'], numerator=summary['successes'], denominator=140,
            unknown=summary['unknown_success'], launch_sha256=launch['sha256'],
            summary_sha256=file_hash(root / 'eval/summary.json'), status='standalone_dev140')
    return result


def merge(pairs, conditions, directory, *, m4=None):
    directory = Path(directory)
    plan = checked(read(directory / 'plan.json'), 'p1-evaluation-plan/1')
    values = condition_set(conditions, pairs, trial=plan['trial'])
    if plan['pairs_sha256'] != pairs['sha256'] or plan['conditions'] != {n: v['sha256'] for n, v in values.items()}:
        raise ValueError('Merge inputs differ from frozen plan')
    expected_paths = {digest(u) + '.json' for u in plan['units']}
    if {p.name for p in (directory / 'units').glob('*.json')} != expected_paths:
        raise ValueError('Missing/extra units; refusing partial report')
    rows = [verify_unit(directory / 'units' / (digest(u) + '.json'), u, plan, directory) for u in plan['units']]
    if any(identity_key(row['runtime_identity']) != identity_key(values[row['unit']['condition']]['student_identity']) for row in rows):
        raise ValueError('Merged unit runtime differs from frozen condition')
    by_key = {(r['unit']['condition'], r['unit']['metric'], r['unit']['key']): r['result'] for r in rows}
    base = ('E' if pairs['setting'] == 'A' else 'B') + '0'
    table, appendix = {}, {}
    for name in sorted(values):
        metrics = {}
        for metric in ('M1', 'M2_probe', 'M2_k20', 'M3'):
            items = [(u, by_key[(name, metric, u['key'])]) for u in plan['units'] if u['condition'] == name and u['metric'] == metric]
            if metric == 'M1' and pairs['setting'] == 'B':
                metrics[metric] = dict(value=None, status='not_applicable: all B pairs used for diagnosis')
            elif metric == 'M3':
                metrics[metric] = metric_rate(r['greedy'] != by_key[(base, metric, u['key'])]['greedy'] for u, r in items)
            else:
                metrics[metric] = metric_rate(r['success'] for _, r in items)
                if metric == 'M1':
                    metrics['decision_flip_rate'] = metric_rate(r['greedy'] != r['a_f'] for u, r in items
                        if by_key[(base, metric, u['key'])]['greedy'] == r['a_f'])
                left = [by_key[(base, metric, u['key'])]['success'] for u, _ in items]
                right = [r['success'] for _, r in items]
                appendix.setdefault(name, {})[metric] = mcnemar_exact(left, right) if items else None
        metrics['M4'] = (m4 or {}).get(name, dict(value=None, status='pending standalone dev140 evaluation'))
        table[name] = metrics
    prefix = base[0]
    random_rows = [table[prefix + '2_' + str(i)] for i in range(5)] if not plan['trial'] else []
    if random_rows:
        table[prefix + '2_mean'] = {}
        for metric in table[base]:
            estimates = [r[metric].get('value') for r in random_rows]
            table[prefix + '2_mean'][metric] = dict(value=sum(estimates) / 5 if all(v is not None for v in estimates) else None,
                per_seed=estimates, note='five-seed mean; never treat seeds as independent tasks')
    report = seal(dict(schema='p1-report/1', setting=pairs['setting'], complete=True, trial=plan['trial'],
        plan_sha256=plan['sha256'], metrics=table, k20_label=plan['k20_label'],
        appendix=dict(mcnemar_vs_unedited=appendix, note='Descriptive appendix only; no significance gate'),
        rows=rows, conditions=plan['conditions'], M4_complete=all(table[n]['M4'].get('value') is not None for n in values),
        interpretation='Engineering tooling; no Stage-1 scientific efficacy claim'))
    write_json(directory / 'report.json', report)
    columns = ['M1', 'decision_flip_rate', 'M2_probe', 'M2_k20', 'M3', 'M4']
    lines = ['# P1 ' + pairs['setting'] + (' TRIAL (excluded)' if plan['trial'] else ''), '',
        'Point estimates; M2 K20: ' + plan['k20_label'] + '. Unknown outcomes are retained.', '',
        '| Condition | ' + ' | '.join(columns) + ' |', '|---|' + '---:|' * len(columns)]
    for name, metrics in table.items():
        formatted = []
        for col in columns:
            value = metrics.get(col, {}).get('value')
            formatted.append('—' if value is None else f'{100 * value:.2f}%')
        lines.append('| ' + name + ' | ' + ' | '.join(formatted) + ' |')
    lines += ['', 'McNemar results are in the JSON appendix only. M4 remains pending until standalone dev140 results are imported.', '']
    atomic_write(directory / 'report.md', '\n'.join(lines).encode())
    return report
