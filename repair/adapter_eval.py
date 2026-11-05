"""Opt-in, teacher-free evaluation identity and training-exposure checks.

These fields are audit metadata only; none is passed to the student prompt.
"""
from pathlib import Path
import re

from repair.artifacts import read_json
from sed.storage import file_hash


PREFIX_VERIFICATION = (
    'Reset and replay the source theta0 public replies verbatim through the action parser. '
    'Check game hash, environment seed/version, reset and every prefix transition '
    '(visible observation/admissible commands, parsed action, reward, termination/truncation, '
    'and audit digest) against the source reference before entering continuation. '
    'No student generation occurs during replay. prefix_verified means continuation was '
    'entered after these checks, not that the evaluated adapter reproduced theta0 replies '
    'or that its suffix succeeded. With from_start only reset equality is checked. '
    'As in V1, the reference environment version is rebound to the current harness; '
    'audit fields remain verifier-only.'
)


class NoTeacherBudget:
    """Run-local zero accounting. Never opens a supplied shared ledger."""
    def __init__(self, caps=None):
        self.caps = caps or {'input_tokens': None, 'output_tokens': None}

    def summary(self):
        return dict(caps=dict(self.caps), tokens_or_unsettled_bounds=dict(input_tokens=0, output_tokens=0),
                    reported_usage=dict(input_tokens=0, output_tokens=0, reasoning_tokens=0, visible_tokens=0),
                    unknown_quantities={}, reserved_attempts=0, unsettled_attempts=0, stop_reason=None,
                    scope='eval_adapter_run_only', shared_ledger_accessed=False)

    def reserve(self, *args, **kwargs):
        raise AssertionError('Teacher calls are forbidden in eval-adapter mode')

    settle = halt = check = reserve


def assert_no_teacher_calls(rows, output=None):
    from repair.joint import usage_summary
    usages = [row['teacher_usage'] for row in rows]
    if output is not None:
        usages += [usage_summary(p) for p in Path(output).glob('calls/*/*/usage.jsonl')]
    if any(u['calls'] or u['unknown_usage_attempts'] or u['interrupted_requests']
           or any(u['totals'].values()) for u in usages):
        raise AssertionError('Teacher usage found in eval-adapter run')
    if any(row['mode'] != 'U' or row['guidance_injections'] or any(
            s['review_attempts'] or s['executor'] == 'teacher' or s['guidance_applied']
            for s in row['steps']) for row in rows):
        raise AssertionError('Guidance found in eval-adapter run')


def _exposure(manifest, *, synthetic_groups=None):
    """Read explicit exposure fields, including interchange's group-valued tasks.

    A scalar training_exposure is a count, not a list of IDs. Bundle inventories
    are conservative exclusions, not a claim every task was sampled by an update.
    """
    tasks, groups, sources = set(), set(), []

    def add(values, source):
        if not isinstance(values, list) or any(not isinstance(v, str) or not v for v in values):
            raise ValueError('Training exposure identifiers must be lists of nonempty strings')
        sources.append(source)
        for value in values:
            parts = value.split('/')
            if any(p in ('', '.', '..') for p in parts) or len(parts) > 3:
                raise ValueError('Invalid training exposure identifier')
            if len(parts) == 3:
                tasks.add(value)
                if synthetic_groups is not None and parts[:2] == ['train', 'synth']:
                    if value not in synthetic_groups:
                        raise ValueError('V0 training exposure has an unmapped synthetic task')
                    groups.add(synthetic_groups[value])
                else:
                    groups.add('/'.join(parts[:2]))
            else:
                groups.add(value if len(parts) == 2 else 'train/' + value)

    def read(document, source):
        exposure = document.get('training_exposure')
        if isinstance(exposure, dict):
            fields = {'tasks', 'task_ids', 'groups', 'group_ids'} & exposure.keys()
            if not fields:
                raise ValueError('Unrecognized structured training_exposure')
            for key in sorted(fields):
                add(exposure[key], source + '.training_exposure.' + key)
        elif isinstance(exposure, list):
            add(exposure, source + '.training_exposure')
        elif exposure is not None and (type(exposure) is not int or exposure < 0):
            raise ValueError('Invalid training_exposure')
        for key in ('training_exposure_tasks', 'training_exposure_groups'):
            if key in document:
                add(document[key], source + '.' + key)

    read(manifest, 'manifest')
    for i, bundle in enumerate(manifest.get('input_manifests', [])):
        if not isinstance(bundle, dict):
            raise ValueError('Embedded interchange manifest must be an object')
        read(bundle, f'input_manifests[{i}]')
    return dict(task_ids=sorted(tasks), group_ids=sorted(groups)), sources


def _v0_exposure(run, update, run_path):
    """Exclude the complete, hash-bound R/G/S/D inventory, even unsampled rows.

    V0 realised_exposure/buckets contain counts and opaque episode IDs, while
    example_sources.json binds each episode to its task and explicit group.
    Synthetic groups are fingerprints, not the shared train/synth directory.
    The input synthesis manifest also contains held-out tasks: it is not a
    training inventory and must not be swept into exposure.
    """
    path = run_path.parent / 'example_sources.json'
    expected = run.get('artifacts_sha256', {}).get(path.name)
    if not isinstance(expected, str) or not re.fullmatch(r'[0-9a-f]{64}', expected):
        raise ValueError('V0 training exposure requires a hash-bound example_sources.json')
    if file_hash(path) != expected:
        raise ValueError('V0 training exposure example_sources.json hash mismatch')
    rows = read_json(path)
    if not isinstance(rows, list) or not rows:
        raise ValueError('V0 training exposure requires nonempty example sources')
    tasks, episodes = {}, {}
    buckets = {b: dict(rows=0, episodes=set()) for b in 'RGSD'}
    for row in rows:
        if not isinstance(row, dict) or not isinstance(row.get('v0'), dict):
            raise ValueError('V0 training exposure requires episode metadata for every example')
        task, group = row.get('task_id'), row.get('group_id')
        for value, size in ((task, 3), (group, 2)):
            if (not isinstance(value, str) or len(value.split('/')) != size
                    or any(p in ('', '.', '..') for p in value.split('/'))):
                raise ValueError('V0 training exposure requires valid task and group IDs')
        if (task.startswith('train/synth/') and not group.startswith('synth/')
                or not task.startswith('train/synth/') and group != task.rsplit('/', 1)[0]):
            raise ValueError('V0 training exposure task/group mismatch')
        bucket, episode = row['v0'].get('bucket'), row['v0'].get('episode_id')
        if (not isinstance(bucket, str) or bucket not in buckets
                or not isinstance(episode, str) or not episode):
            raise ValueError('V0 training exposure requires R/G/S/D bucket and episode IDs')
        if tasks.setdefault(task, group) != group or episodes.setdefault(episode, (task, group, bucket)) != (task, group, bucket):
            raise ValueError('V0 training exposure has conflicting episode/task identities')
        buckets[bucket]['rows'] += 1
        buckets[bucket]['episodes'].add(episode)

    def check_count(document, key, actual, source):
        if key in document and (type(document[key]) is not int or document[key] != actual):
            raise ValueError(f'V0 training exposure {source}.{key} differs from example sources')

    inventory = dict(task_ids=sorted(tasks), group_ids=sorted(set(tasks.values())))
    for key, count in (('unique_instances', len(tasks)), ('task_groups', len(inventory['group_ids'])),
                       ('episodes', len(episodes))):
        check_count(run, key, count, 'run')
    input_manifest = run.get('input_manifest', {})
    if not isinstance(input_manifest, dict):
        raise ValueError('V0 input_manifest must be an object')
    for source, counts in (('run.buckets', run.get('buckets')),
                           ('input_manifest.counts', input_manifest.get('counts', run.get('buckets')))):
        if not isinstance(counts, dict) or not counts or set(counts) - buckets.keys():
            raise ValueError('V0 training exposure requires R/G/S/D bucket counts')
        for bucket, actual in buckets.items():
            declared = counts.get(bucket, {})
            if (not isinstance(declared, dict)
                    or (actual['rows'] and not {'rows', 'episodes'} <= declared.keys())):
                raise ValueError('V0 training exposure has an incomplete bucket inventory')
            check_count(declared, 'rows', actual['rows'], source + '.' + bucket)
            check_count(declared, 'episodes', len(actual['episodes']), source + '.' + bucket)

    sources = ['run.example_sources.json (sha256=' + expected + '; all R/G/S/D examples)']
    for source, accounting in (('run.realised_exposure', run.get('realised_exposure', {})),
                                ('checkpoint', update),
                                ('checkpoint.realised_exposure', update.get('realised_exposure', {}))):
        if not isinstance(accounting, dict):
            raise ValueError('V0 realised exposure must be an object')
        if 'episode_exposures' not in accounting:
            continue
        counts = accounting['episode_exposures']
        if not isinstance(counts, dict) or any(type(n) is not int or n < 0 for n in counts.values()):
            raise ValueError('V0 episode exposure counts must be nonnegative integers')
        seen = {episode for episode, n in counts.items() if n}
        if seen - episodes.keys():
            raise ValueError('V0 training exposure contains unknown exposed episodes')
        check_count(accounting, 'unique_instances_seen', len({episodes[e][0] for e in seen}), source)
        check_count(accounting, 'task_groups_seen', len({episodes[e][1] for e in seen}), source)
        sources.append(source + '.episode_exposures')

    # Explicit run, embedded interchange, and update exclusions are additive.
    for source, document in (('run', run), ('run.input_manifest', input_manifest), ('checkpoint', update)):
        exposure, declared_sources = _exposure(document, synthetic_groups=tasks)
        for key in inventory:
            inventory[key] = sorted(set(inventory[key]) | set(exposure[key]))
        sources.extend(source + '.' + s for s in declared_sources)
    return inventory, sources, [str(path)]


def training_identity(adapter):
    checkpoint = Path(adapter['path']).resolve()
    if checkpoint.name != 'checkpoint' or not re.fullmatch(r'update-\d+', checkpoint.parent.name):
        return None, []
    # A recognizable training checkpoint may not silently lose its manifest.
    run_path, update_path = checkpoint.parent.parent / 'manifest.json', checkpoint.parent / 'manifest.json'
    run, update = read_json(run_path), read_json(update_path)
    if run.get('schema') not in ('v2-training-run/1', 'v0-training-run/1'):
        raise ValueError('Expected V2 or V0 training run manifest')
    v0 = run['schema'] == 'v0-training-run/1'
    version = 'V0' if v0 else 'V2'
    arm = run.get('variant', run.get('dataset_sha256')) if v0 else run.get('arm')
    updates = update.get('optimizer_updates')
    if (type(updates) is not int or updates < 0 or updates != int(checkpoint.parent.name.split('-')[1])
            or not isinstance(arm, str) or not arm):
        raise ValueError(f'Invalid {version} arm or checkpoint optimizer_updates')
    for name, field in [('adapter_model.safetensors', 'adapter_weights_sha256'),
                        ('adapter_config.json', 'adapter_config_sha256')]:
        expected = update.get('sha256', {}).get(name)
        if (not isinstance(expected, str) or not re.fullmatch(r'[0-9a-f]{64}', expected)
                or expected != adapter.get(field)):
            raise ValueError(f'{version} checkpoint manifest differs from evaluated adapter')
    inputs = [str(run_path), str(update_path)]
    if v0:
        exposure, sources, exposure_inputs = _v0_exposure(run, update, run_path)
        inputs.extend(exposure_inputs)
        arm = 'v0:' + arm
        scope = 'declared IDs plus conservative complete R/G/S/D example inventory exclusions'
    else:
        exposure, sources = _exposure(run)
        update_exposure, update_sources = _exposure(update)
        for key in exposure:
            exposure[key] = sorted(set(exposure[key]) | set(update_exposure[key]))
        sources = ['run.' + s for s in sources] + ['checkpoint.' + s for s in update_sources]
        scope = 'declared IDs plus conservative interchange pool exclusions'
    return dict(manifest_path=str(run_path), manifest_sha256=file_hash(run_path),
                checkpoint_manifest_path=str(update_path), checkpoint_manifest_sha256=file_hash(update_path),
                arm=arm, optimizer_updates=updates, training_exposure=exposure,
                exposure_sources=sources, exposure_scope=scope), inputs


def leakage_check(cases, evaluated_adapter):
    training = evaluated_adapter['training']
    exposure = training['training_exposure'] if training else dict(task_ids=[], group_ids=[])
    tasks, groups = set(exposure['task_ids']), set(exposure['group_ids'])
    evaluated_tasks = sorted({c['spec']['task_id'] for c in cases})
    evaluated_groups = sorted({str(Path(t).parent) for t in evaluated_tasks})
    overlap_tasks = sorted(tasks.intersection(evaluated_tasks))
    overlap_groups = sorted(groups.intersection(evaluated_groups))
    if overlap_tasks or overlap_groups:
        raise ValueError('Evaluation overlaps adapter training exposure: tasks=' +
                         repr(overlap_tasks) + ', groups=' + repr(overlap_groups))
    return dict(status='passed' if training and training['exposure_sources'] else 'no_exposure_ids_declared',
                evaluated_task_ids=evaluated_tasks, evaluated_group_ids=evaluated_groups,
                training_task_ids=sorted(tasks), training_group_ids=sorted(groups),
                overlapping_task_ids=overlap_tasks, overlapping_group_ids=overlap_groups,
                scope='all evaluated cases, including from-start; task and group disjointness')


def serving_spec(args, config):
    """Coverage's LoRA gate, without requiring the evaluated weights to be theta0."""
    from students.vllm_student import Student, context_options
    from tools.check_lora_serving import validate_report
    fields = (args.endpoint, args.server_provenance, args.lora_check, args.model_name)
    if args.dry_run and not any(fields):
        return {}, dict(name='synthetic-eval-adapter', path=None, adapter_weights_sha256=None,
                        adapter_config_sha256=None, training=None, synthetic=True), []
    if not all(fields):
        raise ValueError('Adapter evaluation requires endpoint, server-provenance, lora-check and model-name')
    server, report = read_json(args.server_provenance), read_json(args.lora_check)
    model = config['models']['student']
    spec = dict(endpoint=args.endpoint, model=args.model_name, revision=model['revision'],
                enable_thinking=False, max_tokens=1024, **context_options(config))
    Student(log_dir=Path(args.output) / 'unused', **spec)
    validate_report(report, server, spec['model'], spec['endpoint'])
    checked = report['identity']['adapter']
    if checked['base_model'] != model['requested_repo'] or checked['revision'] != model['revision']:
        raise ValueError('Adapter differs from configured student')
    adapter = next(a for a in server['config']['server']['lora_adapters'] if a['name'] == args.model_name)
    training, inputs = training_identity(adapter)
    config['server_provenance'], config['lora_equivalence'] = server, report
    identity = {k: adapter[k] for k in ('name', 'path', 'adapter_weights_sha256', 'adapter_config_sha256',
                                       'base_model', 'revision')}
    identity.update(training=training, synthetic=False)
    return spec, identity, [args.server_provenance, args.lora_check, *inputs]
