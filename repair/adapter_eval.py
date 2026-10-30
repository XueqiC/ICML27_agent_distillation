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


def _exposure(manifest):
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


def training_identity(adapter):
    checkpoint = Path(adapter['path']).resolve()
    if checkpoint.name != 'checkpoint' or not re.fullmatch(r'update-\d+', checkpoint.parent.name):
        return None, []
    # A recognizable V2 checkpoint may not silently lose its training manifest.
    run_path, update_path = checkpoint.parent.parent / 'manifest.json', checkpoint.parent / 'manifest.json'
    run, update = read_json(run_path), read_json(update_path)
    if run.get('schema') != 'v2-training-run/1':
        raise ValueError('Expected V2 training run manifest')
    updates = update.get('optimizer_updates')
    if (type(updates) is not int or updates < 0 or updates != int(checkpoint.parent.name.split('-')[1])
            or not isinstance(run.get('arm'), str) or not run['arm']):
        raise ValueError('Invalid V2 arm or checkpoint optimizer_updates')
    for name, field in [('adapter_model.safetensors', 'adapter_weights_sha256'),
                        ('adapter_config.json', 'adapter_config_sha256')]:
        if update.get('sha256', {}).get(name) != adapter[field]:
            raise ValueError('V2 checkpoint manifest differs from evaluated adapter')
    exposure, sources = _exposure(run)
    update_exposure, update_sources = _exposure(update)
    for key in exposure:
        exposure[key] = sorted(set(exposure[key]) | set(update_exposure[key]))
    return dict(manifest_path=str(run_path), manifest_sha256=file_hash(run_path),
                checkpoint_manifest_path=str(update_path), checkpoint_manifest_sha256=file_hash(update_path),
                arm=run['arm'], optimizer_updates=updates, training_exposure=exposure,
                exposure_sources=['run.' + s for s in sources] + ['checkpoint.' + s for s in update_sources],
                exposure_scope='declared IDs plus conservative interchange pool exclusions'), [str(run_path), str(update_path)]


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
