"""Pilot branch contracts and path-independent base snapshot identity."""
from pathlib import Path

from sed.manifest import authorize_task, resolve_task

BRANCHES = {
    'P0_K20': ('P0', 'adapter', 'repair_tasks'),
    'P1_U_mech': ('P1', 'adapter', 'original_support'),
    'P0_K0': ('P0', 'base_model', 'repair_tasks'),
}
ADAPTER_KEYS = ('adapter_config_sha256', 'adapter_weights_sha256', 'base_model', 'revision',
                'training_state_sha256', 'training_provenance_sha256')


def branch_fields(branch):
    if branch not in BRANCHES:
        raise ValueError('Unknown repair branch')
    return dict(zip(('branch', 'protocol', 'theta0_kind', 'seed_replay_scope'),
                    (branch, *BRANCHES[branch])))


def collection_branch(value):
    # Unlabelled historical collections belong only to the original P0 branch.
    fields = branch_fields(value.get('branch', 'P0_K20'))
    if any(k in value and value[k] != v for k, v in fields.items()):
        raise ValueError('Inconsistent repair branch contract')
    if fields['branch'] != 'P0_K20' and any(k not in value for k in fields):
        raise ValueError('New branches require explicit branch metadata')
    return fields


def base_checkpoint(snapshot, student):
    from train.modeling import snapshot_metadata
    path = Path(snapshot)
    if any((path / name).exists() for name in
           ('adapter_config.json', 'adapter_model.safetensors', 'training_state.json')):
        raise ValueError('Base snapshot cannot contain adapter or training state')
    metadata = snapshot_metadata(path)
    return {'identity': {'base_model': student['requested_repo'], 'revision': student['revision'],
                        'metadata_sha256': metadata['metadata_sha256'],
                        'weight_verification': metadata['weight_verification']},
            'files': metadata['metadata_sha256']}


def validate_base_serving(config, spec, *, expected=None, snapshot=None):
    server_config = config.get('server_provenance', {}).get('config', {})
    server = server_config.get('server', {})
    student = config['models']['student']
    if (server.get('lora_adapters') != [] or server.get('merged_model') is not False
            or server.get('checkpoint_kind') != 'base' or server.get('training_provenance_sha256')
            or spec.get('model') != student['requested_repo']
            or spec.get('revision') != student['revision']
            or server.get('served_model_name') != student['requested_repo']
            or server.get('endpoint') != spec['endpoint'].rstrip('/')
            or server_config.get('models', {}).get('student') != student
            or config.get('lora_equivalence')):
        raise ValueError('P0_K0 requires the configured base model with no LoRA adapter')
    checkpoint = base_checkpoint(snapshot or server['model_path'], student)
    if server.get('model_metadata_sha256') != checkpoint['files']:
        raise ValueError('Served base snapshot metadata differs')
    if expected is not None and checkpoint != expected:
        raise ValueError('Checkpoint zero/coverage is not collection base theta0')
    return checkpoint


def validate_coverage_branch(branch, summary, provenance, plan, seed_collection,
                             splits, manifest, freeze, checkpoint, *, snapshot=None):
    """Check the complete coverage population, not only selected failures."""
    branch_fields(branch)
    cfg = provenance['config']
    if branch != 'P0_K0':
        identity = cfg['lora_equivalence']['identity']['adapter']
        if any(identity.get(k) != checkpoint['identity'].get(k) for k in ADAPTER_KEYS):
            raise ValueError('Coverage was not collected from theta0')
    else:
        validate_base_serving(cfg, summary['student'], expected=checkpoint, snapshot=snapshot)
    seed_ids = {r['task_id'] for r in seed_collection['verified']}
    failure_ids = {r['task_id'] for r in summary['failure_list']}
    if branch == 'P0_K20':
        if not failure_ids <= seed_ids:
            raise ValueError('P0 collector failures must belong to the supplied K teacher seeds')
        return
    if seed_collection['K'] != 20 or len(seed_collection['verified']) != 20 or len(seed_ids) != 20:
        raise ValueError('New pilot branches require the original K20 trusted demonstrations')
    recorded = cfg['repair_coverage']
    if (any(recorded.get(k) != v for k, v in plan.items())
            or summary['student'] != plan['student']
            or summary['task_source'] != recorded['task_source']):
        raise ValueError('Coverage plan, summary and provenance disagree')
    expected = set(splits['sets']['U_mech']['task_ids']) if branch == 'P1_U_mech' else seed_ids
    if len(expected) != 20:
        raise ValueError('Pilot coverage requires exactly 20 frozen tasks')
    if (summary['task_source'] != ('U_mech' if branch == 'P1_U_mech' else 'K20_seed_tasks')
            or summary['protocol'] != branch_fields(branch)['protocol']):
        raise ValueError('Coverage task_source/protocol differs from branch')
    from tools.repair_coverage import attempts
    schedule = attempts(plan['extra_sampled'])
    rows = plan['episodes']
    if ({r['attempt'] for r in rows} != {a['attempt'] for a in schedule}
            or not failure_ids <= expected):
        raise ValueError('Coverage tasks differ from the frozen branch set')
    for attempt in schedule:
        ids = [r['task_id'] for r in rows if r['attempt'] == attempt['attempt']]
        if len(ids) != 20 or set(ids) != expected:
            raise ValueError('Coverage tasks must be exactly the frozen branch set on every attempt')
    tasks = [resolve_task(manifest, freeze, tid) for tid in expected]
    groups = {t['group_id'] for t in tasks}
    blocked = set(splits['sets']['D_probe']['group_ids'])
    blocked |= {t['group_id'] for t in manifest['tasks']
                if t['source_split'] != 'train'
                or freeze['assignments'].get(t['task_id'], {}).get('sealed', False)
                or freeze['assignments'].get(t['task_id'], {}).get('split', 'train') != 'train'}
    if branch == 'P1_U_mech':
        blocked |= {resolve_task(manifest, freeze, t)['group_id'] for t in seed_ids}
        blocked |= set(splits.get('E', {}).get('group_ids', []))
    for task in tasks:
        authorize_task(task, 'generation')
    if len(groups) != 20 or groups & blocked:
        raise ValueError('Coverage branch overlaps seed/probe/dev groups or repeats a group')
    for row in rows:
        task = resolve_task(manifest, freeze, row['task_id'])
        if row['group_id'] != task['group_id']:
            raise ValueError('Coverage group differs from frozen task identity')
