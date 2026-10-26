"""Read the sealed E2 interchange, without importing the evaluation branch.

Receipts and approvals are evidence only. Only public_episode's schema-2
projection reaches this tree's unchanged episode_examples encoder.
"""
from copy import deepcopy
from pathlib import Path
import re

from baselines.common.conversation import group_turns
from harness.lora_adapters import inspect_adapter, adapter_identity
from repair.artifacts import checked_document, checkpoint_files, tokenizer_identity, read_json
from sed.actions import parse_action
from sed.manifest import authorize_task, load_frozen, resolve_task
from sed.seed_evidence import load_verified_collection, checked_episode
from sed.storage import digest, file_hash
from sed.visibility import visible_observation
from tools.repair_coverage import checked_repair_splits
from train.dataset import episode_examples


def require(condition, message):
    if not condition:
        raise ValueError(message)


def remap(path, mappings=()):
    """Explicit host-root relocation; identity remains the original byte hashes."""
    path = Path(path).expanduser()
    for old, new in sorted(mappings, key=lambda pair: len(str(pair[0])), reverse=True):
        try:
            return (Path(new) / path.relative_to(old)).resolve()
        except ValueError:
            continue
    return path.resolve()


def public_episode(episode, *, strict=False):
    require(type(episode.get('schema_version')) is int and episode['schema_version'] == 2
            and episode.get('split') == 'train' and episode.get('sealed') is False,
            'Expected unsealed schema-2 TRAIN episode')
    require(isinstance(episode.get('steps'), list) and bool(episode['steps']), 'Empty episode')
    steps = []
    for step in episode['steps']:
        require(isinstance(step.get('event_id'), str) and bool(step['event_id'])
                and isinstance(step.get('action_raw'), str), 'Invalid public decision')
        require(parse_action(step['action_raw']) is not None, 'Invalid public action')
        steps.append({'event_id': step['event_id'], 'action_raw': step['action_raw'],
                      'visible': visible_observation(step['visible'])})
    require(len({s['event_id'] for s in steps}) == len(steps), 'Duplicate event IDs')
    result = {'schema_version': 2, 'task_id': episode['task_id'], 'split': 'train', 'sealed': False,
              'initial': {'visible': visible_observation(episode['initial']['visible'])}, 'steps': steps}
    if strict:
        require(result == episode, 'Repair episode differs from schema-2 whitelist')
    return result


def approved_entry(entry, task):
    require(entry['task_id'] == task['task_id'] and entry['group_id'] == task['group_id'],
            'Arm task/group differs from frozen task')
    require(entry['source_bucket'] == 'successful_repair', 'Unexpected source bucket')
    episode = public_episode(entry['episode'], strict=True)
    require(episode['task_id'] == entry['task_id'], 'Episode task differs')
    require(entry['episode_sha256'] == digest(episode), 'Changed episode hash')
    review = entry['human_review']
    require(review.get('status') == 'approved' and review.get('reviewed') is True
            and review.get('approved') is True, 'Target is not human approved')
    require(review.get('episode_sha256') == entry['episode_sha256'], 'Review episode hash differs')
    require(isinstance(review.get('evidence_sha256'), str)
            and re.fullmatch('[0-9a-f]{64}', review['evidence_sha256']), 'Missing review evidence hash')
    decision = review['decision']
    require(decision.get('trajectory_id') == entry['trajectory_id']
            and decision.get('episode_id') == entry['learning_episode_id'], 'Review identity differs')
    require(all(decision.get(k) is True for k in
                ('no_rejected_actions', 'meaningful_route', 'labels_supported')),
            'Human review decisions are not approved')
    require(all(isinstance(decision.get(k), str) and decision[k].strip()
                for k in ('reviewer', 'notes')), 'Missing human review attribution')
    require(all(isinstance(entry.get(k), str) and entry[k] for k in
                ('learning_episode_id', 'repair_id', 'trajectory_id')), 'Missing target identity')
    return episode


def read_collection(path, *, config, theta0, snapshot, tokenizer,
                    repair_splits='data/manifest/repair_v01_splits.json', path_maps=()):
    """All gates apply even to a training dry-run; synthetic exports are refused."""
    try:
        return _read_collection(path, config=config, theta0=theta0, snapshot=snapshot,
                                tokenizer=tokenizer, repair_splits=repair_splits, path_maps=path_maps)
    except (KeyError, TypeError, IndexError, AttributeError, OSError):
        raise ValueError('Incomplete or malformed E2 collection/evidence') from None


def _read_collection(path, *, config, theta0, snapshot, tokenizer, repair_splits, path_maps):
    path = Path(path).resolve()
    data = checked_document(path)
    require(data.get('format') == 'repair_e2_training_collection'
            and type(data.get('schema_version')) is int and data['schema_version'] == 1,
            'Unknown E2 format/version')
    source_path = path.with_name('collection.json')
    source = checked_document(source_path)
    require(file_hash(source_path) == data['source_collection_sha256']
            and source['content_sha256'] == data['source_collection_content_sha256'],
            'Source collection mismatch')
    require(data.get('dry_run') is False and data.get('canary') is False
            and not data.get('synthetic') and source.get('dry_run') is False
            and source.get('canary') is False and not source.get('synthetic'),
            'Synthetic/canary collection is not trainable')
    require(data.get('ready_for_training') is True and data.get('training_blockers') == []
            and type(data.get('training_exposure')) is int and data['training_exposure'] == 0,
            'Collection is not ready for training')
    for exported, original in [('theta0', 'checkpoint'), ('tokenizer', 'tokenizer'),
                               ('split', 'split'), ('D_repair_train', 'D_repair_train'),
                               ('D_repair_probe', 'D_repair_probe'), ('input_hashes', 'input_hashes')]:
        require(data[exported] == source[original], 'Source collection metadata mismatch: ' + exported)
    manifest, freeze = load_frozen(config['paths']['manifest'], config['paths']['split_freeze'])
    splits = checked_repair_splits(repair_splits, manifest, freeze)
    # Bind the local frozen files to the collection's input receipts. Other
    # input_hashes may be evaluation-branch code; retain, never import it.
    for name in (config['paths']['manifest'], config['paths']['split_freeze'], repair_splits):
        require(any(remap(p, path_maps) == Path(name).resolve() and h == file_hash(name)
                    for p, h in data['input_hashes'].items()), 'Frozen input hash mismatch')
    train, probes, split = data['D_repair_train'], data['D_repair_probe'], data['split']
    def ids(value, size=None):
        return (isinstance(value, list) and all(isinstance(t, str) and t for t in value)
                and len(set(value)) == len(value) and (size is None or len(value) == size))
    require(ids(train, 4) and ids(probes, 2), 'Missing/duplicate training or probe tasks')
    order = split['failure_order']
    require(ids(order) and 6 <= len(order) <= 20, 'Invalid failure task order')
    expected_probes = sorted(order, key=lambda t: (digest([20261029, t]), t))[:2]
    require(split['seed'] == 20261029 and bool(split['rule'])
            and probes == split['D_repair_probe'] == expected_probes
            and split['training_candidates'] == [t for t in order if t not in probes]
            and [t for t in split['training_candidates'] if t in train] == train,
            'Misordered tasks or changed frozen split')
    tasks = {tid: resolve_task(manifest, freeze, tid) for tid in order}
    blocked = set(splits['sets']['D_probe']['group_ids'])
    for task in tasks.values():
        authorize_task(task, 'generation')
        require(task['source_split'] == 'train' and task['group_id'] not in blocked,
                'Independent probe/evaluation task cannot supply training')
    require(not set(train) & set(probes) and not
            {tasks[t]['group_id'] for t in train} & {tasks[t]['group_id'] for t in probes},
            'Train/probe overlap including groups')
    require(len({t['group_id'] for t in tasks.values()}) == len(order), 'Duplicate failure task groups')
    require(data['unique_instances'] == len(train)
            and data['task_groups'] == len({tasks[t]['group_id'] for t in train}), 'Incorrect task counts')
    require(set(data['arms']) == {'N', 'F'}, 'Expected exactly N/F arms')
    episodes = {}
    for arm in ('N', 'F'):
        entries = data['arms'][arm]
        require([e['task_id'] for e in entries] == train, 'Missing/duplicate/misordered arm tasks')
        episodes[arm] = [approved_entry(e, tasks[e['task_id']]) for e in entries]
    require([e['seed_demonstrations'] for e in data['arms']['N']] ==
            [e['seed_demonstrations'] for e in data['arms']['F']], 'Arms have different seed references')
    identity = inspect_adapter(data['theta0']['identity']['name'], theta0, config['models']['student'], snapshot)
    require(adapter_identity(identity) == adapter_identity(data['theta0']['identity'])
            and checkpoint_files(theta0) == data['theta0']['files'], 'Changed theta0 files/base revision')
    require(tokenizer_identity(snapshot, tokenizer) == data['tokenizer'], 'Changed tokenizer/template identity')
    theta_provenance = read_json(Path(theta0).parent / 'provenance.json')
    original_support = str(Path(theta_provenance['config']['baseline_training']['verified']) / 'manifest.json')
    original_support_hash = theta_provenance['inputs'][original_support]
    # Verify all original K evidence through the existing loader, then select
    # before encoding. Nested K views resolve to the loader's evidence root.
    collections, seeds, evidence_inputs = {}, {}, set()
    for entry in data['arms']['N']:
        tid = entry['task_id']
        references = entry['seed_demonstrations']
        require(isinstance(references, list) and bool(references), 'Absent seed evidence')
        seeds[tid] = []
        seen = set()
        for ref in references:
            root = remap(ref['verified_collection_path'], path_maps)
            require(file_hash(root / 'manifest.json') == ref['verified_manifest_sha256'],
                    'Changed verified manifest')
            require(ref['verified_manifest_sha256'] == original_support_hash,
                    'Seed evidence is not theta0 original support')
            if root not in collections:
                collections[root] = load_verified_collection(root, manifest, freeze)
            verified, _, evidence_root = collections[root]
            evidence_root = Path(evidence_root).resolve()
            require(type(data['K_trusted_seed_trajectories']) is int
                    and verified['K'] == data['K_trusted_seed_trajectories'] == source['K'],
                    'Original trusted K differs')
            require(remap(ref['evidence_root'], path_maps) == evidence_root, 'Wrong evidence root')
            receipt = ref['receipt']
            require(digest(receipt) == ref['receipt_sha256'] and receipt in verified['verified'],
                    'Seed receipt is not original verified evidence')
            directory = (evidence_root / ref['episode_dir']).resolve()
            require(not Path(ref['episode_dir']).is_absolute() and directory.is_relative_to(evidence_root)
                    and ref['episode_dir'] == receipt['episode_dir'], 'Seed path escapes evidence root')
            require(ref['task_id'] == receipt['task_id'] == tid
                    and receipt['group_id'] == tasks[tid]['group_id'], 'Seed task/group outside training scope')
            require(file_hash(directory / 'episode.json') == ref['episode_sha256'], 'Changed seed episode')
            episode = checked_episode(evidence_root, receipt)
            require(episode['task_id'] == tid and episode['episode_id'] == ref['episode_id'],
                    'Seed episode identity differs')
            require(ref['episode_id'] not in seen, 'Duplicate seed reference')
            seen.add(ref['episode_id'])
            seeds[tid].append(public_episode(episode))
            evidence_inputs.update([root / 'manifest.json', directory / 'episode.json'])
    require(len(collections) == 1, 'Seed replay must use one original support collection')
    return {'document': deepcopy(data), 'episodes': episodes, 'seeds': seeds,
            'tasks': [tasks[t] for t in train], 'path': str(path),
            'training_collection_sha256': file_hash(path),
            'inputs': [str(p) for p in [path, source_path, *sorted(evidence_inputs)]]}


def prepare_examples(collection, tokenizer, arm, *, packing, max_length):
    require(arm in {'N', 'F'}, 'Expected one N/F arm')
    examples, slots = [], {}
    def add(episode, bucket, reference):
        rows = episode_examples(public_episode(episode), tokenizer, packing=packing, max_length=max_length)
        row = group_turns(rows) if packing == 'turn' else rows[0]
        row['label_provenance'].update(bucket=bucket,
            source='successful_repair' if bucket == 'repair' else 'trusted_teacher_seed', evidence=reference)
        slots.setdefault((episode['task_id'], bucket), []).append(len(examples))
        examples.append(row)
    document = collection['document']
    for episode, entry in zip(collection['episodes'][arm], document['arms'][arm]):
        add(episode, 'repair', {k: entry[k] for k in
            ('episode_sha256', 'learning_episode_id', 'repair_id', 'trajectory_id')})
    for entry in document['arms'][arm]:
        for episode, reference in zip(collection['seeds'][entry['task_id']], entry['seed_demonstrations']):
            add(episode, 'seed', {k: reference[k] for k in ('episode_id', 'episode_sha256', 'receipt_sha256')})
    return examples, slots
