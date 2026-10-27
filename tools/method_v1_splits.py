"""Exclusive method-v1 freeze, reusing the repair-v01 metadata-only builder."""
import argparse
import json
from pathlib import Path

from data.manifest import build_repair_splits as base
from sed.manifest import load_frozen
from sed.storage import digest, file_hash, write_json

SUBSETS = (10, 20, 30, 100)
TYPE_ORDER = tuple(base.KINDS)
NAME_EXCLUSION_SETS = ('dev140', 'D_probe', 'D_unseen_quick', 'U_mech')


def read_json(path):
    return json.loads(Path(path).read_text())


def task_name(group_id):
    """Keep the full type/object/movable/receptacle/scene name, without its split."""
    return group_id.split('/', 1)[1]


def equal_quotas(total):
    """Equal type quotas, with surplus slots in canonical task-type order."""
    quota, surplus = divmod(total, len(TYPE_ORDER))
    return {kind: quota + (i < surplus) for i, kind in enumerate(TYPE_ORDER)}


def select_exact(name, candidates, quotas, seed):
    rows = base.select(candidates, quotas, seed)
    counts = base.population_counts(rows)['groups_by_task_type']
    shortfalls = {k: quotas[k] - counts[k] for k in TYPE_ORDER if counts[k] < quotas[k]}
    if shortfalls:
        raise ValueError(f'{name} task-group quota shortfalls: {json.dumps(shortfalls, sort_keys=True)}')
    return rows


def validate_repair(document, manifest, freeze, dev140):
    if document['builder_sha256'] != file_hash(base.__file__):
        raise ValueError('Repair builder hash differs')
    if document['content_sha256'] != digest({k: v for k, v in document.items() if k != 'content_sha256'}):
        raise ValueError('Repair content hash differs')
    sources = []
    for old in document['exclusion_sources']:
        source = base.read_source(old['path'], old['category'], manifest, freeze,
                                 selected_only=old['extraction_rule'] == 'selected/task_order only')
        if source['sha256'] != old['sha256']:
            raise ValueError('Repair exclusion source changed')
        sources.append(source)
    rebuilt = base.build_splits(manifest, freeze, sources, dev140)
    if any(document[k] != v for k, v in rebuilt.items()):
        raise ValueError('Repair manifest cannot be reproduced')
    return sources


def listing_source(path, category, rule, task_ids, group_ids):
    ids, groups = sorted(set(task_ids)), sorted(set(group_ids))
    return dict(path=str(path), category=category, sha256=file_hash(path), extraction_rule=rule,
                task_ids=ids, list_sha256=digest(ids), unique_instances=len(ids),
                group_ids=groups, group_list_sha256=digest(groups), task_groups=len(groups))


def exclusion_sources(repair, repair_path, manifest, freeze, freeze_path, dev140_path,
                      seed_dir, verified_root):
    """Method exclusions are independent of repair's historical K500/R03 exclusions.

    Seed declarations contain the whole replacement universe: use their selected
    views only. Collection manifests also exclude actually attempted groups.
    Absent standalone K30 declarations are disclosed; the K30 full-TRAIN view
    remains required. No K300/K500 selection or collection is an exclusion.
    """
    sources, absent = [], []
    for name in ('D_probe', 'D_unseen_quick', 'U_mech'):
        listing = repair['sets'][name]
        sources.append(listing_source(repair_path, name, 'repair sets.' + name,
                                      listing['task_ids'], listing['group_ids']))
    root = Path(verified_root)
    for k in SUBSETS:
        for name in ('selection.json', 'manifest.json'):
            sources.append(base.read_source(root / f'subsets/k{k}' / name, f'full_train_k{k}_{name}',
                                            manifest, freeze, selected_only=False))
    for k in (5, *SUBSETS):
        for seed in range(3):
            path = Path(seed_dir) / f'seeds_k{k}_s{seed}.json'
            if not path.exists() and k in (5, 30):
                absent.append(dict(path=str(path), category='seed_selection',
                                   reason='optional standalone view does not exist'))
                continue
            sources.append(base.read_source(path, 'seed_selection', manifest, freeze))
    for k in (20, 100):
        for seed in range(3):
            directory = root.parent / f'seeds_k{k}_s{seed}'
            if not directory.is_dir():
                raise FileNotFoundError(directory)
            views = [directory]
            # Include smaller views actually used by the seed grids (K1/2/5/10/20/50).
            for path in sorted(directory.glob('subsets/k*')):
                if path.is_dir() and path.name[1:].isdigit() and 0 < int(path.name[1:]) <= 100:
                    views.append(path)
            for view in views:
                for name in ('selection.json', 'manifest.json'):
                    sources.append(base.read_source(view / name, 'verified_seed_view', manifest, freeze,
                                                    selected_only=name == 'selection.json'))
    sources.append(base.read_source(dev140_path, 'dev140', manifest, freeze, selected_only=False))
    tasks = {t['task_id']: t for t in manifest['tasks']}
    eval_ids = [tid for tid, a in freeze['assignments'].items() if a['sealed'] or a['split'] != 'train']
    sources.append(listing_source(freeze_path, 'evaluation_or_sealed',
                                  'assignments with sealed=true or split!=train', eval_ids,
                                  [tasks[tid]['group_id'] for tid in eval_ids]))
    return sources, absent


def verified_trajectories(root, manifest, freeze):
    """Read successful collection receipts, never game state or prompt inputs."""
    root = Path(root).resolve()
    collection = read_json(root / 'manifest.json')
    base.read_source(root / 'selection.json', 'full_train_candidates', manifest, freeze)
    if collection['selection_sha256'] != digest(read_json(root / 'selection.json')) or collection['status'] != 'complete':
        raise ValueError('Incomplete or changed full-TRAIN collection')
    tasks = {t['task_id']: t for t in manifest['tasks']}
    result, hashes = {}, {str(root / p): file_hash(root / p) for p in ('manifest.json', 'selection.json')}
    for receipt in collection['verified']:
        task = tasks[receipt['task_id']]
        if (receipt['group_id'] != task['group_id'] or task['source_split'] != 'train'
                or freeze['assignments'][task['task_id']]['split'] != 'train'
                or freeze['assignments'][task['task_id']]['sealed']
                or receipt['success'] is not True or receipt['won'] is not True
                or receipt['status'] != 'ok' or receipt['rejection'] is not None
                or receipt['source'] != 'real_environment_teacher_rollout'):
            raise ValueError('Invalid verified teacher receipt')
        directory = (root / receipt['episode_dir']).resolve()
        if not directory.is_relative_to(root):
            raise ValueError('Episode escapes collection')
        # Verify the recorded episode, event and provenance receipts. No game/PDDL reads.
        for name in ('episode.json', 'events.jsonl', 'provenance.json'):
            p = directory / name
            if not p.resolve().is_relative_to(root) or file_hash(p) != receipt['episode_hashes'][name]:
                raise ValueError('Teacher execution evidence changed')
            hashes[str(p)] = receipt['episode_hashes'][name]
        episode = read_json(directory / 'episode.json')
        if (episode['task_id'] != task['task_id'] or episode['game_sha256'] != task['game_sha256']
                or episode['success'] is not True or episode['complete'] is not True
                or episode['status'] != 'ok' or episode['split'] != 'train' or episode['sealed']):
            raise ValueError('Verified teacher episode is not a complete TRAIN success')
        entry = dict(task_id=task['task_id'], group_id=task['group_id'], trajectory_id=episode['episode_id'],
                     episode_path=str(directory / 'episode.json'),
                     episode_sha256=receipt['episode_hashes']['episode.json'], receipt_sha256=digest(receipt))
        result.setdefault(task['task_id'], []).append(entry)
    return {tid: min(rows, key=lambda r: (r['trajectory_id'], r['episode_path'])) for tid, rows in result.items()}, hashes


def build_splits(manifest, freeze, sources, dev140, verified):
    # Execute the original checks, group exclusions, and universe validation first.
    checked = base.build_splits(manifest, freeze, sources, dev140)
    tasks = {t['task_id']: t for t in manifest['tasks']}
    if any(tid not in tasks or row['task_id'] != tid or row['group_id'] != tasks[tid]['group_id']
           or not row.get('trajectory_id') for tid, row in verified.items()):
        raise ValueError('Verified trajectory identity differs from frozen task')
    blocked = set(checked['E']['group_ids']) | {tasks[t]['group_id'] for t, a in
        freeze['assignments'].items() if a['split'] != 'train'} | {tasks[t]['group_id'] for t in dev140['task_ids']}
    eligible = [t for t in manifest['tasks'] if t['eligible'] and t['source_split'] == 'train'
        and freeze['assignments'].get(t['task_id'], {}).get('split') == 'train'
        and not freeze['assignments'][t['task_id']]['sealed'] and t['group_id'] not in blocked]
    blocked_names = {name: set() for name in NAME_EXCLUSION_SETS}
    blocked_names['dev140'] = {task_name(tasks[tid]['group_id']) for tid in dev140['task_ids']}
    for source in sources:
        category = source.get('category')
        if category in NAME_EXCLUSION_SETS[1:]:
            groups = set(source.get('group_ids', [])) | {tasks[tid]['group_id'] for tid in source['task_ids']}
            blocked_names[category].update(task_name(gid) for gid in groups)
    all_blocked_names = set().union(*blocked_names.values())
    excluded = [t for t in eligible if task_name(t['group_id']) in all_blocked_names]
    excluded_names = {task_name(t['group_id']) for t in excluded}
    name_exclusions = dict(
        rule='exact group_id after the first split/ prefix; all sibling trials excluded before sampling',
        count_scope='additional exclusions from eligible TRAIN candidates after group-level exclusions',
        dev140_content_sha256=digest(dev140),
        source_hashes=[{k: s[k] for k in ('category', 'path', 'sha256', 'extraction_rule')}
                       for s in sources if s.get('category') in NAME_EXCLUSION_SETS],
        excluded_task_names=sorted(excluded_names), excluded_task_name_count=len(excluded_names),
        excluded_population=base.population_counts(excluded),
        excluded_task_name_counts_by_source={name: len(excluded_names & names)
                                            for name, names in blocked_names.items()})
    eligible = [t for t in eligible if task_name(t['group_id']) not in all_blocked_names]
    dev_quotas = equal_quotas(60)
    dev = select_exact('U_dev', eligible, dev_quotas, 20261101)
    dev_groups = {t['group_id'] for t in dev}
    pool = [t for t in eligible if t['group_id'] not in dev_groups and t['task_id'] in verified]
    counts = base.population_counts(pool)
    quotas = equal_quotas(200)
    training = select_exact('V2_pool', pool, quotas, 20261102)
    sets = {}
    for name, rows, qs, seed, purpose, train in (
        ('U_dev', dev, dev_quotas, 20261101, 'diagnosis_only', False),
        ('V2_pool', training, quotas, 20261102, 'method_v1_training_pool', True)):
        listing = base.task_listing(name, rows, qs, seed, manifest, freeze, purpose=purpose)
        listing.update(training_eligible=train, requested_task_groups=60 if name == 'U_dev' else 200,
                       total_shortfall=(60 if name == 'U_dev' else 200) - len(rows))
        sets[name] = listing
    sets['V2_pool']['verified_trajectories'] = [verified[t['task_id']] for t in training]
    overlaps = {name + '_vs_' + str(i): len(set(s['group_ids']) & set(source['group_ids']))
        for name, s in sets.items() for i, source in enumerate(sources)}
    overlaps['U_dev_vs_V2_pool'] = len(dev_groups & set(sets['V2_pool']['group_ids']))
    evaluation_groups = {tasks[t]['group_id'] for t, a in freeze['assignments'].items()
                         if a['sealed'] or a['split'] != 'train'}
    dev140_groups = {tasks[t]['group_id'] for t in dev140['task_ids']}
    for name, listing in sets.items():
        overlaps[name + '_vs_evaluation_or_sealed'] = len(set(listing['group_ids']) & evaluation_groups)
        overlaps[name + '_vs_dev140'] = len(set(listing['group_ids']) & dev140_groups)
    name_overlaps = {name + '_vs_' + source: len({task_name(gid) for gid in listing['group_ids']} & names)
                     for name, listing in sets.items() for source, names in blocked_names.items()}
    name_overlaps['U_dev_vs_V2_pool'] = len({task_name(gid) for gid in dev_groups}
        & {task_name(gid) for gid in sets['V2_pool']['group_ids']})
    if any(name_overlaps.values()):
        raise ValueError('Method split task-name overlap')
    if any(overlaps.values()) or any(s['unique_instances'] != s['task_groups'] for s in sets.values()):
        raise ValueError('Method split overlap or duplicate group')
    return dict(schema_version=1, name='method_v1_splits', manifest_sha256=digest(manifest),
        split_freeze_sha256=digest(freeze), universe=checked['universe'],
        eligible_after_exclusions=base.population_counts(eligible), v2_eligible_after_U_dev=counts,
        selection_rule={'ordering': base.ORDER_RULE, 'type_order': list(TYPE_ORDER),
            'U_dev_quotas': 'exactly 10 task groups per type',
            'V2_quotas': 'equal quotas: 34 for the first two canonical types, 33 for the other four',
            'V2_trial': 'hash order among verified won trials; exact selected trial reused offline',
            'shortfall': 'fail with per-type shortfalls; no redistribution',
            'exclusions': 'repair sets D_probe/D_unseen_quick/U_mech; all K<=100 main-grid and seed views; dev140/evaluation/sealed; task names ignoring split prefix against dev140/D_probe/D_unseen_quick/U_mech; no inherited repair exclusions or K300/K500-only groups'},
        exclusion_sources=checked['exclusion_sources'], task_name_exclusions=name_exclusions, sets=sets,
        assertions={'all_passed': True, 'overlap_counts': overlaps, 'train_only': True,
                    'task_name_overlap_counts': name_overlaps,
                    'one_trial_per_group': True, 'v2_has_verified_won_trajectory': True},
        new_teacher_calls=0, new_environment_steps=0, training_exposure=0)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--repair-splits', default='data/manifest/repair_v01_splits.json')
    p.add_argument('--manifest', default='data/manifest/alfworld_manifest.json')
    p.add_argument('--freeze', default='data/manifest/alfworld_split_freeze.json')
    p.add_argument('--dev140', default='data/manifest/dev140.json')
    p.add_argument('--seed-dir', default='data/manifest')
    p.add_argument('--verified-root', default='data/verified/seeds_full_train_r2')
    p.add_argument('--output', default='data/manifest/method_v1_splits.json')
    args = p.parse_args(argv)
    if Path(args.output).exists():
        raise FileExistsError('Output must be a new manifest')
    manifest, freeze = load_frozen(args.manifest, args.freeze)
    repair, dev140 = read_json(args.repair_splits), read_json(args.dev140)
    repair_sources = validate_repair(repair, manifest, freeze, dev140)
    sources, absent = exclusion_sources(repair, args.repair_splits, manifest, freeze, args.freeze,
                                         args.dev140, args.seed_dir, args.verified_root)
    root = Path(args.verified_root)
    verified, hashes = verified_trajectories(root, manifest, freeze)
    result = build_splits(manifest, freeze, sources, dev140, verified)
    hashes.update({str(p): file_hash(p) for p in (args.manifest, args.freeze, args.dev140, args.repair_splits)})
    hashes.update({s['path']: s['sha256'] for s in sources})
    hashes.update({s['path']: s['sha256'] for s in repair_sources})
    result.update(input_sha256=hashes, builder_sha256=file_hash(__file__),
                  imported_builder_sha256=file_hash(base.__file__), rebuild_arguments=vars(args),
                  absent_optional_sources=absent,
                  repair_validation_sources=[{k: s[k] for k in ('path', 'sha256', 'extraction_rule')}
                                             for s in repair_sources])
    result['content_sha256'] = digest(result)
    write_json(args.output, result, exclusive=True)
    print(json.dumps({'output': args.output, 'counts': {k: s['task_groups'] for k, s in result['sets'].items()},
                      'counts_by_task_type': {k: s['counts_by_task_type'] for k, s in result['sets'].items()},
                      'name_collision_exclusions': result['task_name_exclusions']['excluded_task_name_count'],
                      'name_collision_exclusions_by_source': result['task_name_exclusions']['excluded_task_name_counts_by_source'],
                      'shortfalls': {k: s['total_shortfall'] for k, s in result['sets'].items()}}))
    return result


if __name__ == '__main__':
    main()
