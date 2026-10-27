"""Metadata-only split checks; no environments, models, GPU or network."""
from copy import deepcopy
import json
from pathlib import Path

import pytest

from data.manifest import build_repair_splits as base
from sed.manifest import freeze_splits
from sed.storage import digest, file_hash, write_json
from tools.method_v1_splits import (NAME_EXCLUSION_SETS, TYPE_ORDER, build_splits, equal_quotas, exclusion_sources,
                                    main, validate_repair, verified_trajectories)


@pytest.fixture
def universe():
    rows = []
    for source in ('train', 'valid_seen', 'valid_unseen'):
        for i, kind in enumerate(base.KINDS):
            for group in range(85 + i * 3 if source == 'train' else 2):
                for trial in range(2):
                    gid = f'{source}/{kind}-{group}'
                    rows.append(dict(task_id=f'{gid}/trial_{trial}', group_id=gid, task_type=kind,
                                     source_split=source, eligible=True, game_sha256='fake'))
    manifest = {'tasks': rows}
    freeze = freeze_splits(manifest)
    ids = [t['task_id'] for t in rows if t['source_split'] == 'valid_seen']
    dev = dict(task_ids=ids, list_sha256=digest(ids), manifest_sha256=digest(manifest), split_freeze_sha256=digest(freeze))
    verified = {t['task_id']: dict(task_id=t['task_id'], group_id=t['group_id'], trajectory_id='won-' + t['task_id'])
                for t in rows if t['source_split'] == 'train' and t['task_id'].endswith('trial_1')}
    return manifest, freeze, dev, verified


def test_order_disjointness_exact_trial_reuse_and_quotas(universe):
    m, f, dev, verified = universe
    old = base.build_splits(m, f, [], dev)
    sources = [dict(task_ids=s['task_ids'], group_ids=s['group_ids']) for s in old['sets'].values()]
    # A sibling trial in any subset excludes the whole group.
    selected = next(t for t in m['tasks'] if t['group_id'] not in set(old['sets']['D_probe']['group_ids']))
    sources.append(dict(task_ids=[selected['task_id']], group_ids=[selected['group_id']]))
    result = build_splits(m, f, sources, dev, verified)
    assert result == build_splits(m, f, sources, dev, verified)
    assert result['sets']['U_dev']['task_groups'] == 60
    assert result['sets']['V2_pool']['task_groups'] == 200
    assert result['sets']['U_dev']['training_eligible'] is False
    assert result['sets']['U_dev']['purpose'] == 'diagnosis_only'
    assert result['sets']['V2_pool']['training_eligible'] is True
    assert result['assertions']['all_passed'] and not any(result['assertions']['overlap_counts'].values())
    assert not any(result['assertions']['task_name_overlap_counts'].values())
    for name, s in result['sets'].items():
        assert s['list_sha256'] == digest(s['task_ids'])
        assert len(set(s['group_ids'])) == len(s['task_ids'])
        assert s['counts_by_task_type'] == s['quotas']
        assert s['task_ids'] == sorted(s['task_ids'], key=lambda t: (digest([s['seed'], t]), t))
        assert not set(s['group_ids']) & set().union(*(set(x['group_ids']) for x in sources))
        assert all(t.startswith('train/') for t in s['task_ids'])
    pool = result['sets']['V2_pool']
    assert pool['verified_trajectories'] == [verified[t] for t in pool['task_ids']]
    assert result['selection_rule']['type_order'] == list(TYPE_ORDER) == sorted(base.KINDS)
    assert result['sets']['U_dev']['quotas'] == dict.fromkeys(TYPE_ORDER, 10)
    assert result['sets']['U_dev']['seed'] == 20261101
    assert pool['seed'] == 20261102
    assert list(pool['quotas'].values()) == [34, 34, 33, 33, 33, 33]
    assert all(s['total_shortfall'] == 0 and not any(s['shortfalls'].values())
               for s in result['sets'].values())
    assert not set(pool['group_ids']) & set(result['sets']['U_dev']['group_ids'])


@pytest.mark.parametrize('dev_trial_count', [None, 2])
def test_dev_task_names_exclude_all_train_trials_before_sampling(universe, dev_trial_count):
    m, f, dev, verified = universe
    # Only names in the configured dev manifest count, even when it is a subset.
    dev = dict(dev, task_ids=dev['task_ids'][:dev_trial_count])
    dev['list_sha256'] = digest(dev['task_ids'])
    names = {tid.split('/')[1] for tid in dev['task_ids']}
    train = [t for t in m['tasks'] if t['source_split'] == 'train']
    candidates = [t for t in train if t['group_id'].split('/')[1] not in names]
    removed = [t for t in train if t not in candidates]
    result = build_splits(m, f, [], dev, verified)
    audit = result['task_name_exclusions']
    assert audit['excluded_task_names'] == sorted(names)
    assert audit['excluded_task_name_count'] == len(names)
    assert audit['excluded_population'] == base.population_counts(removed)
    assert audit['excluded_population']['unique_instances'] == 2 * len(names)
    assert audit['excluded_task_name_counts_by_source'] == dict(
        dev140=len(names), D_probe=0, D_unseen_quick=0, U_mech=0)
    assert audit['dev140_content_sha256'] == digest(dev)
    assert result['eligible_after_exclusions'] == base.population_counts(candidates)
    expected_dev = base.select(candidates, equal_quotas(60), 20261101)
    dev_groups = {t['group_id'] for t in expected_dev}
    expected_pool = base.select([t for t in candidates if t['group_id'] not in dev_groups
                                and t['task_id'] in verified], equal_quotas(200), 20261102)
    for name, expected in (('U_dev', expected_dev), ('V2_pool', expected_pool)):
        listing = result['sets'][name]
        assert listing['task_ids'] == [t['task_id'] for t in expected]
        assert not names & {gid.split('/')[1] for gid in listing['group_ids']}
        assert listing['counts_by_task_type'] == equal_quotas(len(expected))
    assert not any(result['assertions']['task_name_overlap_counts'].values())


@pytest.mark.parametrize('category', NAME_EXCLUSION_SETS[1:])
@pytest.mark.parametrize('source_split', ['train', 'valid_seen'])
def test_repair_names_excluded_even_across_split_prefixes(tmp_path, universe, category, source_split):
    m, f, dev, verified = universe
    # Leave the valid_seen '-1' name outside dev so only this source can block it.
    ids = [tid for tid in dev['task_ids'] if '/'+ TYPE_ORDER[0] + '-1/' not in tid]
    dev = dict(dev, task_ids=ids, list_sha256=digest(ids))
    gid = f'{source_split}/{TYPE_ORDER[0]}-1'
    path = tmp_path / 'repair_listing.json'
    write_json(path, dict(group_ids=[gid]))
    source = base.read_source(path, category, m, f, selected_only=False)
    result = build_splits(m, f, [source], dev, verified)
    audit = result['task_name_exclusions']
    # Existing TRAIN group exclusions already remove these names; cross-split
    # names are additional exclusions, counted once despite two sibling trials.
    assert audit['excluded_task_name_counts_by_source'][category] == (source_split != 'train')
    assert audit['source_hashes'] == [{k: source[k] for k in ('category', 'path', 'sha256', 'extraction_rule')}]
    for name, listing in result['sets'].items():
        assert f'train/{TYPE_ORDER[0]}-1' not in listing['group_ids']
        assert result['assertions']['task_name_overlap_counts'][name + '_vs_' + category] == 0


def test_equal_quotas_and_empty_pool_fail_loudly(universe):
    assert list(equal_quotas(200).values()) == [34, 34, 33, 33, 33, 33]
    m, f, dev, verified = universe
    with pytest.raises(ValueError, match='V2_pool task-group quota shortfalls:') as error:
        build_splits(m, f, [], dev, {})
    assert json.loads(str(error.value).split(': ', 1)[1]) == equal_quotas(200)
    all_ids = [t['task_id'] for t in m['tasks'] if t['source_split'] == 'train']
    gs = sorted({str(Path(t).parent) for t in all_ids})
    with pytest.raises(ValueError, match='U_dev task-group quota shortfalls:') as error:
        build_splits(m, f, [dict(task_ids=all_ids, group_ids=gs)], dev, verified)
    assert json.loads(str(error.value).split(': ', 1)[1]) == dict.fromkeys(base.KINDS, 10)


@pytest.mark.parametrize('name', ['U_dev', 'V2_pool'])
def test_type_shortfall_never_redistributed_or_filled_by_siblings(universe, name):
    m, f, dev, verified = universe
    kind = TYPE_ORDER[0]
    dev_names = {tid.split('/')[1] for tid in dev['task_ids']}
    candidates = [t for t in m['tasks'] if t['source_split'] == 'train'
                  and t['group_id'].split('/')[1] not in dev_names]
    if name == 'U_dev':
        groups = sorted({t['group_id'] for t in candidates if t['task_type'] == kind})
        sources = [dict(task_ids=[], group_ids=groups[9:])]
    else:
        sources = []
        # Preserve 33 verified groups after U_dev, including both sibling trials.
        dev_groups = {t['group_id'] for t in base.select(candidates, equal_quotas(60), 20261101)}
        groups = sorted({t['group_id'] for t in candidates
                         if t['task_type'] == kind and t['group_id'] not in dev_groups})[:33]
        verified = {t['task_id']: dict(task_id=t['task_id'], group_id=t['group_id'], trajectory_id=t['task_id'])
                    for t in m['tasks'] if t['source_split'] == 'train'
                    and (t['task_type'] != kind or t['group_id'] in groups)}
    with pytest.raises(ValueError, match=name + ' task-group quota shortfalls:') as error:
        build_splits(m, f, sources, dev, verified)
    assert json.loads(str(error.value).split(': ', 1)[1]) == {kind: 1}


def test_invalid_frozen_inputs_and_verified_identity_rejected(universe):
    m, f, dev, verified = universe
    broken = deepcopy(f)
    broken['manifest_sha256'] = 'changed'
    with pytest.raises(ValueError):
        build_splits(m, broken, [], dev, verified)
    broken = deepcopy(verified)
    next(iter(broken.values()))['group_id'] = 'wrong'
    with pytest.raises(ValueError, match='identity'):
        build_splits(m, f, [], dev, broken)


def fixture_sources(tmp_path, universe):
    m, f, dev, _ = universe
    write_json(tmp_path / 'manifest.json', m)
    write_json(tmp_path / 'freeze.json', f)
    write_json(tmp_path / 'dev.json', dev)
    legacy = [t for t in m['tasks'] if t['source_split'] == 'train'
              and int(t['group_id'].rsplit('-', 1)[1]) < 50 and t['task_id'].endswith('trial_0')]
    write_json(tmp_path / 'legacy_k500.json', dict(task_order=[t['task_id'] for t in legacy]))
    legacy_source = base.read_source(tmp_path / 'legacy_k500.json', 'k500_chain_selection', m, f)
    old = base.build_splits(m, f, [legacy_source], dev)
    old.update(builder_sha256=file_hash(base.__file__), rebuild_arguments={})
    old['content_sha256'] = digest(old)
    write_json(tmp_path / 'repair.json', old)
    root = tmp_path / 'verified/seeds_full_train_r2'
    task = next(t for t in m['tasks'] if t['source_split'] == 'train')
    won_tasks = [t for t in m['tasks'] if t['source_split'] == 'train' and t['task_id'].endswith('trial_1')]
    selection = dict(task_order=[t['task_id'] for t in won_tasks], manifest_sha256=digest(m), split_freeze_sha256=digest(f))
    write_json(root / 'selection.json', selection)
    receipts = []
    for i, won_task in enumerate(won_tasks):
        episode = dict(task_id=won_task['task_id'], game_sha256='fake', success=True, complete=True,
                       status='ok', split='train', sealed=False, episode_id=f'verified-real-execution-{i}')
        directory = root / f'attempts/{i}/episode'
        write_json(directory / 'episode.json', episode)
        write_json(directory / 'events.jsonl', {})
        write_json(directory / 'provenance.json', {})
        receipts.append(dict(task_id=won_task['task_id'], group_id=won_task['group_id'], success=True, won=True,
            status='ok', rejection=None, source='real_environment_teacher_rollout',
            episode_dir=str(directory.relative_to(root)), episode_hashes={p.name: file_hash(p) for p in directory.iterdir()}))
    write_json(root / 'manifest.json', dict(status='complete', selection_sha256=digest(selection), verified=receipts))
    for k in (10, 20, 30, 100, 300, 500):
        rows = [task] if k <= 100 else legacy
        write_json(root / f'subsets/k{k}/selection.json', dict(task_order=[], replacements={'unused': rows}))
        write_json(root / f'subsets/k{k}/manifest.json', dict(verified=[], attempts=rows))
    for k in (5, 10, 20, 30, 100):
        for seed in range(3):
            # Unselected replacement candidates do not exclude the entire universe.
            write_json(tmp_path / f'seeds_k{k}_s{seed}.json',
                       dict(task_order=[won_tasks[seed + 1]['task_id']], replacements=won_tasks))
    for k in (20, 100):
        for seed in range(3):
            directory = root.parent / f'seeds_k{k}_s{seed}'
            for view in (directory, directory / 'subsets/k5'):
                write_json(view / 'selection.json', dict(task_order=[task['task_id']], replacements=won_tasks))
                write_json(view / 'manifest.json', dict(verified=[], attempts=[won_tasks[4]],
                                                       group_ids=[won_tasks[5]['group_id']]))
    return root, old


def cli_args(tmp_path, root):
    return ['--manifest', str(tmp_path / 'manifest.json'), '--freeze', str(tmp_path / 'freeze.json'),
            '--dev140', str(tmp_path / 'dev.json'), '--repair-splits', str(tmp_path / 'repair.json'),
            '--seed-dir', str(tmp_path), '--verified-root', str(root), '--output', str(tmp_path / 'result.json')]


def test_cli_exclusive_create_source_hashes_and_all_subset_fields(tmp_path, universe, capsys):
    root, old = fixture_sources(tmp_path, universe)
    args = cli_args(tmp_path, root)
    result = main(args)
    printed = json.loads(capsys.readouterr().out)
    assert result['builder_sha256'] == file_hash('tools/method_v1_splits.py')
    assert len(result['exclusion_sources']) == 52
    assert all(s['sha256'] == file_hash(s['path']) for s in result['exclusion_sources'])
    assert all(result['input_sha256'][s['path']] == s['sha256'] for s in result['exclusion_sources'])
    audit = result['task_name_exclusions']
    assert {s['category'] for s in audit['source_hashes']} == set(NAME_EXCLUSION_SETS)
    assert all(s['sha256'] == file_hash(s['path']) for s in audit['source_hashes'])
    dev_source = next(s for s in audit['source_hashes'] if s['category'] == 'dev140')
    assert dev_source['path'] == str(tmp_path / 'dev.json')
    assert printed['name_collision_exclusions'] == audit['excluded_task_name_count']
    assert printed['name_collision_exclusions_by_source'] == audit['excluded_task_name_counts_by_source']
    assert printed['counts_by_task_type'] == {n: s['counts_by_task_type'] for n, s in result['sets'].items()}
    assert not any(result['assertions']['task_name_overlap_counts'].values())
    assert all('k300' not in s['path'] and 'k500' not in s['path'] for s in result['exclusion_sources'])
    assert result['repair_validation_sources'][0]['path'] == str(tmp_path / 'legacy_k500.json')
    excluded = set().union(*(set(s['group_ids']) for s in result['exclusion_sources']))
    selected = set().union(*(set(s['group_ids']) for s in result['sets'].values()))
    assert not selected & excluded
    blocked_names = {gid.split('/')[1] for s in result['exclusion_sources']
                     if s['category'] in NAME_EXCLUSION_SETS for gid in s['group_ids']}
    assert not {gid.split('/')[1] for gid in selected} & blocked_names
    assert selected & set(old['E']['group_ids'])  # K300/K500-only groups remain usable.
    assert 'train/' + TYPE_ORDER[0] + '-0' in excluded  # Full-view replacement/attempt.
    assert all('train/' + TYPE_ORDER[0] + f'-{i}' in excluded for i in range(1, 6))
    assert all(s['task_groups'] == n for s, n in zip(result['sets'].values(), (60, 200)))
    assert result['content_sha256'] == digest({k: v for k, v in result.items() if k != 'content_sha256'})
    before = (tmp_path / 'result.json').read_bytes()
    with pytest.raises(FileExistsError):
        main(args)
    assert (tmp_path / 'result.json').read_bytes() == before
    m, f, dev, _ = universe
    validate_repair(old, m, f, dev)
    old['builder_sha256'] = 'changed'
    with pytest.raises(ValueError):
        validate_repair(old, m, f, dev)


def test_missing_optional_view_disclosed_and_required_sources_fail(tmp_path, universe):
    root, old = fixture_sources(tmp_path, universe)
    m, f, _, _ = universe
    path = tmp_path / 'seeds_k30_s2.json'
    path.unlink()
    sources, absent = exclusion_sources(old, tmp_path / 'repair.json', m, f,
        tmp_path / 'freeze.json', tmp_path / 'dev.json', tmp_path, root)
    assert [s['path'] for s in absent] == [str(path)]
    assert any(s['path'] == str(root / 'subsets/k30/selection.json') for s in sources)
    (root / 'subsets/k100/manifest.json').unlink()
    with pytest.raises(FileNotFoundError):
        main(cli_args(tmp_path, root))
    assert not (tmp_path / 'result.json').exists()


def test_cli_shortfall_does_not_write_manifest(tmp_path, universe):
    root, _ = fixture_sources(tmp_path, universe)
    path = root / 'manifest.json'
    document = json.loads(path.read_text())
    document['verified'] = []
    write_json(path, document)
    with pytest.raises(ValueError, match='V2_pool task-group quota shortfalls:'):
        main(cli_args(tmp_path, root))
    assert not (tmp_path / 'result.json').exists()


def test_verified_won_requires_hash_bound_success_and_complete_episode(tmp_path, universe):
    root, _ = fixture_sources(tmp_path, universe)
    m, f, _, _ = universe
    result, _ = verified_trajectories(root, m, f)
    assert next(iter(result.values()))['trajectory_id'] == 'verified-real-execution-0'
    p = root / 'manifest.json'
    value = json.loads(p.read_text())
    value['verified'][0]['won'] = False
    write_json(p, value)
    with pytest.raises(ValueError, match='receipt'):
        verified_trajectories(root, m, f)
    value['verified'][0]['won'] = True
    write_json(p, value)
    write_json(root / 'attempts/0/episode/episode.json', {'success': False})
    with pytest.raises(ValueError, match='evidence'):
        verified_trajectories(root, m, f)
