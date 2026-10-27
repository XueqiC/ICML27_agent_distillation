"""Offline dp/1 selection, coverage provenance, and CG compatibility checks."""
from copy import deepcopy
import json
from pathlib import Path

import pytest

from repair.prefixes import (ATTEMPT_FIELDS, EVIDENCE_FILES, coverage_rows, coverage_stratum,
                             decision_point, load_original)
from sed.config import load_config
from sed.manifest import freeze_splits
from sed.storage import digest, file_hash, write_json
from tools import joint_guidance, teacher_takeover
from tools.cg_probe import GROUPS, load_cases
from tools.prefix_points import checked_set, main
from tools.repair_coverage import summarize_coverage
from test_cg_probe import original, coverage_fixture


INITIAL = 'Your task is to: put an apple on a table.'


def step(action='look', text='There is nothing here.', outcome='legal', **extra):
    return dict(action_parsed=action, action_outcome=outcome, observation=text, **extra)


@pytest.mark.parametrize('steps,label,event,branch', [
    ([step()] * 40, 'search_after_10', 9, 10),
    ([step()] * 4, 'search_after_10', 2, 3),
    ([step()], 'search_after_10', -1, 0),
    ([step(), step(text='On the table 1, you see an apple 1.'), step()], 'target_seen', 1, 2),
    ([step('take apple 1 from table 1', 'Nothing happens. Apple 1 remains here.'), step()], 'refused', 0, 1),
    # A public legal take with an uninformative result precedes the first sighting.
    ([step('take apple 1 from table 1', 'Taken.'), step(text='You are carrying: an apple 1.'), step()],
     'target_taken', 0, 1),
    # Seen wins a tie with taken, as specified by the rule's (i), (ii), (iii) order.
    ([step('take apple 1 from table 1', 'You pick up the apple 1.'), step()], 'target_seen', 0, 1),
    ([step(), step(text='Apple 1 is here.')], 'target_seen', 1, 1),
    ([step(text='Apple 1 is here.')], 'target_seen', 0, 0),
    ([step()] * 9 + [step(text='Apple 1 here.'), step()], 'target_seen', 9, 10),
    ([step()] * 10 + [step(text='Apple 1 here.')], 'target_seen', 10, 10),
    ([step()] * 12 + [step(text='Apple 1 here.'), step()], 'target_seen', 12, 13),
    ([step()] * 12 + [step('take apple 1 from table 1', 'Nothing happens. Apple 1 remains here.'), step()],
     'refused', 12, 13),
    ([step()] * 12 + [step('take apple 1 from table 1', 'Taken.'),
                      step(text='You are carrying: an apple 1.'), step()], 'target_taken', 12, 13),
])
def test_decision_points_labels_ties_and_cap(steps, label, event, branch):
    assert decision_point(INITIAL, steps) == dict(decision_point=label, decision_rule_version='dp/1',
        event_step=event, branch_index=branch, steps_remaining=40 - branch)


@pytest.mark.parametrize('prefix_length', [0, 12])
def test_refused_take_while_holding_and_earliest_event(prefix_length):
    steps = [step()] * prefix_length + [step('inventory', 'You are carrying: a mug 1.'),
             step('take book 1 from table 1', 'Nothing happens.'),
             step(text='On the table 1, you see an apple 1.'), step()]
    assert decision_point(INITIAL, steps)['event_step'] == prefix_length + 1
    assert decision_point(INITIAL, steps)['branch_index'] == prefix_length + 2
    assert decision_point(INITIAL, steps)['decision_point'] == 'refused'
    steps[prefix_length] = step(text='On the table 1, you see an apple 1.')
    assert decision_point(INITIAL, steps)['decision_point'] == 'target_seen'
    assert decision_point(INITIAL, steps)['event_step'] == prefix_length


def test_initial_seen_unrelated_refusal_and_search_precedence():
    assert decision_point(INITIAL + ' On the table 1, you see an apple 1.', [step()])['branch_index'] == 0
    unrelated = [step('take book 1 from table 1', 'Nothing happens.'), step(text='Apple 1 here.'), step()]
    assert decision_point(INITIAL, unrelated)['decision_point'] == 'target_seen'
    # A command containing a target is not evidence of observing that target.
    unseen = [step('take apple 1 from table 1', 'Nothing happens.'), *[step()] * 20]
    assert decision_point(INITIAL, unseen)['decision_point'] == 'search_after_10'


def test_decision_rule_public_whitelist():
    clean = [step()] * 12
    polluted = [dict(s, audit={'object': 'apple 1'}, policy_commands=['take apple 1 from table 1'],
                     reward=1, success=True, extra={'expert_plan': 'Apple 1 here.'}) for s in clean]
    initial = dict(visible=dict(observation=INITIAL, admissible_commands=['take apple 1 from table 1']),
                   hidden='Apple 1 here.')
    assert decision_point(initial, polluted) == decision_point(INITIAL, clean)
    with pytest.raises(ValueError, match='publicly parsed goal'):
        decision_point('Unknown task', clean)
    with pytest.raises(ValueError, match='at least one'):
        decision_point(INITIAL, [])


@pytest.mark.parametrize('k,n,label', [(0, 4, 'floor'), (1, 4, 'middle'), (3, 4, 'middle'),
                                       (4, 4, 'excluded'), (0, 1, 'floor'), (1, 1, 'excluded')])
def test_strata(k, n, label):
    assert coverage_stratum(k, n) == label


@pytest.mark.parametrize('k,n', [(0, 0), (2, 1), (-1, 4), (True, 4), (0, 1.5)])
def test_invalid_strata(k, n):
    with pytest.raises(ValueError):
        coverage_stratum(k, n)


def write_jsonl(path, rows):
    path.write_text(''.join(json.dumps(row) + '\n' for row in rows))


def write_indexes(root, rows):
    # Actual repair_coverage layout: hashes and outcome in rollouts; attempt
    # metadata only in episode_index. The failure list may truncate to 20 tasks.
    write_jsonl(root / 'rollouts.jsonl', [{k: v for k, v in row.items() if k not in ATTEMPT_FIELDS} for row in rows])
    write_jsonl(root / 'episode_index.jsonl', [{k: row[k] for k in
        ('episode_id', 'episode_dir', 'task_id', *ATTEMPT_FIELDS)} for row in rows])
    summary = summarize_coverage(rows, [], extra_sampled=2)
    summary['branch_decision_actionable'] = True
    write_json(root / 'summary.json', summary)
    write_json(root / 'failures.json', dict(failures=summary['failure_list'], list_sha256=summary['failure_list_sha256']))


def make_coverage(tmp_path, original, specs):
    root, rows = tmp_path / 'coverage', []
    for i, spec in enumerate(specs):
        group = spec.get('group', GROUPS[5])
        attempt, seed = spec.get('attempt', 1), spec.get('sampling_seed')
        decoding = spec.get('decoding', 'greedy')
        directory = root / f'attempt-{attempt:02d}-{decoding}' / 'episodes' / str(i)
        directory.mkdir(parents=True)
        ep = dict(task_id='train/' + group + '/trial_0', episode_id=f'episode-{i}', success=spec.get('success', False),
            split='train', sealed=False, complete=True, status='ok', seed=0, max_episode_steps=40,
            game_sha256=original.game_sha256, initial=deepcopy(original.initial),
            steps=deepcopy(original.steps), truncated=True)
        ep.update(spec.get('episode', {}))
        write_json(directory / 'episode.json', ep)
        metadata = dict(attempt=attempt, decoding=decoding, sampling_seed=seed)
        write_json(directory / 'provenance.json', dict(config=dict(r03_rollout=metadata)))
        (directory / 'events.jsonl').write_text('{}\n')
        rows.append(dict(task_id=ep['task_id'], task_type=group.split('-')[0], group_id='train/' + group,
            episode_id=ep['episode_id'], episode_dir=str(directory.relative_to(root)), **metadata,
            **{k: ep[k] for k in ('success', 'status', 'complete')},
            **{key: file_hash(directory / name) for name, key in EVIDENCE_FILES.items()}))
    write_indexes(root, rows)
    return root, rows


def entry_for(root, row, **extra):
    return dict(case_id=row['episode_id'], group=row['group_id'].removeprefix('train/'), task_id=row['task_id'],
        source=dict(failures_json=str(root / 'failures.json'), coverage_root=str(root), episode_dir=row['episode_dir']),
        branch_index=3, decision_point='target_seen', **extra)


def rewrite_episode(root, rows, index, change):
    row = rows[index]
    directory = root / row['episode_dir']
    episode = json.loads((directory / 'episode.json').read_text())
    change(episode)
    write_json(directory / 'episode.json', episode)
    row.update({key: file_hash(directory / name) for name, key in EVIDENCE_FILES.items()})
    row.update({k: episode[k] for k in ('success', 'status', 'complete')})
    write_indexes(root, rows)


def test_sampled_import_outside_failure_list_and_arbitrary_prefix(tmp_path, original):
    root, rows = make_coverage(tmp_path, original, [{}, dict(attempt=2, decoding='sampled', sampling_seed=0)])
    rewrite_episode(root, rows, 1, lambda ep: ep['steps'][0]['visible'].update(observation='Apple 1 here.'))
    entry = entry_for(root, rows[1], attempt=2, decoding='sampled', sampling_seed=0)
    loaded = load_original(entry, load_config('configs/base_e4b.yaml'), dry_run=True)
    assert loaded['original'].task_id == entry['task_id']
    assert loaded['source']['attempt'] == 2 and loaded['source']['sampling_seed'] == 0
    assert str(root / 'episode_index.jsonl') in loaded['source']['coverage_sha256']
    # Only [0, branch_index) is checked, so a later parser error is allowed.
    rewrite_episode(root, rows, 1, lambda ep: ep['steps'][3].update(action_raw='broken', action_outcome='parser_error'))
    load_original(entry, load_config('configs/base_e4b.yaml'), dry_run=True)
    entry['branch_index'] = 4
    with pytest.raises(ValueError, match='parser-error'):
        load_original(entry, load_config('configs/base_e4b.yaml'), dry_run=True)


@pytest.mark.parametrize('change,error', [
    (lambda ep: ep.update(success=True), 'must have failed'),
    (lambda ep: ep.update(seed=99), 'config differs'),
    (lambda ep: ep.update(max_episode_steps=39), 'config differs'),
    (lambda ep: ep.update(split='dev'), 'config differs'),
    (lambda ep: ep.update(sealed=True), 'config differs'),
    (lambda ep: ep.update(complete=False), 'config differs'),
    (lambda ep: ep.update(status='error'), 'config differs'),
    (lambda ep: ep['steps'][1].update(action_raw='broken'), 'parser-error'),
    (lambda ep: ep['steps'][1].update(action_parsed='fly'), 'parser-error'),
    (lambda ep: ep['steps'][1].update(terminated=True), 'terminal'),
    (lambda ep: ep['steps'][1].update(truncated=True), 'terminal'),
    (lambda ep: ep['initial'].update(terminated=True), 'terminal'),
    (lambda ep: ep['initial'].update(visible=None), 'public initial'),
    (lambda ep: ep['steps'][1].update(visible=None), 'parser-error'),
])
def test_general_loader_strict_episode_checks(tmp_path, original, change, error):
    root, rows = make_coverage(tmp_path, original, [{}])
    rewrite_episode(root, rows, 0, change)
    with pytest.raises(ValueError, match=error):
        load_original(entry_for(root, rows[0]), load_config('configs/base_e4b.yaml'), dry_run=True)


@pytest.mark.parametrize('name', ['episode.json', 'events.jsonl', 'provenance.json'])
def test_evidence_hash_mismatch(tmp_path, original, name):
    root, rows = make_coverage(tmp_path, original, [{}])
    path = root / rows[0]['episode_dir'] / name
    path.write_bytes(path.read_bytes() + b' ')
    with pytest.raises(ValueError, match='evidence changed'):
        load_original(entry_for(root, rows[0]), load_config('configs/base_e4b.yaml'), dry_run=True)


@pytest.mark.parametrize('change', ['failure_hash', 'summary_hash', 'not_actionable', 'index_mismatch', 'duplicate'])
def test_coverage_index_and_summary_checks(tmp_path, original, change):
    root, rows = make_coverage(tmp_path, original, [{}])
    path = root / ('failures.json' if change == 'failure_hash' else 'summary.json')
    doc = json.loads(path.read_text())
    if change == 'failure_hash':
        doc['list_sha256'] = 'changed'
    elif change == 'summary_hash':
        doc['failure_list_sha256'] = 'changed'
    elif change == 'not_actionable':
        doc['branch_decision_actionable'] = False
    elif change == 'index_mismatch':
        index = json.loads((root / 'episode_index.jsonl').read_text())
        index['task_id'] = 'train/other/trial_0'
        write_jsonl(root / 'episode_index.jsonl', [index])
    else:
        path2 = root / 'episode_index.jsonl'
        path2.write_text(path2.read_text() * 2)
    write_json(path, doc)
    with pytest.raises(ValueError):
        coverage_rows(root / 'failures.json', root)


def test_branch_bounds_metadata_path_and_legacy_failure_only_import(tmp_path, original):
    root, rows = make_coverage(tmp_path, original, [{}])
    config = load_config('configs/base_e4b.yaml')
    entry = entry_for(root, rows[0])
    for n in (-1, True, 40, 3.0):
        with pytest.raises(ValueError, match='executable continuation'):
            load_original(dict(entry, branch_index=n), config, dry_run=True)
    for key, value in [('attempt', 2), ('decoding', 'sampled'), ('sampling_seed', 0)]:
        with pytest.raises(ValueError, match='attempt differs'):
            load_original(dict(entry, **{key: value}), config, dry_run=True)
    rewrite_episode(root, rows, 0, lambda ep: ep.update(steps=ep['steps'][:4]))
    assert len(load_original(entry, config, dry_run=True)['original'].steps) == 4
    (root / 'rollouts.jsonl').unlink()
    (root / 'episode_index.jsonl').unlink()
    assert load_original(entry, config, dry_run=True)['source']['attempt'] == 1
    rows[0]['episode_dir'] = '../escape'
    write_indexes(root, rows)
    with pytest.raises(ValueError, match='escapes'):
        load_original(entry_for(root, rows[0]), config, dry_run=True)


def frozen_config(tmp_path, row, *, dev=False, game='synthetic_game'):
    task = dict(task_id=row['task_id'], group_id=row['group_id'], task_type=row['task_type'],
                source_split='train', game_sha256=game, eligible=True)
    manifest = dict(tasks=[task])
    freeze = freeze_splits(manifest, **dict(dev_source='train', dev_count=1) if dev else {})
    config = load_config('configs/base_e4b.yaml')
    for key, value in [('manifest', manifest), ('split_freeze', freeze)]:
        path = tmp_path / (key + '.json')
        write_json(path, value)
        config['paths'][key] = str(path)
    return config, (manifest, freeze)


def test_real_frozen_authorization_and_game_binding(tmp_path, original):
    root, rows = make_coverage(tmp_path, original, [{}])
    entry = entry_for(root, rows[0])
    config, _ = frozen_config(tmp_path, rows[0])
    assert load_original(entry, config)['task']['split'] == 'train'
    config, _ = frozen_config(tmp_path, rows[0], dev=True)
    with pytest.raises(ValueError, match='frozen train'):
        load_original(entry, config)
    config, _ = frozen_config(tmp_path, rows[0], game='different')
    with pytest.raises(ValueError, match='task/game'):
        load_original(entry, config)


def write_splits(path, ids, frozen=None):
    bound = {} if frozen is None else dict(manifest_sha256=digest(frozen[0]), split_freeze_sha256=digest(frozen[1]))
    listing = dict(task_ids=ids, group_ids=[str(Path(t).parent) for t in ids], list_sha256=digest(ids),
                   purpose='diagnosis_only', training_eligible=False, **bound)
    doc = dict(sets=dict(U_dev=listing), **bound)
    doc['content_sha256'] = digest(doc)
    write_json(path, doc)


def test_builder_full_index_first_failed_selection_strata_and_summary(tmp_path, original):
    specs = []
    groups = [f'pick_and_place_simple-Apple-None-Table-{i}' for i in range(23)]
    for i, group in enumerate(groups):
        for attempt, (decoding, seed) in enumerate([('greedy', None), ('sampled', 0), ('sampled', 1), ('sampled', 2)], 1):
            specs.append(dict(group=group, attempt=attempt, decoding=decoding, sampling_seed=seed,
                              success=i == 22 or (i == 21 and attempt in (1, 2))))
    # Reverse file order: selection must still obey the registered attempt order.
    root, rows = make_coverage(tmp_path, original, list(reversed(specs)))
    splits, output = tmp_path / 'splits.json', tmp_path / 'spec.json'
    ids = ['train/' + group + '/trial_0' for group in groups]
    write_splits(splits, ids)
    assert main(['build', '--coverage', str(root), '--splits', str(splits), '--set', 'U_dev',
                 '--dry-run', '--output', str(output)]) == 0
    spec = json.loads(output.read_text())
    assert len(spec) == 22  # Includes failures omitted by the capped failure list.
    assert all(e['stratum'] == 'floor' and e['attempt'] == 1 for e in spec[:21])
    assert spec[-1]['attempt'] == 3 and spec[-1]['sampling_seed'] == 1
    assert spec[-1]['coverage_successes'] == 2 and spec[-1]['coverage_rollouts'] == 4
    assert spec[-1]['stratum'] == 'middle'
    assert spec[-1]['case_id'] == groups[21] + '/dp/1'
    summary = json.loads(output.with_suffix('.summary.json').read_text())
    assert summary['unique_instances'] == summary['task_groups'] == 22
    assert sum(r['count'] for r in summary['counts']) == 22
    assert summary['excluded'][0]['reason'] == 'no_failed_rollout'
    assert output.with_suffix('.summary.md').is_file()
    cases, inputs = teacher_takeover.load_spec(output, load_config('configs/base_e4b.yaml'), dry_run=True)
    assert len(cases) == 22 and str(root / 'episode_index.jsonl') in inputs


def test_split_hash_and_frozen_bindings(tmp_path, original):
    root, rows = make_coverage(tmp_path, original, [{}])
    _, frozen = frozen_config(tmp_path, rows[0])
    splits = tmp_path / 'splits.json'
    write_splits(splits, [rows[0]['task_id']], frozen)
    assert checked_set(splits, 'U_dev', frozen) == [rows[0]['task_id']]
    doc = json.loads(splits.read_text())
    doc['manifest_sha256'] = 'different'
    doc['content_sha256'] = digest({k: v for k, v in doc.items() if k != 'content_sha256'})
    write_json(splits, doc)
    with pytest.raises(ValueError, match='frozen manifest'):
        checked_set(splits, 'U_dev', frozen)
    doc['content_sha256'] = 'different'
    write_json(splits, doc)
    with pytest.raises(ValueError, match='content hash'):
        checked_set(splits, 'U_dev', frozen)


def test_mixed_spec_and_non_cg_dispatch(tmp_path, original, monkeypatch):
    root, rows = make_coverage(tmp_path, original, [{}, dict(attempt=2, decoding='sampled', sampling_seed=0),
        dict(group='pick_and_place_simple-Apple-None-Table-999')])
    cg = entry_for(root, rows[0])
    del cg['decision_point']
    cg['branch_index'] = 10
    sampled = entry_for(root, rows[1])
    non_cg = entry_for(root, rows[2])
    del non_cg['decision_point']
    path = tmp_path / 'spec.json'
    write_json(path, [cg, sampled, non_cg])
    native = teacher_takeover.load_cases
    native_coverage = teacher_takeover.coverage_rows
    calls, coverage_calls = [], []
    def legacy(*args, **kwargs):
        calls.append(args[2])
        return native(*args, **kwargs)
    monkeypatch.setattr(teacher_takeover, 'load_cases', legacy)
    def indexed(*args):
        coverage_calls.append(args)
        return native_coverage(*args)
    monkeypatch.setattr(teacher_takeover, 'coverage_rows', indexed)
    for loader in (teacher_takeover.load_spec, joint_guidance.load_spec):
        cases, _ = loader(path, load_config('configs/base_e4b.yaml'), dry_run=True)
        assert [c['source']['episode_id'] for c in cases] == [r['episode_id'] for r in rows]
        assert all('instruction' not in c['task'] for c in cases)
    assert calls == [[GROUPS[5]], [GROUPS[5]]]
    assert len(coverage_calls) == 2  # Once per load_spec, not once per case.


@pytest.mark.parametrize('cli', [teacher_takeover, joint_guidance])
def test_cg_spec_keeps_legacy_run_sha_for_same_code(tmp_path, original, monkeypatch, cli):
    root, _ = coverage_fixture(tmp_path, original)
    path = tmp_path / 'spec.json'
    teacher_takeover.main(['spec-from-cg', '--failures', str(root / 'failures.json'), '--coverage-root', str(root),
                          '--groups', GROUPS[5], '--dry-run', '--output', str(path)])
    spec = json.loads(path.read_text())
    loaded = load_cases(root / 'failures.json', root, [GROUPS[5]], load_config('configs/base_e4b.yaml'), dry_run=True)[0]
    task = {k: v for k, v in loaded['task'].items() if k != 'instruction'}
    old_cases = [{'spec': spec[0], **loaded, 'task': task}]
    old_inputs = sorted({str(path.resolve()), str(root / 'failures.json'), str(root / 'summary.json'),
                         *loaded['source']['evidence_sha256']})
    cases, inputs = cli.load_spec(path, load_config('configs/base_e4b.yaml'), dry_run=True)
    assert cases == old_cases and inputs == old_inputs
    common = ['run', '--spec', str(path), '--dry-run', '--max-teacher-output-tokens', '0']
    if cli is teacher_takeover:
        common += ['--runs', '1']
    else:
        common += ['--modes', 'U', '--seeds', '--budget-ledger', str(tmp_path / 'budget.jsonl')]
    cli.main([*common, '--output', str(tmp_path / 'new')])
    monkeypatch.setattr(cli, 'load_spec', lambda *a, **kw: (old_cases, list(old_inputs)))
    cli.main([*common, '--output', str(tmp_path / 'legacy')])
    new = json.loads((tmp_path / 'new/plan.json').read_text())
    old = json.loads((tmp_path / 'legacy/plan.json').read_text())
    assert new['run_sha256'] == old['run_sha256']
