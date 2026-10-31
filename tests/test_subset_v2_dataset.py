"""Turn-matching contract with builder fixtures and the actual sibling E0.5 reader."""
from collections import Counter
from copy import deepcopy
import json
from pathlib import Path
import random
import shutil
import subprocess
import sys

import pytest

from repair.artifacts import seal
from repair.supervision import InvalidExample, seal_dataset
from sed.storage import canonical_bytes, digest, file_hash, write_json
from tools import build_v2_dataset as builder
from tools import subset_v2_dataset as subset
from test_supervision import TASK, cli_args, dataset_inputs


def read_manifest(root):
    return json.loads((root / 'manifest.json').read_text())


def read_rows(root):
    return [json.loads(line) for line in (root / 'examples.jsonl').read_bytes().splitlines()]


def arguments(source, match, output, seed=0):
    return ['--source', str(source), '--match', str(match), '--seed', str(seed), '--output', str(output)]


@pytest.fixture
def datasets(dataset_inputs, tmp_path, capsys):
    """Expand only synthetic frozen train membership before either build."""
    d = dataset_inputs
    tasks = [TASK, TASK.replace('-1/', '-B/'), TASK.replace('-1/', '-C/')]
    splits = json.loads(d['splits'].read_text())
    splits.pop('content_sha256')
    splits['sets']['V2_pool']['task_ids'] = tasks
    write_json(d['splits'], seal(splits))
    (d['run'] / 'records/test.json').unlink()
    for task_index, (task, count) in enumerate(zip(tasks, (4, 3, 3))):
        for index in range(count):
            row = deepcopy(d['row'])
            row['task_id'] = row['execution']['task_id'] = task
            row['run_key'] = f'{task_index}-{index}'
            if task_index or index:
                for step in row['steps']:
                    step['executor'] = 'student'
            write_json(d['run'] / 'records' / (row['run_key'] + '.json'), row)
    # A separate run gives 7 prefix-expert turns without mutating source inputs.
    prefix_run = tmp_path / 'prefix-run'
    shutil.copytree(d['run'], prefix_run)
    for path in (prefix_run / 'records').glob('*.json'):
        path.unlink()
    for index in range(4):
        row = deepcopy(d['row'])
        row.update(mode='takeover', run_key=f'prefix-{index}', branch_index=int(index == 3))
        write_json(prefix_run / 'records' / f'{index}.json', row)
    source, match = tmp_path / 'JOINT', tmp_path / 'PREFIX_EXPERT'
    assert builder.main(cli_args(d, source)) == 0
    assert builder.main(cli_args({**d, 'run': prefix_run}, match, 'PREFIX_EXPERT')) == 0
    assert read_manifest(source)['supervised_turns'] == 20
    assert read_manifest(match)['supervised_turns'] == 7
    capsys.readouterr()
    return source, match, d


def test_exact_stratified_bytes_accounting_and_determinism(datasets, tmp_path, capsys):
    source, match, _ = datasets
    sm, mm = read_manifest(source), read_manifest(match)
    source_lines = {json.loads(line)['example_id']: line
                    for line in (source / 'examples.jsonl').read_bytes().splitlines(keepends=True)}
    assert all(len(row['supervised_assistant_indices']) == 1 for row in read_rows(source))
    outputs = [tmp_path / name for name in ('seed-0-a', 'seed-0-b', 'seed-1')]
    for output, seed in zip(outputs, (0, 0, 1)):
        assert subset.main(arguments(source, match, output, seed)) == 0
        summary = json.loads(capsys.readouterr().out)
        assert summary == dict(source=str(source), match=str(match), source_turns=20,
            match_turns=7, target_turns=7, selected_examples=7, selected_turns=7, tasks=3, output=str(output))
        manifest, rows = read_manifest(output), read_rows(output)
        tasks = sorted({e['task_id'] for e in read_rows(source)})
        quotas = dict(zip(tasks, ([8, 3], [6, 2], [6, 2])))
        assert manifest['subset'] == dict(source=str(source), source_manifest_sha256=file_hash(source / 'manifest.json'),
            source_example_ids_sha256=sm['example_ids_sha256'], source_supervised_turns=20,
            match=str(match), match_manifest_sha256=file_hash(match / 'manifest.json'), target_turns=7,
            seed=seed, stratification='task_id largest-remainder', per_task=quotas)
        assert sum(q for _, q in quotas.values()) == mm['supervised_turns']
        assert Counter(e['task_id'] for e in rows) == {task: q for task, (_, q) in quotas.items()}
        # Check the specified per-task RNG and sorted-before-shuffle order too.
        expected = []
        for task, (_, q) in quotas.items():
            ids = sorted(e['example_id'] for e in read_rows(source) if e['task_id'] == task)
            random.Random(f'{seed}:{task}').shuffle(ids)
            expected.extend(ids[:q])
        assert [e['example_id'] for e in rows] == sorted(expected)
        assert (output / 'examples.jsonl').read_bytes() == b''.join(source_lines[i] for i in sorted(expected))
        for path in (source / 'replay').rglob('*'):
            if path.is_file():
                assert (output / path.relative_to(source)).read_bytes() == path.read_bytes()
        for key in sm.keys() - subset.RECOMPUTED:
            assert manifest[key] == sm[key]
        assert manifest['dry_run'] and not manifest['training_eligible']
        assert manifest['supervised_turns'] == 7
        assert manifest['supervised_tokens'] == sum((len(e['messages'][i]['content']) + 3) // 4
            for e in rows for i in e['supervised_assistant_indices'])
        assert manifest['supervised_token_estimate']['value'] == manifest['supervised_tokens']
        assert manifest['training_exposure_tasks'] == sorted({e['group_id'] for e in rows})
        assert manifest['example_ids_sha256'] == digest(sorted(expected))
        assert not (output / 'labels').exists() and not (output / 'labels.jsonl').exists()
        subset.verify_dataset(output)
    assert (outputs[0] / 'examples.jsonl').read_bytes() == (outputs[1] / 'examples.jsonl').read_bytes()
    assert (outputs[0] / 'examples.jsonl').read_bytes() != (outputs[2] / 'examples.jsonl').read_bytes()


def test_actual_e05_reader_accepts_subset_and_copied_replay(datasets, tmp_path):
    source, match, d = datasets
    output = tmp_path / 'reader-subset'
    subset.main(arguments(source, match, output))
    sibling = Path(__file__).resolve().parents[2] / 'ICML27_wt_repair_train'
    assert (sibling / 'repair/v2_train.py').is_file(), 'Actual E0.5 sibling reader is required for this test'
    # Isolate sibling imports so its train.dataset cannot replace this tree's
    # renderer in later builder tests. These are the actual checks, not copies.
    script = '''
import json
from pathlib import Path
import socket
import sys
def forbidden(*args, **kwargs):
    raise AssertionError("Network/API access is forbidden in tests")
for name in ("connect", "connect_ex", "sendto"):
    setattr(socket.socket, name, forbidden)
socket.create_connection = socket.getaddrinfo = forbidden
sibling, output, theta0 = map(Path, sys.argv[1:])
sys.path[:0] = [str(sibling), str(sibling / "src")]
from repair import v2_train as reader
from repair.v2_inputs import InputFiles, Relocation
assert Path(reader.__file__).resolve() == sibling / "repair/v2_train.py"
files = InputFiles(Relocation(output))
counts = []
for root in (output, output / "replay"):
    manifest = json.loads((root / "manifest.json").read_text())
    sha = (root / "examples.sha256").read_text().strip()
    files.add(root / "examples.jsonl", sha)
    for name, expected in manifest["inputs"].items():
        files.reference(name, expected, root=root)
    files.verify()
    loaded = reader._read_bundle(dict(root=root, manifest=manifest, examples_sha256=sha), theta0, files)
    assert loaded["rows"] == [json.loads(line) for line in (root / "examples.jsonl").read_text().splitlines()]
    counts.append(len(loaded["rows"]))
print(json.dumps(counts))
'''
    result = subprocess.run([sys.executable, '-c', script, str(sibling), str(output), str(d['train'] / 'checkpoint')],
                            capture_output=True, text=True, check=True)
    assert json.loads(result.stdout) == [7, 20]


def reseal_copy(source, destination, *, updates=None, rows=None):
    """Modify synthetic flags/rows using the same seal helper, including replay."""
    verified = subset.verify_dataset(source)
    shutil.copytree(source, destination)
    for root in (destination / 'replay', destination):
        m = read_manifest(root)
        examples = read_rows(root) if root.name == 'replay' or rows is None else rows
        m.update(updates or {})
        if root == destination:
            m['replay'] = {**m['replay'], 'manifest_sha256': file_hash(destination / 'replay/manifest.json')}
        metadata = subset._metadata(m, verified['tokenizer'], verified['tokenizer_info'])
        for name in ('manifest.json', 'examples.jsonl', 'examples.sha256'):
            (root / name).unlink()
        seal_dataset(root, examples, policy=verified['policy'], **metadata)
    return destination


@pytest.mark.parametrize('case', ['existing', 'equal', 'larger', 'real-ineligible', 'mixed'])
def test_required_refusals(datasets, tmp_path, case):
    source, match, _ = datasets
    output = tmp_path / 'refused'
    reason = None
    if case == 'existing':
        output.mkdir()
        (output / 'keep.txt').write_text('untouched')
        error = FileExistsError
    else:
        error = InvalidExample
        if case == 'equal':
            match = source
            reason = 'target_must_be_smaller'
        elif case == 'larger':
            source, match = match, source
            reason = 'target_must_be_smaller'
        else:
            source = reseal_copy(source, tmp_path / 'flagged',
                updates=dict(dry_run=False, training_eligible=case == 'mixed'))
            reason = 'real_source_with_dry_run_match' if case == 'mixed' else 'real_source_not_training_eligible'
    with pytest.raises(error, match=reason):
        subset.main(arguments(source, match, output))
    assert not (output / 'manifest.json').exists()
    if case == 'existing':
        assert list(output.iterdir()) == [output / 'keep.txt']
        assert (output / 'keep.txt').read_text() == 'untouched'
    else:
        assert not output.exists()


@pytest.mark.parametrize('which', ['source', 'match'])
@pytest.mark.parametrize('damage', ['examples', 'ids', 'counts', 'turns', 'replay-examples', 'replay-manifest', 'schema'])
def test_both_inputs_verified_before_output(datasets, tmp_path, which, damage):
    source, match, _ = datasets
    damaged = tmp_path / 'damaged'
    shutil.copytree(source if which == 'source' else match, damaged)
    if damage in ('examples', 'replay-examples'):
        path = damaged / ('replay/examples.jsonl' if damage.startswith('replay') else 'examples.jsonl')
        path.write_bytes(path.read_bytes() + b'\n')
    else:
        path = damaged / ('replay/manifest.json' if damage == 'replay-manifest' else 'manifest.json')
        manifest = json.loads(path.read_text())
        if damage == 'counts':
            manifest['counts']['examples'] += 1
        elif damage == 'turns':
            manifest['supervised_turns'] += 1
        elif damage == 'ids':
            manifest['example_ids_sha256'] = '0' * 64
        elif damage == 'schema':
            manifest['schema'] = 'unsealed/1'
        else:
            manifest['created'] = 'tampered'
        write_json(path, manifest)
    output = tmp_path / 'never-created'
    with pytest.raises(InvalidExample):
        subset.main(arguments(damaged if which == 'source' else source,
                              damaged if which == 'match' else match, output))
    assert not output.exists()


def test_largest_remainder_tie_and_zero_quota(datasets, tmp_path):
    source, prefix, _ = datasets
    match = reseal_copy(prefix, tmp_path / 'two-turns', rows=read_rows(prefix)[:2])
    output = tmp_path / 'tie'
    subset.main(arguments(source, match, output))
    manifest = read_manifest(output)
    quotas = manifest['subset']['per_task']
    assert list(quotas.values()) == [[8, 1], [6, 1], [6, 0]]
    assert manifest['counts']['unique_instances'] == manifest['counts']['task_groups'] == 2
    assert len(manifest['training_exposure_tasks']) == 2


def test_multiturn_inexact_quota_reports_and_refuses(datasets, tmp_path, capsys):
    _, match, d = datasets
    # PREFIX_EXPERT has seven one-turn rows; convert its replay into a CONTINUE
    # dataset with the builder, whose 20 two-turn trajectories each get quota 0/1.
    source = tmp_path / 'CONTINUE'
    assert builder.main(cli_args(d, source, 'CONTINUE')) == 0
    capsys.readouterr()
    output = tmp_path / 'inexact'
    with pytest.raises(InvalidExample, match='inexact_task_quotas'):
        subset.main(arguments(source, match, output))
    report = json.loads(capsys.readouterr().out)
    assert report['status'] == 'refused' and report['target_turns'] == 7
    assert not report['all_examples_one_turn']
    assert len(report['per_task']) == 7
    assert all(v == dict(source_turns=2, quota=1, selected_turns=0, shortfall=1)
               for v in report['per_task'].values())
    assert not output.exists()


def test_multiturn_exact_and_diagnostics_restricted(dataset_inputs, tmp_path):
    d = dataset_inputs
    offline = tmp_path / 'OFFLINE'
    assert builder.main(cli_args(d, offline, 'OFFLINE')) == 0
    # Two complete trajectories of the same frozen task, from synthetic repeats.
    row = read_rows(offline)[0]
    second = deepcopy(row)
    second['evidence']['episode_ref'] += '.repeat'
    second['example_id'] = digest([second['arm'], second['source'], second['task_id'],
                                   second['evidence']['episode_ref'], None])
    repeated = reseal_copy(offline, tmp_path / 'repeated', rows=[row, second])
    output = tmp_path / 'exact-trajectories'
    subset.main(arguments(repeated, offline, output))
    manifest = read_manifest(output)
    assert manifest['supervised_turns'] == 2 and manifest['counts']['examples'] == 1
    assert set(manifest['trajectory_diagnostics']) == {read_rows(output)[0]['example_id']}


def test_failure_during_sealing_leaves_no_root_manifest(datasets, tmp_path, monkeypatch):
    source, match, _ = datasets
    def fail(*args, **kwargs):
        raise OSError('synthetic copy failure')
    monkeypatch.setattr(subset.shutil, 'copytree', fail)
    output = tmp_path / 'unsealed'
    with pytest.raises(OSError, match='synthetic copy failure'):
        subset.main(arguments(source, match, output))
    assert output.is_dir() and not list(output.iterdir())
