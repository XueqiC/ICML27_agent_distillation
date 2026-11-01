"""Reuse sealed, fabricated L evidence without any teacher or network access."""
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
import json
from types import SimpleNamespace

import pytest

from sed.storage import canonical_bytes, digest, file_hash, write_json
from tools import build_v2_dataset as builder
from test_build_v2_label_workers import records
from test_supervision import cli_args, dataset_inputs, label


def read_rows(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def read_manifest(directory):
    return json.loads((directory / 'manifest.json').read_text())


def forbid_teacher(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError('Reuse must never construct a labeler, teacher client or budget')
    for target in ('tools.build_v2_dataset.RationaleLabeler', 'repair.supervision.RationaleLabeler',
                   'tools.build_v2_dataset.SharedTeacherBudget', 'repair.takeover.SharedTeacherBudget',
                   'repair.supervision.FakeLabelClient', 'teachers.capped.from_config',
                   'teachers.openai_teacher.Teacher.__init__', 'teachers.openai_teacher.Teacher._client'):
        monkeypatch.setattr(target, forbidden)


def reuse_args(d, output, source, **kwargs):
    return cli_args(d, output, **kwargs) + ['--reuse-labels-from', str(source)]


@pytest.fixture
def sealed_source(dataset_inputs, tmp_path):
    source = tmp_path / 'source'
    assert builder.main(cli_args(dataset_inputs, source)) == 0
    return source


def rewrite_requests(source, rows):
    """Refreeze fabricated test evidence; never modifies a real dataset."""
    path = next((source / 'labels').glob('*/requests.jsonl'))
    path.write_bytes(b''.join(canonical_bytes(row) + b'\n' for row in rows))
    manifest = read_manifest(source)
    manifest['label_artifact_sha256'][str(path.relative_to(source))] = file_hash(path)
    write_json(source / 'manifest.json', manifest)
    return path


@pytest.mark.parametrize('workers', [1, 4])
def test_joint_complete_reuse_matches_normal_build(dataset_inputs, tmp_path, monkeypatch, workers):
    d = dataset_inputs
    # Two sequential labels per record exercise rewritten public prefixes.
    d['row']['steps'][1]['executor'] = 'teacher'
    paths = records(d, 6)
    failed = json.loads(paths[-1].read_text())
    failed['success'] = failed['execution']['success'] = False
    failed['execution']['termination_reason'] = 'STEP_LIMIT'
    write_json(paths[-1], failed)
    source, normal, reused = (tmp_path / name for name in ('joint', 'complete', 'reused'))
    assert builder.main(cli_args(d, source)) == 0
    assert builder.main(cli_args(d, normal, 'JOINT_COMPLETE_ONLY')) == 0
    for directory in (source, normal):
        manifest = read_manifest(directory)
        assert 'label_reuse' not in manifest
        assert 'label_reuse_from' not in manifest['effective_config']
    assert read_rows(source / 'labels.jsonl') == read_rows(normal / 'labels.jsonl')
    forbid_teacher(monkeypatch)
    assert builder.main(reuse_args(d, reused, source, arm='JOINT_COMPLETE_ONLY') +
                        ['--label-workers', str(workers)]) == 0
    for relative in ('examples.jsonl', 'examples.sha256', 'replay/examples.jsonl', 'replay/examples.sha256'):
        assert (reused / relative).read_bytes() == (normal / relative).read_bytes()
    assert sorted(map(canonical_bytes, read_rows(reused / 'labels.jsonl'))) == sorted(
        map(canonical_bytes, read_rows(normal / 'labels.jsonl')))
    for directory in (reused, reused / 'replay'):
        manifest = read_manifest(directory)
        assert manifest['label_reuse'] == dict(source_dataset=str(source.resolve()),
            source_manifest_sha256=file_hash(source / 'manifest.json'),
            source_examples_sha256=file_hash(source / 'examples.jsonl'), reused_calls=12, unused_source_calls=0)
        assert manifest['inputs'][str(source / 'manifest.json')] == file_hash(source / 'manifest.json')
        assert manifest['new_teacher_calls'] == manifest['synthetic_label_calls'] == 0
        assert manifest['teacher_attempts'] == [] and manifest['global_teacher_budget'] is None
        assert manifest['effective_config']['rationale_teacher'] is False
        assert manifest['effective_config']['label_reuse_from'] == str(source.resolve())
        assert manifest['effective_config']['label_workers'] == workers
        assert manifest['dry_run'] and not manifest['training_eligible']
        for relative, expected in manifest['label_artifact_sha256'].items():
            assert file_hash(reused / relative) == expected
    receipts = list((reused / 'labels').glob('*/requests.jsonl'))
    assert len(receipts) == 6
    for path in receipts:
        assert path.parent.name in {digest(str(ref)) for ref in paths}
        rows = read_rows(path)
        assert len(rows) == 2
        for row in rows:
            assert set(row) == {'timestamp', 'request', 'status', 'raw_output', 'source'}
            assert row['status'] == 'reused' and row['timestamp']
            assert row['source']['dataset'] == str(source.resolve())
            assert row['source']['requests_file'] == str(path.relative_to(reused))
            original = read_rows(source / row['source']['requests_file'])[row['source']['line'] - 1]
            assert original['status'] == 'completed'
            assert row['request'] == original['request'] and row['raw_output'] == original['raw_output']
    assert not list(reused.rglob('usage.jsonl'))
    assert not (reused / 'synthetic_budget.jsonl').exists()


@pytest.mark.parametrize('change', ['edit', 'delete', 'unlisted'])
def test_changed_source_requests_refused_before_output(dataset_inputs, sealed_source, tmp_path, monkeypatch, change):
    path = next((sealed_source / 'labels').glob('*/requests.jsonl'))
    if change == 'edit':
        path.write_bytes(path.read_bytes() + b'{}\n')
    elif change == 'delete':
        path.unlink()
    else:
        extra = sealed_source / 'labels/unlisted/requests.jsonl'
        extra.parent.mkdir()
        extra.write_bytes(path.read_bytes())
    forbid_teacher(monkeypatch)
    output = tmp_path / 'refused'
    with pytest.raises(ValueError, match='label_artifact_sha256'):
        builder.main(reuse_args(dataset_inputs, output, sealed_source))
    assert not output.exists()


@pytest.mark.parametrize('field,value', [
    ('schema', 'foreign/1'), ('arm', 'OFFLINE'), ('splits_sha256', 'different-split'),
    ('theta0', {'adapter_weights_sha256': 'different'}), ('teacher_alias', 'different'),
    ('dry_run', False), ('effective_config.teacher', {}),
    ('effective_config.label_token_cap', 65), ('effective_config.label_output_cap_including_reasoning', 2048),
    ('effective_config.rationale_template', 'different'),
] + [('code_sha256.' + name, 'different') for name in builder.LABEL_CODE_PATHS])
def test_source_identity_mismatch_refused(dataset_inputs, sealed_source, tmp_path, monkeypatch, field, value):
    manifest = read_manifest(sealed_source)
    keys = field.split('.', 1)
    if len(keys) == 1:
        manifest[field] = value
    else:
        manifest[keys[0]][keys[1]] = value
    write_json(sealed_source / 'manifest.json', manifest)
    forbid_teacher(monkeypatch)
    output = tmp_path / 'refused'
    with pytest.raises(ValueError, match='Label reuse source.*' + keys[0]):
        builder.main(reuse_args(dataset_inputs, output, sealed_source))
    assert not output.exists()


def test_unsealed_source_refused(dataset_inputs, sealed_source, tmp_path):
    (sealed_source / 'manifest.json').unlink()
    output = tmp_path / 'refused'
    with pytest.raises(ValueError, match='sealed source manifest'):
        builder.main(reuse_args(dataset_inputs, output, sealed_source))
    assert not output.exists()


@pytest.mark.parametrize('workers', [1, 4])
def test_missing_request_aborts_without_sealing(dataset_inputs, sealed_source, tmp_path, monkeypatch, workers):
    path = next((sealed_source / 'labels').glob('*/requests.jsonl'))
    rows = read_rows(path)
    for row in rows:
        if row['status'] == 'completed':
            row['request']['payload']['current_action'] = 'inventory'
    rewrite_requests(sealed_source, rows)
    forbid_teacher(monkeypatch)
    output = tmp_path / 'missing'
    with pytest.raises(ValueError, match='L halted: no remaining reused label'):
        builder.main(reuse_args(dataset_inputs, output, sealed_source) + ['--label-workers', str(workers)])
    assert output.is_dir()
    assert not list(output.rglob('manifest.json'))


@pytest.mark.parametrize('flag', [['--rationale-teacher'], ['--budget-ledger', 'unused.jsonl'],
                                  ['--max-teacher-input-tokens', '0'], ['--max-teacher-output-tokens', '0']])
def test_reuse_conflicts_with_teacher_flags(dataset_inputs, tmp_path, flag):
    output = tmp_path / 'refused'
    with pytest.raises(ValueError, match='mutually exclusive'):
        builder.main(reuse_args(dataset_inputs, output, tmp_path / 'source') + flag)
    assert not output.exists()


@pytest.mark.parametrize('arm', ['PREFIX_EXPERT', 'OFFLINE', 'CONTINUE'])
def test_reuse_requires_joint_arm(dataset_inputs, tmp_path, arm):
    output = tmp_path / 'refused'
    with pytest.raises(ValueError, match='requires --arm JOINT or JOINT_COMPLETE_ONLY'):
        builder.main(reuse_args(dataset_inputs, output, tmp_path / 'source', arm=arm))
    assert not output.exists()


@pytest.mark.parametrize('raw,reason', [('not json', 'invalid_json'),
    (label('x' * 65), 'reasoning_token_cap'), (label('apple 99 is here.'), 'label_grounding_failed')])
def test_reused_output_still_passes_normal_label_checks(dataset_inputs, sealed_source, tmp_path, monkeypatch, raw, reason):
    path = next((sealed_source / 'labels').glob('*/requests.jsonl'))
    rows = read_rows(path)
    for row in rows:
        if row['status'] == 'completed':
            row['raw_output'] = raw
    rewrite_requests(sealed_source, rows)
    forbid_teacher(monkeypatch)
    output = tmp_path / 'checked'
    assert builder.main(reuse_args(dataset_inputs, output, sealed_source)) == 0
    audit = read_rows(output / 'labels.jsonl')
    assert len(audit) == 1 and audit[0]['raw_output'] == raw
    assert audit[0]['label_check'] == reason and audit[0]['label_source'] == 'action_only_invalid_label'
    assert dict((r['reason'], r['count']) for r in read_manifest(output)['excluded'])['invalid_rationale_label'] == 1


def test_fifo_canonical_requests_exhaustion_and_concurrent_consumption(sealed_source, tmp_path):
    path = next((sealed_source / 'labels').glob('*/requests.jsonl'))
    completed = next(row for row in read_rows(path) if row['status'] == 'completed')
    rows = [dict(completed, status=status) for status in ('started', 'failed', 'reused')]
    rows += [dict(completed, raw_output=label(f'Public evidence {n}.')) for n in range(10)]
    rewrite_requests(sealed_source, rows)
    manifest = read_manifest(sealed_source)
    source = builder.LabelReuseSource(sealed_source, dry_run=True, splits_sha256=manifest['splits_sha256'],
        theta0=manifest['theta0'], teacher=manifest['effective_config']['teacher'])
    request = dict(reversed(list(completed['request'].items())))
    request['payload'] = dict(reversed(list(request['payload'].items())))
    prompt = SimpleNamespace(to_json=lambda: request)
    directory = tmp_path / 'fifo/labels' / path.parent.name
    with pytest.raises(ValueError, match='no remaining reused label'):
        builder.ReusedLabeler(source, directory.parent / 'different-record')(prompt)
    labeler = builder.ReusedLabeler(source, directory)
    assert labeler(prompt) == rows[3]['raw_output']
    assert labeler(prompt) == rows[4]['raw_output']
    assert source.summary()['unused_source_calls'] == 8
    # Separate per-record labelers still share one consumption lock/index.
    with ThreadPoolExecutor(max_workers=4) as pool:
        outputs = list(pool.map(lambda _: builder.ReusedLabeler(source, directory)(prompt), range(8)))
    assert sorted(outputs) == sorted(row['raw_output'] for row in rows[5:])
    assert [row['raw_output'] for row in read_rows(directory / 'requests.jsonl')] == [
        row['raw_output'] for row in rows[3:]]
    assert [row['source']['line'] for row in read_rows(directory / 'requests.jsonl')] == list(range(4, 14))
    assert source.summary()['reused_calls'] == 10 and source.summary()['unused_source_calls'] == 0
    with pytest.raises(ValueError, match='no remaining reused label'):
        labeler(prompt)


def test_manifest_counts_unused_completed_calls(dataset_inputs, sealed_source, tmp_path, monkeypatch):
    path = next((sealed_source / 'labels').glob('*/requests.jsonl'))
    rows = read_rows(path)
    completed = next(row for row in rows if row['status'] == 'completed')
    rows += [dict(completed, status='failed'), dict(completed, status='reused'),
             dict(completed, raw_output=label('A second sample.'))]
    rewrite_requests(sealed_source, rows)
    forbid_teacher(monkeypatch)
    output = tmp_path / 'unused'
    assert builder.main(reuse_args(dataset_inputs, output, sealed_source)) == 0
    manifest = read_manifest(output)
    assert manifest['label_reuse']['reused_calls'] == manifest['label_reuse']['unused_source_calls'] == 1
    assert read_rows(output / 'labels.jsonl')[0]['raw_output'] == completed['raw_output']


def test_real_build_refuses_dry_source(dataset_inputs, sealed_source, tmp_path, monkeypatch):
    forbid_teacher(monkeypatch)
    output = tmp_path / 'refused'
    args = reuse_args(dataset_inputs, output, sealed_source)
    args.remove('--dry-run')
    with pytest.raises(ValueError, match='source dry_run mismatch'):
        builder.main(args)
    assert not output.exists()


@pytest.mark.parametrize('eligible', [True, False])
def test_real_reuse_requires_training_eligible_source(dataset_inputs, sealed_source, tmp_path, monkeypatch, eligible):
    # Change only fabricated fixture flags to exercise the real-mode gate offline.
    manifest = read_manifest(sealed_source)
    manifest.update(dry_run=False, training_eligible=eligible, arm='JOINT_COMPLETE_ONLY')
    manifest['effective_config']['dry_run'] = False
    write_json(sealed_source / 'manifest.json', manifest)
    d = dataset_inputs
    identity = {'dry_run': False}
    write_json(d['run'] / 'plan.json', dict(identity=identity, run_sha256=digest(identity)))
    row = deepcopy(d['row'])
    row['run_sha256'] = digest(identity)
    write_json(d['run'] / 'records/test.json', row)
    forbid_teacher(monkeypatch)
    output = tmp_path / 'real-mode'
    args = reuse_args(d, output, sealed_source)
    args.remove('--dry-run')
    if not eligible:
        with pytest.raises(ValueError, match='training_eligible source'):
            builder.main(args)
        assert not output.exists()
    else:
        assert builder.main(args) == 0
        manifest = read_manifest(output)
        assert not manifest['dry_run'] and manifest['training_eligible']
        assert manifest['label_reuse']['reused_calls'] == 1
        assert manifest['teacher_attempts'] == [] and manifest['new_teacher_calls'] == 0
