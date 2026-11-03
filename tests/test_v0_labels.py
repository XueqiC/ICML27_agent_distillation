"""Offline regression coverage for student token caps and raw-label reuse."""
from collections import Counter
from copy import deepcopy
import json
from pathlib import Path

import pytest

from repair.dry_fakes import FakeEnvironment, N
from repair.execute import execute_actions
from repair.schemas import SchemaError, parse_label
from repair.supervision import ByteBoundTokenizer
from repair.v0_common import completed, finish, phase, read
from sed.storage import canonical_bytes, file_hash, write_json
from tools import v0_round as cli
from test_v0_pipeline import arguments, fake_adapter, setup_synth
from test_v0_round import TASK


RATIONALE = ('The visible observations support this action because it follows the current task '
             'and uses information already available in the public history before taking this step.')
RAW = json.dumps(dict(reasoning=RATIONALE, evidence_steps=[-1], unsupported=False))


class GemmaLikeTokenizer:
    chat_template = 'fake local template'

    def encode(self, text, add_special_tokens=False):
        assert add_special_tokens is False
        return text.split()


@pytest.fixture
def runtime(tmp_path, monkeypatch):
    root = tmp_path / 'round'
    args = arguments(root)
    cli.round_root(args)
    manifest = setup_synth(root)
    phase(root / 'attempt', dict(command='attempt', inputs={}, complete_task_coverage=True,
                                 adapter=fake_adapter()))
    write_json(root / 'attempt/records.json', [])
    finish(root / 'attempt')
    execution = execute_actions(FakeEnvironment, TASK, N, 40).to_json()
    episodes = []
    for index, step in enumerate((3, 4)):
        origins = ['student'] * len(N)
        origins[step] = 'patch'
        episodes.append(dict(episode_id=f'repair-{index}', task_id=TASK, prefix_length=0,
            success=True, verified=True, execution=execution, executors=origins))
    phase(root / 'review', dict(command='review', inputs={}))
    write_json(root / 'review/records.json', episodes)
    finish(root / 'review')
    args.action_only = False
    calls = []
    def labeler(*unused):
        def generate(prompt):
            calls.append(prompt.to_json())
            return RAW
        return generate
    monkeypatch.setattr('repair.v0_review.make_rationale_labeler', labeler)
    monkeypatch.setattr('train.dataset.load_tokenizer', lambda path: GemmaLikeTokenizer())
    return args, root, manifest, calls


def use_student_tokens(args, tmp_path):
    revision = cli.config_for(args)['models']['student']['revision']
    snapshot = tmp_path / revision
    snapshot.mkdir(exist_ok=True)
    (snapshot / 'tokenizer.json').write_text('{"fake": true}')
    args.byte_token_cap, args.snapshot = False, str(snapshot)
    return snapshot


def label(runtime, directory=None):
    args, root, manifest, _ = runtime
    return cli.run_labels(args, root, manifest, root / 'attempt', root / 'review', directory=directory)


def test_student_token_cap_accepts_normal_rationale():
    assert len(RATIONALE.split()) == 25
    assert parse_label(RAW, GemmaLikeTokenizer()).reasoning == RATIONALE
    with pytest.raises(SchemaError, match='reasoning_token_cap'):
        parse_label(RAW, ByteBoundTokenizer())


@pytest.mark.parametrize('workers', [1, 3])
def test_all_matching_labels_revalidate_without_teacher_or_usage(runtime, tmp_path, monkeypatch, workers):
    args, root, _, calls = runtime
    old_rows = label(runtime)
    assert Counter(r['evidence']['label_check'] for r in old_rows if r['v0']['edit_decision']) == {
        'reasoning_token_cap': 2}
    assert len(calls) == 2
    old_files = cli.artifacts(root / 'label')
    budget_before = cli.budget_for(root).summary()
    snapshot = use_student_tokens(args, tmp_path)
    args.reuse_labels_from, args.label_workers = str(root / 'label'), workers
    def forbidden(*unused):
        pytest.fail('All matching requests must avoid constructing a teacher')
    monkeypatch.setattr('repair.v0_review.make_rationale_labeler', forbidden)
    target = root / 'relabeled'
    rows = label(runtime, target)
    assert [r['evidence']['label_source'] for r in rows if r['v0']['edit_decision']] == ['teacher_rationale'] * 2
    assert cli.budget_for(root).summary() == budget_before
    assert cli.artifacts(root / 'label') == old_files
    assert not list(target.glob('episodes/*/teacher'))
    receipts = [read(p) for p in target.glob('episodes/*/requests/*.json')]
    assert len(receipts) == 2 and all(r['reused'] is True for r in receipts)
    for receipt in receipts:
        assert file_hash(receipt['source_path']) == receipt['source_sha256']
        assert receipt['raw_output'] == RAW
    identity = read(target / 'plan.json')['identity']
    assert identity['tokenizer']['path'] == str(snapshot)
    assert identity['tokenizer']['files']['tokenizer.json'] == file_hash(snapshot / 'tokenizer.json')
    assert identity['inputs'][str(snapshot / 'tokenizer.json')] == file_hash(snapshot / 'tokenizer.json')
    assert completed(target)
    args.resume = True
    assert label(runtime, target) == rows


def test_partial_reuse_calls_teacher_only_for_missing_request(runtime, tmp_path):
    args, root, _, calls = runtime
    label(runtime)
    # Simulate an interrupted source containing just one completed label.
    (root / 'label/completion.json').unlink()
    missing = next((root / 'label/episodes/repair-1/requests').glob('*.json'))
    missing.unlink()
    calls.clear()
    use_student_tokens(args, tmp_path)
    args.reuse_labels_from = str(root / 'label')
    rows = label(runtime, root / 'partial')
    assert len(calls) == 1
    assert all(r['evidence']['label_check'] == 'passed' for r in rows)
    receipts = [read(p) for p in (root / 'partial').glob('episodes/*/requests/*.json')]
    assert sum(r.get('reused', False) for r in receipts) == 1


def test_corrected_prefix_requires_a_new_request_for_later_patch(runtime, tmp_path):
    args, root, _, calls = runtime
    (root / 'review/completion.json').unlink()
    episode = read(root / 'review/records.json')[0]
    episode['executors'][4] = 'patch'
    write_json(root / 'review/records.json', [episode])
    finish(root / 'review')
    label(runtime)
    assert len(calls) == 2
    old_second_request = calls[1]
    calls.clear()
    use_student_tokens(args, tmp_path)
    args.reuse_labels_from = str(root / 'label')
    label(runtime, root / 'new-prefix')
    assert len(calls) == 1 and calls[0] != old_second_request
    assert RATIONALE in calls[0]['payload']['prefix'][3]['reply']
    receipts = [read(p) for p in (root / 'new-prefix').glob('episodes/*/requests/*.json')]
    assert sum(r.get('reused', False) for r in receipts) == 1


@pytest.mark.parametrize('changed', ['manifest', 'reviews'])
def test_reuse_input_mismatch_aborts_before_output_or_teacher(runtime, tmp_path, changed):
    args, root, manifest, calls = runtime
    label(runtime)
    calls.clear()
    use_student_tokens(args, tmp_path)
    args.reuse_labels_from = str(root / 'label')
    if changed == 'manifest':
        value = read(manifest.path)
        value['tasks'][0]['task_text'] += ' Changed input.'
        write_json(manifest.path, value)
    else:
        (root / 'review/completion.json').unlink()
        value = read(root / 'review/records.json')
        value[0]['episode_id'] = 'different-review'
        write_json(root / 'review/records.json', value)
        finish(root / 'review')
    with pytest.raises(cli.LabelSetupError, match='inputs differ'):
        label(runtime, root / 'mismatch')
    assert calls == [] and not (root / 'mismatch').exists()


@pytest.mark.parametrize('explicit', [False, True])
def test_production_label_missing_tokenizer_aborts(runtime, tmp_path, monkeypatch, explicit):
    args, root, _, calls = runtime
    args.byte_token_cap = False
    args.snapshot = str(tmp_path / 'missing') if explicit else None
    monkeypatch.setattr(Path, 'home', lambda: tmp_path)
    with pytest.raises(cli.LabelSetupError, match='tokenizer.json'):
        label(runtime)
    assert calls == [] and not (root / 'label').exists()


def test_default_discovery_uses_v2_pinned_snapshot(runtime, tmp_path, monkeypatch):
    args, root, _, _ = runtime
    student = cli.config_for(args)['models']['student']
    snapshot = (tmp_path / '.cache/huggingface/hub' / ('models--' + student['requested_repo'].replace('/', '--'))
                / 'snapshots' / student['revision'])
    snapshot.mkdir(parents=True)
    (snapshot / 'tokenizer.json').write_text('{}')
    monkeypatch.setattr(Path, 'home', lambda: tmp_path)
    args.byte_token_cap = False
    label(runtime)
    assert read(root / 'label/plan.json')['identity']['tokenizer']['path'] == str(snapshot)


def test_reuse_transport_completion_and_exact_canonical_match(runtime, tmp_path):
    args, root, _, calls = runtime
    label(runtime)
    (root / 'label/completion.json').unlink()
    path = next((root / 'label/episodes/repair-0/requests').glob('*.json'))
    receipt = read(path)
    path.unlink()
    transport = path.parent.parent / 'teacher/requests.jsonl'
    transport.parent.mkdir()
    transport.write_text(json.dumps(dict(receipt, status='completed')) + '\n' +
                         json.dumps(dict(receipt, status='started', raw_output=None)) + '\n')
    inputs = read(root / 'label/plan.json')['identity']['label_source_inputs']
    source = cli.LabelReuseSource(root / 'label', inputs, file_hash(args.config))
    reused = source.find(dict(reversed(list(receipt['request'].items()))))
    assert reused['source_path'] == str(transport) and reused['source_line'] == 1
    assert reused['source_sha256'] == file_hash(transport)
    changed = deepcopy(receipt['request'])
    changed['payload']['current_action'] += ' '
    assert canonical_bytes(changed) != canonical_bytes(receipt['request'])
    assert source.find(changed) is None
    use_student_tokens(args, tmp_path)
    args.reuse_labels_from = str(root / 'label')
    calls.clear()
    label(runtime, root / 'transport-reuse')
    assert calls == []


def test_reuse_rejects_tampered_sealed_receipt(runtime):
    args, root, _, calls = runtime
    label(runtime)
    path = next((root / 'label').glob('episodes/*/requests/*.json'))
    write_json(path, dict(read(path), raw_output='changed'))
    args.reuse_labels_from = str(root / 'label')
    calls.clear()
    with pytest.raises(ValueError, match='artifacts changed'):
        label(runtime, root / 'tampered')
    assert calls == [] and not (root / 'tampered').exists()


def test_reuse_legacy_plan_allows_identical_config_in_another_worktree(runtime, tmp_path):
    args, root, _, calls = runtime
    label(runtime)
    (root / 'label/completion.json').unlink()
    path = root / 'label/plan.json'
    plan = read(path)
    for key in ('label_source_inputs', 'label_config_sha256', 'tokenizer', 'reuse_labels_from'):
        del plan['identity'][key]
    from sed.storage import digest
    plan['run_sha256'] = digest(plan['identity'])
    write_json(path, plan)
    finish(root / 'label')
    copied_config = tmp_path / 'copied-config.yaml'
    copied_config.write_bytes(Path(args.config).read_bytes())
    args.config = str(copied_config)
    use_student_tokens(args, tmp_path)
    args.reuse_labels_from = str(root / 'label')
    calls.clear()
    label(runtime, root / 'legacy')
    assert calls == []


def test_label_cli_exposes_tokenizer_and_reuse_flags(tmp_path, monkeypatch):
    seen = []
    monkeypatch.setattr(cli, 'round_root', lambda args: seen.append(args) or tmp_path)
    def stop(*unused):
        raise RuntimeError('parsed')
    monkeypatch.setattr(cli, 'manifest_for', stop)
    with pytest.raises(RuntimeError, match='parsed'):
        cli.main(['label', '--output', str(tmp_path), '--snapshot', '/local/snapshot',
                  '--reuse-labels-from', '/old/labels'])
    assert seen[0].snapshot == '/local/snapshot' and seen[0].reuse_labels_from == '/old/labels'
    assert seen[0].byte_token_cap is False
