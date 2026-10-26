"""Interchange checks against BOTH unchanged dataset encoders, without models."""
from copy import deepcopy
import json
from pathlib import Path
import subprocess

import pytest

from repair.artifacts import checked_document, checkpoint_files
from repair.dry_fakes import FakeTokenizer, fake_review
from repair.export import require_approved, training_export, seed_references
from repair.probes import validate_training_checkpoint
from sed.storage import digest, file_hash, write_json
from test_repair_collect import run


def test_export_public_turns_approvals_and_training_tree_compatibility(tmp_path):
    data = run(tmp_path / 'collection', reviewer=fake_review)
    export = checked_document(tmp_path / 'collection/e2_training_collection.json')
    assert export['schema_version'] == 2 and export['format'] == 'repair_e2_training_collection'
    assert export['source_collection_sha256'] == file_hash(tmp_path / 'collection/collection.json')
    assert export['theta0'] == data['checkpoint'] and export['training_exposure'] == 0
    assert not export['ready_for_training'] and export['training_blockers'] == ['synthetic_or_canary']
    # Read only the existing training tree's encoder; no code is copied into this tree.
    source = subprocess.run(['git', 'show', '2506fa9:train/dataset.py'], check=True,
                            capture_output=True, text=True).stdout
    namespace = {'__name__': 'training_tree_dataset'}
    exec(compile(source, '2506fa9:train/dataset.py', 'exec'), namespace)
    from train.dataset import episode_examples
    for arm, side in [('N', 'near'), ('F', 'far')]:
        assert [e['task_id'] for e in export['arms'][arm]] == data['D_repair_train']
        for entry, pair in zip(export['arms'][arm], data['pairs']):
            require_approved(entry)
            episode = entry['episode']
            original = pair[side]['episode']
            assert set(episode) == {'schema_version', 'task_id', 'split', 'sealed', 'initial', 'steps'}
            assert set(episode['initial']) == {'visible'}
            for step, turn in zip(episode['steps'], original['history']):
                assert set(step) == {'event_id', 'action_raw', 'visible'}
                assert step['action_raw'] == turn['reply']
                assert set(step['visible']) == {'observation', 'admissible_commands'}
            for packing in ('turn', 'episode'):
                a = episode_examples(episode, FakeTokenizer(), packing=packing)
                b = namespace['episode_examples'](episode, FakeTokenizer(), packing=packing)
                assert a == b
                assert sum(r['supervised_tokens'] for r in a) > 0
            for ref in entry['seed_demonstrations']:
                assert ref['task_id'] == entry['task_id'] and ref['task_id'] not in data['D_repair_probe']
                path = Path(ref['evidence_root']) / ref['episode_dir'] / 'episode.json'
                assert file_hash(path) == ref['episode_sha256']
                assert digest(ref['receipt']) == ref['receipt_sha256']


def test_export_refuses_changed_review_or_labels_and_never_exports_audit(tmp_path):
    data = run(tmp_path / 'collection', reviewer=fake_review)
    export = checked_document(tmp_path / 'collection/e2_training_collection.json')
    entry = export['arms']['N'][0]
    for field, value in [('status', 'pending_human_review'), ('reviewed', False), ('approved', False)]:
        bad = deepcopy(entry); bad['human_review'][field] = value
        with pytest.raises(ValueError, match='Training must refuse'):
            require_approved(bad)
    bad = deepcopy(entry); bad['episode']['steps'][0]['action_raw'] = 'Action: look'
    with pytest.raises(ValueError): require_approved(bad)
    bad = deepcopy(data); bad['pairs'][0]['near']['review']['labels_supported'] = False
    with pytest.raises(ValueError, match='approved E2 pair'):
        training_export(bad, collection_sha256='fake')
    bad = deepcopy(data); bad['pairs'][0]['task_id'] = data['D_repair_probe'][0]
    with pytest.raises(ValueError): training_export(bad, collection_sha256='fake')
    assert 'internal_state' not in json.dumps([e['episode'] for arm in export['arms'].values() for e in arm])


def test_pending_export_is_empty_and_offline_approval_publishes_bound_export(tmp_path, monkeypatch):
    run(tmp_path / 'pending')
    pending = checked_document(tmp_path / 'pending/e2_training_collection.json')
    assert pending['arms'] == {'N': [], 'F': []} and not pending['ready_for_training']
    from tools.repair_collect import main
    import repair.runtime
    monkeypatch.setattr(repair.runtime, 'make_teachers', lambda *a, **k: pytest.fail('No teacher at finalize'))
    requests = json.loads((tmp_path / 'pending/review_requests.json').read_text())
    write_json(tmp_path / 'reviews.json', {k: fake_review(v) for k, v in requests.items()})
    main(['--branch', 'P0_K20', '--finalize', str(tmp_path / 'pending/collection.json'), '--reviews', str(tmp_path / 'reviews.json'),
          '--output', str(tmp_path / 'approved')])
    approved = checked_document(tmp_path / 'approved/e2_training_collection.json')
    assert len(approved['arms']['N']) == 4
    assert approved['source_collection_sha256'] == file_hash(tmp_path / 'approved/collection.json')
    assert (tmp_path / 'approved/run_manifest.json').is_file()


def test_seed_reference_hashes_point_to_verified_evidence(tmp_path):
    write_json(tmp_path / 'manifest.json', {'fixture': True})
    receipt = {'task_id': 'train/a/trial', 'episode_dir': 'attempts/a/episode',
               'episode_hashes': {'episode.json': 'raw-file-hash'}}
    collection = {'verified': [receipt]}
    refs = seed_references(tmp_path, collection, tmp_path, [{'task_id': 'train/a/trial', 'episode_id': 'trusted-seed'}])
    ref, = refs['train/a/trial']
    assert ref['episode_sha256'] == 'raw-file-hash' and ref['receipt_sha256'] == digest(receipt)
    assert ref['verified_manifest_sha256'] == file_hash(tmp_path / 'manifest.json')
    with pytest.raises(ValueError): seed_references(tmp_path, collection, tmp_path, [])


def test_probe_training_receipt_is_bound_to_export_and_served_checkpoint(tmp_path):
    run(tmp_path / 'collection', reviewer=fake_review)
    export = checked_document(tmp_path / 'collection/e2_training_collection.json')
    # Pure artifact validation fixture, never a real training/serving invocation.
    export.update(dry_run=False, ready_for_training=True, training_blockers=[])
    source = tmp_path / 'collection/collection.json'
    path = tmp_path / 'export.json'; write_json(path, export)
    adapter_path = tmp_path / 'adapter'; adapter_path.mkdir()
    (adapter_path / 'adapter_config.json').write_text('{}')
    (adapter_path / 'adapter_model.safetensors').write_bytes(b'fake')
    hashes = checkpoint_files(adapter_path)
    adapter = {'path': str(adapter_path), 'adapter_config_sha256': hashes['adapter_config.json'],
               'adapter_weights_sha256': hashes['adapter_model.safetensors']}
    receipt = {'format': 'repair_e2_training_run', 'schema_version': 1, 'arm': 'N', 'dry_run': False,
        'theta0': export['theta0'], 'source_collection_sha256': file_hash(source),
        'training_collection_sha256': file_hash(path),
        'checkpoints': [{'update': 1, 'status': 'saved', 'path': '/training-host/old/path', 'sha256': hashes}]}
    validate_training_checkpoint(receipt, export, source, path, arm='N', checkpoint=1, adapter=adapter)
    for key, value in [('arm', 'F'), ('dry_run', True), ('training_collection_sha256', 'changed'),
                       ('theta0', {}), ('source_collection_sha256', 'changed')]:
        bad = {**receipt, key: value}
        with pytest.raises(ValueError):
            validate_training_checkpoint(bad, export, source, path, arm='N', checkpoint=1, adapter=adapter)
    (adapter_path / 'adapter_model.safetensors').write_bytes(b'changed')
    with pytest.raises(ValueError, match='artifact changed'):
        validate_training_checkpoint(receipt, export, source, path, arm='N', checkpoint=1, adapter=adapter)
