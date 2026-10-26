"""Offline interchange fixtures: real receipt readers, fake environment/teacher.

The fixture deliberately uses real-mode flags to exercise acceptance gates;
none of these synthetic artifacts is an experiment or a training dataset.
"""
from copy import deepcopy
import json
from pathlib import Path
import struct
import shutil
from types import SimpleNamespace

import pytest

from baselines.kang.pipeline import RECIPE
from harness.lora_adapters import inspect_adapter
from repair.artifacts import checkpoint_files, seal, tokenizer_identity
from repair.e2_reader import read_collection, public_episode, prepare_examples
from repair.train_arms import THETA0_NAME
from sed.config import load_config
from sed.manifest import build_manifest, freeze_splits
from sed.seed_evidence import load_verified_collection, materialize_full_train_views, checked_episode
from sed.seed_selection import select_seeds, ALL_TRAIN_RULE
from sed.storage import digest, file_hash, write_json
from teachers.collect_seeds import collect
from test_full_train import factories
from test_sad import CharTokenizer
from train.modeling import LORA_TARGETS


def adapter_fixture(tmp_path, config=None):
    config = deepcopy(config or load_config('configs/base_e4b.yaml'))
    snapshot = tmp_path / 'snapshot'; snapshot.mkdir()
    write_json(snapshot / 'config.json', {'model_type': 'gemma4'})
    write_json(snapshot / 'tokenizer_config.json', {'chat_template': CharTokenizer.chat_template})
    theta = tmp_path / 'baseline' / 'checkpoint'; theta.mkdir(parents=True)
    targets = ['model.language_model.layers.0.' + ('mlp.' if t in ['gate_proj', 'up_proj', 'down_proj'] else 'self_attn.') + t
               for t in LORA_TARGETS]
    ac = {'peft_type': 'LORA', 'r': RECIPE['rank'], 'lora_alpha': RECIPE['alpha'], 'lora_dropout': RECIPE['dropout'],
          'target_modules': targets, 'base_model_name_or_path': config['models']['student']['requested_repo'],
          'revision': config['models']['student']['revision'], 'bias': 'none'}
    write_json(theta / 'adapter_config.json', ac)
    header, offset = {}, 0
    for target in targets:
        for side in ['A', 'B']:
            header[f'base_model.model.{target}.lora_{side}.weight'] = {
                'dtype': 'F32', 'shape': [64, 1] if side == 'A' else [1, 64],
                'data_offsets': [offset, offset + 256]}
            offset += 256
    raw = json.dumps(header).encode()
    (theta / 'adapter_model.safetensors').write_bytes(struct.pack('<Q', len(raw)) + raw + b'\0' * offset)
    recipe = {k: RECIPE[k] for k in ('adaptation', 'scheduler', 'warmup_ratio', 'warmup_steps',
        'weight_decay', 'packing', 'max_length')}
    recipe.update(objective='sft', learning_rate=5e-5, scoring='shared_history', epochs=32, seed=0, K=20,
        optimizer={'name': 'AdamW', 'betas': [.9, .999], 'eps': 1e-8}, attention='flex_attention',
        compute_dtype='bfloat16', adapter_master_dtype='float32', frozen_base_dtype='bfloat16')
    prov = {'config': {**config, 'baseline_training': recipe, 'sft': {
        'rank': ac['r'], 'alpha': ac['lora_alpha'], 'dropout': ac['lora_dropout'], 'targets': targets,
        'base_snapshot': {'path': str(snapshot), 'metadata_sha256': {
            p.name: file_hash(p) for p in snapshot.iterdir()}}}}}
    write_json(theta.parent / 'provenance.json', prov)
    state = {'identity_sha256': digest(prov), 'artifact_sha256': checkpoint_files(theta),
             'supervised_tokens': 10, 'passes': 32, 'packing': 'turn', 'gradient_clip': RECIPE['gradient_clip'],
             'loss_reduction': 'per_example_mean_then_global_batch_mean', 'compute_dtype': 'bfloat16',
             'master_dtype': 'float32', 'scoring': 'shared_history'}
    write_json(theta / 'training_state.json', state)
    return theta, snapshot, config


def build_corpus(tmp_path, task_count=24):
    root = tmp_path / 'games'
    for source in ('valid_seen', 'valid_unseen'):
        (root / source).mkdir(parents=True)
    for i in range(task_count):
        folder = root / 'train' / f'room_{i}' / 'trial_0'; folder.mkdir(parents=True)
        write_json(folder / 'traj_data.json', {'task_type': 'pick_and_place_simple'})
        write_json(folder / 'game.tw-pddl', {'solvable': True})
        (folder / 'initial_state.pddl').write_text('PRIVATE_PDDL')
    manifest = build_manifest(root); freeze = freeze_splits(manifest)
    config = load_config('configs/base_e4b.yaml')
    config['environment']['data_root'] = str(root)
    for key, value in [('manifest', manifest), ('split_freeze', freeze)]:
        config['paths'][key] = str(tmp_path / (key + '.json'))
        write_json(config['paths'][key], value)
    selection = select_seeds(manifest, freeze, rule=ALL_TRAIN_RULE)
    env, teacher = factories((root, manifest, freeze, selection, config), tmp_path / 'calls.jsonl')
    support = tmp_path / 'support'
    collect(selection, manifest, freeze, config, support, env_factory=env, teacher_factory=teacher)
    materialize_full_train_views(support, manifest, freeze, [20])
    view = support / 'subsets/k20'
    verified, _, evidence_root = load_verified_collection(view, manifest, freeze)
    failure_order = [r['task_id'] for r in verified['verified'][:6]]
    probes = sorted(failure_order, key=lambda t: (digest([20261029, t]), t))[:2]
    train = [t for t in failure_order if t not in probes]
    split = {'seed': 20261029, 'rule': 'lowest sha256([seed, task_id]); two before teacher calls',
             'failure_order': failure_order, 'D_repair_probe': probes, 'training_candidates': train}
    repair_splits = tmp_path / 'repair_splits.json'
    write_json(repair_splits, seal({'manifest_sha256': digest(manifest), 'split_freeze_sha256': digest(freeze),
                                  'sets': {'D_probe': {'group_ids': []}}}))
    theta, snapshot, config = adapter_fixture(tmp_path, config)
    prov = json.loads((theta.parent / 'provenance.json').read_text())
    prov['config']['baseline_training']['verified'] = str(view)
    prov['inputs'] = {str(view / 'manifest.json'): file_hash(view / 'manifest.json')}
    write_json(theta.parent / 'provenance.json', prov)
    state = json.loads((theta / 'training_state.json').read_text())
    state['identity_sha256'] = digest(prov)
    write_json(theta / 'training_state.json', state)
    tokenizer = CharTokenizer()
    checkpoint = {'identity': inspect_adapter(THETA0_NAME, theta, config['models']['student'], snapshot),
                  'files': checkpoint_files(theta)}
    tokens = tokenizer_identity(snapshot, tokenizer)
    input_hashes = {str(p): file_hash(p) for p in [config['paths']['manifest'], config['paths']['split_freeze'], repair_splits]}
    source = {'schema_version': 1, 'checkpoint': checkpoint, 'tokenizer': tokens, 'split': split,
              'D_repair_train': train, 'D_repair_probe': probes, 'K': 20,
              'input_hashes': input_hashes, 'dry_run': False, 'canary': False}
    source_path = tmp_path / 'collection.json'; write_json(source_path, seal(source))
    arms = {}
    for arm in ('N', 'F'):
        entries = []
        for tid in train:
            receipt = next(r for r in verified['verified'] if r['task_id'] == tid)
            ep = checked_episode(evidence_root, receipt)
            episode = public_episode(ep)
            if arm == 'F':
                episode['steps'][0]['action_raw'] = 'Thought: Inspect the visible room carefully.\nAction: look'
            le_id = digest([arm, tid])
            review = {'status': 'approved', 'reviewed': True, 'approved': True, 'evidence_sha256': digest(['review', arm, tid]),
                      'episode_sha256': digest(episode), 'decision': {'trajectory_id': arm + tid, 'episode_id': le_id,
                      'reviewer': 'TEST_ONLY', 'notes': 'Synthetic fixture, not a human approval.',
                      'no_rejected_actions': True, 'meaningful_route': True, 'labels_supported': True}}
            ref = {'task_id': tid, 'episode_id': ep['episode_id'], 'verified_collection_path': str(view),
                   'verified_manifest_sha256': file_hash(view / 'manifest.json'), 'evidence_root': str(evidence_root),
                   'episode_dir': receipt['episode_dir'], 'episode_sha256': file_hash(evidence_root / receipt['episode_dir'] / 'episode.json'),
                   'receipt': receipt, 'receipt_sha256': digest(receipt)}
            entries.append({'task_id': tid, 'group_id': receipt['group_id'], 'learning_episode_id': le_id,
                            'repair_id': 'repair-' + le_id, 'trajectory_id': arm + tid, 'source_bucket': 'successful_repair',
                            'episode': episode, 'episode_sha256': digest(episode), 'human_review': review,
                            'seed_demonstrations': [ref]})
        arms[arm] = entries
    data = {'format': 'repair_e2_training_collection', 'schema_version': 1,
            'source_collection_sha256': file_hash(source_path), 'source_collection_content_sha256': seal(source)['content_sha256'],
            'theta0': checkpoint, 'tokenizer': tokens, 'input_hashes': input_hashes, 'split': split,
            'D_repair_train': train, 'D_repair_probe': probes, 'K_trusted_seed_trajectories': 20,
            'unique_instances': 4, 'task_groups': 4, 'dry_run': False, 'canary': False, 'training_exposure': 0,
            'ready_for_training': True, 'training_blockers': [], 'arms': arms}
    path = tmp_path / 'e2_training_collection.json'; write_json(path, seal(data))
    return SimpleNamespace(path=path, data=data, source=source, theta=theta, snapshot=snapshot, config=config,
                           tokenizer=tokenizer, repair_splits=repair_splits, support=support, view=view)


_TEMPLATE = None


@pytest.fixture
def corpus(tmp_path, tmp_path_factory):
    # Build real fake-execution receipts once; each rejection gets independent
    # byte-identical evidence, exercising the contract's host path relocation.
    global _TEMPLATE
    if _TEMPLATE is None:
        _TEMPLATE = build_corpus(tmp_path_factory.mktemp('e2-template'))
    c = deepcopy(_TEMPLATE)
    old = c.path.parent
    shutil.copytree(old, tmp_path, dirs_exist_ok=True)
    def local(path):
        return tmp_path / Path(path).relative_to(old)
    c.path_maps = [(str(old), str(tmp_path))]
    c.local = local
    for key in ('path', 'theta', 'snapshot', 'repair_splits', 'support', 'view'):
        setattr(c, key, local(getattr(c, key)))
    for key in ('manifest', 'split_freeze'):
        c.config['paths'][key] = str(local(c.config['paths'][key]))
    return c


def read(corpus):
    return read_collection(corpus.path, config=corpus.config, theta0=corpus.theta, snapshot=corpus.snapshot,
                           tokenizer=corpus.tokenizer, repair_splits=corpus.repair_splits, path_maps=corpus.path_maps)


def save(c):
    write_json(c.path, seal(c.data))


def test_reader_nested_evidence_and_encoder_whitelist(corpus):
    c = read(corpus)
    assert set(c['seeds']) == set(corpus.data['D_repair_train'])
    assert all(len(v) == 1 for v in c['seeds'].values())
    for arm in ('N', 'F'):
        rows, _ = prepare_examples(c, corpus.tokenizer, arm, packing='turn', max_length=32768)
        text = ''.join(corpus.tokenizer.decode(t['input_ids']) for r in rows for t in r['turns'])
        assert not any(s in text for s in ['PRIVATE', 'SECRET', 'TEST_ONLY', 'receipt', 'success'])
        assert all(r['supervised_tokens'] > 0 for r in rows)


@pytest.mark.parametrize('case', ['format', 'version', 'bool_version', 'seal', 'source_bytes', 'source_seal',
    'dry_run', 'canary', 'synthetic', 'not_ready', 'blocker', 'exposure', 'missing_task', 'duplicate_task',
    'task_order', 'arm_order', 'probe_overlap', 'missing_arm', 'group', 'unreviewed', 'unapproved',
    'review_status', 'review_decision', 'review_id', 'review_hash', 'evidence_hash', 'episode_hash',
    'episode_changed', 'whitelist', 'seed_absent', 'seed_different', 'seed_receipt', 'seed_manifest',
    'seed_root', 'seed_escape', 'seed_task', 'seed_id', 'seed_hash', 'seed_file', 'theta_file',
    'base_revision', 'tokenizer_file', 'template', 'K', 'counts', 'frozen_input'])
def test_reader_rejects_every_contract_failure(corpus, case):
    c = corpus; d = c.data; e = d['arms']['N'][0]; r = e['human_review']; ref = e['seed_demonstrations'][0]
    if case == 'format': d['format'] = 'other'
    elif case == 'version': d['schema_version'] = 2
    elif case == 'bool_version': d['schema_version'] = True
    elif case == 'seal':
        save(c); c.path.write_text(c.path.read_text().replace('"schema_version":1', '"schema_version":3', 1)); return assert_reject(c)
    elif case == 'source_bytes': c.path.with_name('collection.json').write_text(c.path.with_name('collection.json').read_text() + '\n')
    elif case == 'source_seal':
        c.source['K'] = 9; write_json(c.path.with_name('collection.json'), seal(c.source))
        d['source_collection_sha256'] = file_hash(c.path.with_name('collection.json'))
    elif case in ('dry_run', 'canary', 'synthetic'): d[case] = True
    elif case == 'not_ready': d['ready_for_training'] = False
    elif case == 'blocker': d['training_blockers'] = ['unapproved']
    elif case == 'exposure': d['training_exposure'] = 1
    elif case == 'missing_task': d['D_repair_train'].pop()
    elif case == 'duplicate_task': d['D_repair_train'][1] = d['D_repair_train'][0]
    elif case == 'task_order': d['D_repair_train'].reverse()
    elif case == 'arm_order': d['arms']['N'].reverse()
    elif case == 'probe_overlap': d['D_repair_probe'][0] = d['D_repair_train'][0]
    elif case == 'missing_arm': del d['arms']['F']
    elif case == 'group': e['group_id'] = 'wrong'
    elif case == 'unreviewed': r['reviewed'] = False
    elif case == 'unapproved': r['approved'] = False
    elif case == 'review_status': r['status'] = 'pending'
    elif case == 'review_decision': r['decision']['labels_supported'] = False
    elif case == 'review_id': r['decision']['episode_id'] = 'different'
    elif case == 'review_hash': r['episode_sha256'] = '0' * 64
    elif case == 'evidence_hash': r['evidence_sha256'] = ''
    elif case == 'episode_hash': e['episode_sha256'] = '0' * 64
    elif case == 'episode_changed': e['episode']['steps'][0]['action_raw'] = 'Action: look'
    elif case == 'whitelist':
        e['episode']['initial']['visible']['facts'] = 'HIDDEN'
        e['episode_sha256'] = r['episode_sha256'] = digest(e['episode'])
    elif case == 'seed_absent':
        e['seed_demonstrations'].clear(); d['arms']['F'][0]['seed_demonstrations'].clear()
    elif case == 'seed_different': ref['episode_id'] = 'wrong'
    elif case.startswith('seed_'):
        if case == 'seed_receipt': ref['receipt']['won'] = False; ref['receipt_sha256'] = digest(ref['receipt'])
        elif case == 'seed_manifest': ref['verified_manifest_sha256'] = '0' * 64
        elif case == 'seed_root': ref['evidence_root'] = str(c.view)
        elif case == 'seed_escape': ref['episode_dir'] = '../outside'
        elif case == 'seed_task': ref['task_id'] = d['D_repair_probe'][0]
        elif case == 'seed_id': ref['episode_id'] = 'wrong'
        elif case == 'seed_hash': ref['episode_sha256'] = '0' * 64
        elif case == 'seed_file':
            p = c.local(Path(ref['evidence_root']) / ref['episode_dir'] / 'episode.json'); p.write_text(p.read_text() + '\n')
        d['arms']['F'][0]['seed_demonstrations'] = deepcopy(e['seed_demonstrations'])
    elif case == 'theta_file': (c.theta / 'extra.txt').write_text('changed')
    elif case == 'base_revision': c.config['models']['student']['revision'] = 'wrong'
    elif case == 'tokenizer_file': (c.snapshot / 'added_tokens.json').write_text('{}')
    elif case == 'template': c.tokenizer.chat_template = 'different'
    elif case == 'K': d['K_trusted_seed_trajectories'] = 4
    elif case == 'counts': d['task_groups'] = 1
    elif case == 'frozen_input':
        p = Path(c.config['paths']['manifest']); p.write_text(p.read_text() + '\n')
    save(c)
    assert_reject(c)


def assert_reject(c):
    with pytest.raises(ValueError):
        read(c)


def test_group_overlap_resolved_from_frozen_manifest(corpus, monkeypatch):
    import repair.e2_reader as reader
    original = reader.resolve_task
    train = corpus.data['D_repair_train'][0]; probe = corpus.data['D_repair_probe'][0]
    def resolve(m, f, tid):
        t = original(m, f, tid)
        if tid == probe:
            t = {**t, 'group_id': original(m, f, train)['group_id']}
        return t
    monkeypatch.setattr(reader, 'resolve_task', resolve)
    with pytest.raises(ValueError, match='overlap including groups'):
        read(corpus)


def test_independent_probe_groups_never_train(corpus):
    value = json.loads(corpus.repair_splits.read_text()); value.pop('content_sha256')
    value['sets']['D_probe']['group_ids'] = [corpus.data['arms']['N'][0]['group_id']]
    write_json(corpus.repair_splits, seal(value))
    corpus.data['input_hashes'][str(corpus.repair_splits)] = file_hash(corpus.repair_splits)
    corpus.source['input_hashes'] = corpus.data['input_hashes']
    write_json(corpus.path.with_name('collection.json'), seal(corpus.source))
    corpus.data['source_collection_sha256'] = file_hash(corpus.path.with_name('collection.json'))
    corpus.data['source_collection_content_sha256'] = seal(corpus.source)['content_sha256']
    save(corpus)
    with pytest.raises(ValueError, match='Independent probe'):
        read(corpus)


@pytest.mark.parametrize('field', ['theta0', 'tokenizer', 'split', 'arms', 'D_repair_train',
                                   'source_collection_sha256', 'K_trusted_seed_trajectories'])
def test_missing_required_fields_fail_closed(corpus, field):
    del corpus.data[field]; save(corpus)
    assert_reject(corpus)


def test_even_resealed_task_reordering_is_refused(corpus):
    corpus.data['D_repair_train'].reverse()
    for entries in corpus.data['arms'].values(): entries.reverse()
    corpus.source['D_repair_train'] = corpus.data['D_repair_train']
    write_json(corpus.path.with_name('collection.json'), seal(corpus.source))
    corpus.data['source_collection_sha256'] = file_hash(corpus.path.with_name('collection.json'))
    corpus.data['source_collection_content_sha256'] = seal(corpus.source)['content_sha256']
    save(corpus)
    with pytest.raises(ValueError, match='Misordered'):
        read(corpus)


def test_audit_and_native_arrays_never_enter_model(corpus):
    for entries in corpus.data['arms'].values():
        for entry in entries:
            entry['audit'] = {'expert_plan': 'PRIVATE_ORACLE'}
            entry['input_ids'] = [999999999]
            entry['labels'] = [999999999]
    save(corpus)
    rows, _ = prepare_examples(read(corpus), corpus.tokenizer, 'N', packing='turn', max_length=32768)
    for row in rows:
        for turn in row['turns']:
            assert 999999999 not in turn['input_ids']
            assert 'PRIVATE_ORACLE' not in corpus.tokenizer.decode(turn['input_ids'])


@pytest.mark.parametrize('text', ['{"x":1,"x":2}', '{"x":NaN}', '{"x":Infinity}'])
def test_noncanonical_json_rejected(tmp_path, text):
    from repair.artifacts import checked_document
    path = tmp_path / 'bad.json'; path.write_text(text)
    with pytest.raises(ValueError): checked_document(path)
