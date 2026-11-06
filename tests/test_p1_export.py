"""Offline native shared-KV regression fixtures; no downloaded model weights."""
import logging
import warnings

import pytest
import torch
from safetensors.torch import load_file, save_file
from transformers import Gemma4ForCausalLM

from repair.p0_introspect import tensor_hash
from sed.storage import file_hash, write_json
from surgery.checkpoint import INDEX, checkpoint_files
from surgery.common import read, seal
from surgery.conditions import freeze
from surgery.edits import save_delta
from surgery.export import export_model, verify_export
from tests.test_p1_evaluation import checkpoint_student


def snapshot_fixture(tmp_path, *, sharded=False):
    s = checkpoint_student(tmp_path)
    snapshot = tmp_path / 'snapshot'
    s.model.save_pretrained(snapshot)
    weights = load_file(snapshot / 'model.safetensors')
    shared = {}
    for layer in (2, 3):
        for target in ('k_proj', 'v_proj', 'k_norm'):
            assert getattr(s.layers[layer].self_attn, target, None) is None
            shape = (8,) if target == 'k_norm' else (8, 16)
            dtype = torch.float32 if target == 'k_norm' else torch.bfloat16
            tensor = torch.arange(torch.tensor(shape).prod().item(), dtype=dtype).reshape(shape)
            tensor.reshape(-1)[0] = -0.0
            shared[f'model.layers.{layer}.self_attn.{target}.weight'] = tensor
    weights.update(shared)
    if sharded:
        (snapshot / 'model.safetensors').unlink()
        names = sorted(weights)
        mapping = {}
        for index, subset in enumerate((names[::2], names[1::2]), 1):
            name = f'model-{index:05d}-of-00002.safetensors'
            save_file({n: weights[n] for n in subset}, snapshot / name, metadata={'format': 'pt'})
            mapping.update({n: name for n in subset})
        write_json(snapshot / INDEX, dict(weight_map=mapping, metadata=dict(
            total_size=sum(t.numel() * t.element_size() for t in weights.values()))))
    else:
        save_file(weights, snapshot / 'model.safetensors', metadata={'format': 'pt'})
    return s, snapshot, shared


def reload(path):
    return Gemma4ForCausalLM.from_pretrained(path, local_files_only=True, dtype=torch.bfloat16,
        attn_implementation='sdpa', output_loading_info=True)


def export(s, snapshot, tmp_path, *, loader=reload, edits=None):
    condition = freeze(tmp_path / 'condition.json', s, 'pairs', 'A', 'E1', edits or [], trial=True)
    return export_model(s, condition, snapshot, tmp_path / 'export', loader=loader, inputs=[[130, 131]])


@pytest.mark.parametrize('base_sharded', [False, True])
@pytest.mark.parametrize('export_sharded', [False, True])
def test_shared_kv_preserved_index_manifest_and_hf_reload(tmp_path, monkeypatch, base_sharded, export_sharded):
    s, snapshot, shared = snapshot_fixture(tmp_path, sharded=base_sharded)
    if export_sharded:
        save = s.model.save_pretrained
        monkeypatch.setattr(s.model, 'save_pretrained', lambda p, **kw: save(p, max_shard_size='10KB', **kw))
    def loader(path):
        warnings.warn('Fixture unexpected shared keys', UserWarning)
        logging.getLogger('transformers.modeling_utils').warning('Fixture shared-KV load warning')
        return reload(path)
    edit = dict(type='scale_neurons', layers=[0], neurons={'0': [0]}, c=0.5)
    manifest = export(s, snapshot, tmp_path, loader=loader, edits=[edit])
    output = tmp_path / 'export'
    assert manifest['passed'] and manifest['weights_exact'] and manifest['edited_weights_exact']
    assert manifest['base_snapshot']['path'] == str(snapshot.resolve())
    assert set(manifest['copied_tensors']) == set(shared)
    assert not (output / 'model.safetensors').exists()  # Force HF to load the index.
    files = checkpoint_files(output)
    assert set(files) == set(checkpoint_files(snapshot))
    size = 0
    for path in set(files.values()):
        size += sum(t.numel() * t.element_size() for t in load_file(path).values())
    assert read(output / INDEX)['metadata']['total_size'] == size
    for name, expected in shared.items():
        actual = load_file(files[name])[name]
        assert actual.dtype == expected.dtype and actual.shape == expected.shape
        assert torch.equal(actual.view(torch.uint8), expected.view(torch.uint8))
        entry = manifest['copied_tensors'][name]
        assert entry['source'] == 'base_snapshot' and entry['sha256'] == tensor_hash(expected)
        assert entry['source_file'] == checkpoint_files(snapshot)[name].name
    report = read(output / 'consistency.json')
    assert set(report['weight_checks']) == set(files)
    assert all(c['exact'] for c in report['weight_checks'].values())
    assert all(c['source'] == 'base_snapshot' and c['base_sha256'] == c['exported_sha256']
               for n, c in report['weight_checks'].items() if n in shared)
    assert report['edited_weight_checks']['0:mlp.down_proj']['source'] == 'installed'
    assert report['checks'][0]['bf16_reload']['exact']
    # Gemma4 5.14.1 explicitly suppresses shared-KV unexpected-key warnings.
    assert report['reload_unexpected_keys'] == []
    assert any(w['message'] == 'Fixture unexpected shared keys' for w in manifest['reload_warnings'])
    assert any(w['message'] == 'Fixture shared-KV load warning' for w in manifest['reload_warnings'])
    assert verify_export(output) == manifest


@pytest.mark.parametrize('kind', ['add_rank1', 'local_finetune'])
def test_refuse_edit_to_copied_tensor_before_edit_or_save(tmp_path, monkeypatch, kind):
    s, snapshot, shared = snapshot_fixture(tmp_path)
    condition = freeze(tmp_path / 'condition.json', s, 'pairs', 'A', 'E1', [])
    if kind == 'local_finetune':
        ref = save_delta(tmp_path / 'delta', {'2:self_attn.k_proj': torch.ones(8, 16)}, {})
        edit = dict(type=kind, delta=ref, alpha=1)
    else:
        edit = dict(type=kind, layers=[2], targets=['self_attn.k_proj'], alpha=1,
                    a={'direction_id': 'unused'}, b={'direction_id': 'unused'})
    condition = seal(dict(condition, edits=[edit]))
    monkeypatch.setattr(s.model, 'save_pretrained', lambda *a, **k: pytest.fail('Must reject before save'))
    with pytest.raises(ValueError, match='copied tensor targeted by edit or LoRA merge'):
        export_model(s, condition, snapshot, tmp_path / 'export', inputs=[[130, 131]])
    assert not (tmp_path / 'export').exists()


@pytest.mark.parametrize('layer', [0, 2])
def test_audit_actual_merged_lora_keys(tmp_path, layer):
    s, snapshot, shared = snapshot_fixture(tmp_path)
    adapter = tmp_path / 'adapter'
    adapter.mkdir()
    prefix = f'base_model.model.model.layers.{layer}.self_attn.k_proj'
    save_file({prefix + '.lora_A.weight': torch.zeros(2, 16),
               prefix + '.lora_B.weight': torch.zeros(8, 2)}, adapter / 'adapter_model.safetensors')
    write_json(adapter / 'adapter_config.json', dict(target_modules=['k_proj', 'v_proj']))
    s.identity['adapter'] = dict(path=str(adapter), adapter_weights_sha256=file_hash(adapter / 'adapter_model.safetensors'))
    s.identity['merge'] = 'PEFT in memory'
    if layer == 2:
        with pytest.raises(ValueError, match='copied tensor targeted by edit or LoRA merge'):
            export(s, snapshot, tmp_path)
    else:
        result = export(s, snapshot, tmp_path)
        assert result['merged_tensor_names'] == ['model.layers.0.self_attn.k_proj.weight']
        assert result['passed']


@pytest.mark.parametrize('target', ['copied', 'unedited'])
def test_all_tensor_byte_check_rejects_signed_zero_corruption(tmp_path, monkeypatch, target):
    import safetensors.torch
    s, snapshot, shared = snapshot_fixture(tmp_path)
    name = next(iter(shared)) if target == 'copied' else 'model.layers.0.mlp.down_proj.weight'
    if target == 'unedited':
        s.layers[0].mlp.down_proj.weight.data[0, 0] = -0.0
    original_save = safetensors.torch.save_file
    def corrupt(weights, path, **kwargs):
        original_save(weights, path, **kwargs)
        if name in weights and str(path).startswith(str(tmp_path / 'export')):
            saved = load_file(path)
            saved[name].reshape(-1)[0] = 0.0
            original_save(saved, path, metadata={'format': 'pt'})
    monkeypatch.setattr(safetensors.torch, 'save_file', corrupt)
    # Transformers may hold a directly imported save_file alias.
    if target == 'unedited':
        save = s.model.save_pretrained
        def corrupt_model(path, **kwargs):
            save(path, **kwargs)
            corrupt(load_file(path / 'model.safetensors'), path / 'model.safetensors', metadata={'format': 'pt'})
        monkeypatch.setattr(s.model, 'save_pretrained', corrupt_model)
    with pytest.raises(ValueError, match='byte-for-byte'):
        export(s, snapshot, tmp_path)
    report = read(tmp_path / 'export/consistency.json')
    assert not report['weight_checks'][name]['exact'] and not report['weights_exact']
    assert report['edited_weights_exact']  # No edits; the old check would have passed.
    assert report['checks'][0]['bf16_reload']['exact']
    with pytest.raises(ValueError, match='consistency'):
        verify_export(tmp_path / 'export')


@pytest.mark.parametrize('change', ['missing', 'extra', 'index', 'duplicate'])
def test_verify_export_checks_actual_base_key_set_even_with_resealed_files(tmp_path, change):
    s, snapshot, shared = snapshot_fixture(tmp_path)
    manifest = export(s, snapshot, tmp_path)
    output = tmp_path / 'export'
    shard = output / 'model-base-only.safetensors'
    weights = load_file(shard)
    index = read(output / INDEX)
    name = next(iter(shared))
    if change in ('missing', 'index'):
        del index['weight_map'][name]
        if change == 'missing':
            del weights[name]
    elif change == 'extra':
        weights['extra.weight'] = torch.zeros(1)
        index['weight_map']['extra.weight'] = shard.name
    else:
        name = 'model.layers.0.mlp.down_proj.weight'
        weights[name] = s.layers[0].mlp.down_proj.weight.detach().clone()
    save_file(weights, shard, metadata={'format': 'pt'})
    write_json(output / INDEX, index)
    manifest['files'] = {str(p.relative_to(output)): file_hash(p) for p in output.rglob('*')
                         if p.is_file() and p.name != 'surgery_manifest.json'}
    write_json(output / 'surgery_manifest.json', seal(manifest))
    with pytest.raises(ValueError, match='tensor key set|shard index|Duplicate checkpoint tensor'):
        verify_export(output)
