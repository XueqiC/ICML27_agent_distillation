"""Preserve snapshot-only weights without applying surgery or adapter deltas."""
from pathlib import Path
import re

from repair.p0_introspect import tensor_hash
from sed.storage import file_hash, write_json
from surgery.common import read
from surgery.edits import load_delta


INDEX = 'model.safetensors.index.json'


def checkpoint_files(directory):
    """Inspect actual shard keys, rejecting duplicate or loader-ambiguous layouts."""
    from safetensors import safe_open
    directory = Path(directory)
    files = {}
    for path in sorted(directory.glob('*.safetensors')):
        with safe_open(path, framework='pt', device='cpu') as handle:
            for name in handle.keys():
                if name in files:
                    raise ValueError('Duplicate checkpoint tensor: ' + name)
                files[name] = path
    if not files:
        raise ValueError('Checkpoint has no safetensors weights')
    if (directory / INDEX).exists():
        if (directory / 'model.safetensors').exists():
            raise ValueError('Ambiguous standalone checkpoint and shard index')
        if read(directory / INDEX)['weight_map'] != {n: p.name for n, p in files.items()}:
            raise ValueError('Checkpoint shard index differs from actual tensor keys')
    elif set(files.values()) != {directory / 'model.safetensors'}:
        raise ValueError('Sharded checkpoint requires a model.safetensors.index.json')
    return files


def edit_tensor_names(student, edits):
    """Resolve intended names even when a shared layer has no native projection."""
    modules = {id(m): n for n, m in student.model.named_modules()}
    keys = set()
    for edit in edits:
        if edit['type'] == 'local_finetune':
            keys.update(load_delta(edit['delta'], student.relocation))
        else:
            keys.update(f'{layer}:{target}' for layer in edit['layers']
                        for target in edit.get('targets', ['mlp.down_proj']))
    result = {}
    for key in keys:
        layer, target = key.split(':')
        student.layer_ids([int(layer)])
        result[key] = modules[id(student.layers[int(layer)])] + '.' + target + '.weight'
    return result


def merged_tensor_names(student, base_keys):
    """Audit actual LoRA weights, not broad config targets that include absent KV."""
    from safetensors import safe_open
    adapter = student.identity.get('adapter')
    if not adapter:
        if student.identity.get('merge'):
            raise ValueError('Cannot audit merged tensors without adapter identity')
        return set()
    path = student.relocation.path(adapter['path']) / 'adapter_model.safetensors'
    if file_hash(path) != adapter['adapter_weights_sha256']:
        raise ValueError('Merged adapter weights changed')
    result = set()
    with safe_open(path, framework='pt', device='cpu') as handle:
        for key in handle.keys():
            name, count = re.subn(r'\.lora_[AB](?:\.[^.]+)?\.weight$', '.weight', key)
            name = name.removeprefix('base_model.model.')
            if count != 1 or name not in base_keys:
                raise ValueError('Cannot audit merged LoRA tensor: ' + key)
            result.add(name)
    return result


def assert_copy_untouched(missing, edited, merged):
    conflict = set(missing) & (set(edited) | set(merged))
    if conflict:
        raise ValueError('Snapshot-only copied tensor targeted by edit or LoRA merge: ' + ', '.join(sorted(conflict)))


def snapshot_record(student, snapshot, files):
    hashes = {p.name: file_hash(p) for p in sorted(set(files.values()))}
    if (Path(snapshot) / INDEX).exists():
        hashes[INDEX] = file_hash(Path(snapshot) / INDEX)
    frozen = student.identity.get('base_files')
    if frozen is not None and any(frozen.get(n) != sha for n, sha in hashes.items()):
        raise ValueError('Base snapshot differs from frozen student')
    return dict(path=str(Path(snapshot).resolve()), files=hashes, tensor_names=sorted(files))


def copy_missing_tensors(output, base_files, edited, merged):
    """Add a standard indexed shard; never round or cast the base tensors."""
    from safetensors import safe_open
    from safetensors.torch import save_file
    output = Path(output)
    saved = checkpoint_files(output)
    if set(saved) - set(base_files):
        raise ValueError('Export tensor key set contains tensors absent from base snapshot')
    missing = set(base_files) - set(saved)
    assert_copy_untouched(missing, edited, merged)
    copied, tensors = {}, {}
    for path in sorted({base_files[n] for n in missing}):
        with safe_open(path, framework='pt', device='cpu') as handle:
            for name in sorted(n for n in missing if base_files[n] == path):
                tensor = handle.get_tensor(name)
                tensors[name] = tensor
                copied[name] = dict(source='base_snapshot', source_file=path.name,
                    sha256=tensor_hash(tensor.reshape(-1)), dtype=str(tensor.dtype).removeprefix('torch.'),
                    shape=list(tensor.shape))
    if missing:
        # HF prefers a standalone file over an index, so it must be renamed.
        standalone = output / 'model.safetensors'
        if standalone.exists():
            shard = output / 'model-installed.safetensors'
            standalone.rename(shard)
            saved = {n: shard for n in saved}
        extra = output / 'model-base-only.safetensors'
        save_file(tensors, extra, metadata={'format': 'pt'})
        saved.update({n: extra for n in missing})
        del tensors
        total_size = 0
        for path in sorted(set(saved.values())):
            with safe_open(path, framework='pt', device='cpu') as handle:
                for name in handle.keys():
                    tensor = handle.get_tensor(name)
                    total_size += tensor.numel() * tensor.element_size()
        write_json(output / INDEX, dict(metadata=dict(total_size=total_size),
            weight_map={n: p.name for n, p in sorted(saved.items())}))
    if set(checkpoint_files(output)) != set(base_files):
        raise ValueError('Export tensor key set differs from base snapshot')
    return copied


def all_weight_checks(student, output, base_files, copied):
    """Every exported tensor must match installed state or original snapshot bytes."""
    import torch
    from safetensors import safe_open
    state = student.model.state_dict()
    files = checkpoint_files(output)
    checks = {}
    for path in sorted(set(files.values())):
        with safe_open(path, framework='pt', device='cpu') as handle:
            for name in sorted(n for n in files if files[n] == path):
                exported = handle.get_tensor(name).contiguous()
                if name in copied:
                    with safe_open(base_files[name], framework='pt', device='cpu') as base:
                        expected = base.get_tensor(name).contiguous()
                    source = 'base_snapshot'
                else:
                    if name not in state:
                        raise ValueError('Exported tensor absent from installed state: ' + name)
                    expected = state[name].detach().cpu().contiguous()
                    source = 'installed'
                checks[name] = dict(tensor=name, source=source,
                    **{('base_sha256' if name in copied else 'installed_sha256'): tensor_hash(expected.reshape(-1))},
                    exported_sha256=tensor_hash(exported.reshape(-1)),
                    dtype=str(expected.dtype).removeprefix('torch.'), shape=list(expected.shape),
                    exact=(expected.dtype == exported.dtype and expected.shape == exported.shape
                           and torch.equal(expected.reshape(-1).view(torch.uint8), exported.reshape(-1).view(torch.uint8))))
                del expected, exported
    return checks
