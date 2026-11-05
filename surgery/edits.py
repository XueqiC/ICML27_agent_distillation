"""Ordered fp32 edits with exact parameter-object/byte rollback and frozen deltas."""
from contextlib import contextmanager
from copy import deepcopy
import math
from pathlib import Path

from repair.p0_introspect import tensor_hash
from sed.storage import digest, file_hash
from surgery.common import Relocation, checked, publish, read

TARGETS = ('self_attn.q_proj', 'self_attn.k_proj', 'self_attn.v_proj', 'self_attn.o_proj',
           'mlp.gate_proj', 'mlp.up_proj', 'mlp.down_proj')
OUTPUTS = ('mlp.down_proj', 'self_attn.o_proj')
NORMS = {'mlp.down_proj': 'post_feedforward_layernorm', 'self_attn.o_proj': 'post_attention_layernorm'}


def finite(value):
    if type(value) not in (int, float) or not math.isfinite(value):
        raise ValueError('Finite scalar required')
    return float(value)


def module_at(student, layer, target):
    student.layer_ids([layer])
    if target not in TARGETS:
        raise ValueError('Only q/k/v/o/gate/up/down projections are supported')
    parent, name = target.split('.')
    module = getattr(getattr(student.layers[layer], parent), name, None)
    if module is None or not hasattr(module, 'weight') or module.weight.ndim != 2:
        raise ValueError('Projection absent (including native shared-KV layers)')
    return module


def tensor_name(layer, target):
    return f'{layer}:{target}'


def all_modules(student):
    result = {}
    for i in range(len(student.layers)):
        for name in TARGETS:
            try:
                result[tensor_name(i, name)] = module_at(student, i, name)
            except ValueError:
                continue
    return result


def save_delta(directory, tensors, metadata):
    from safetensors.torch import save
    from sed.storage import atomic_write
    import torch
    if not tensors or any(t.ndim != 2 or not torch.isfinite(t).all() for t in tensors.values()):
        raise ValueError('Finite dense delta tensors required')
    tensors = {k: v.detach().cpu().float().contiguous() for k, v in tensors.items()}
    ident = digest(dict(tensors={k: tensor_hash(t) for k, t in tensors.items()}, metadata=metadata))
    path = Path(directory) / (ident + '.safetensors')
    data = save(tensors)
    if path.exists():
        if path.read_bytes() != data:
            raise ValueError('Delta artifact changed')
    else:
        atomic_write(path, data, exclusive=True)
    receipt = dict(path=str(path.resolve()), sha256=file_hash(path), metadata=metadata,
                   tensors={k: dict(shape=list(t.shape), frobenius=float(t.norm()), dtype='float32') for k, t in tensors.items()})
    meta = path.with_suffix('.json')
    if not meta.exists():
        publish(meta, receipt)
    return receipt


def load_delta(ref, relocation=None):
    from safetensors.torch import load_file
    import torch
    path = (relocation or Relocation()).path(ref['path'])
    if file_hash(path) != ref['sha256']:
        raise ValueError('Frozen dense delta sha256 mismatch')
    result = load_file(str(path), device='cpu')
    if not result or any(t.dtype != torch.float32 or t.ndim != 2 or not torch.isfinite(t).all() for t in result.values()):
        raise ValueError('Invalid fp32 dense delta')
    return result


def vector(student, ref, layer, size):
    import torch
    if not isinstance(ref, dict) or set(ref) != {'direction_id'}:
        raise ValueError('Vector reference requires a direction_id from a tool')
    d = student.directions[ref['direction_id']][layer].float()
    if d.shape != (size,) or not torch.isfinite(d).all() or abs(float(d.norm()) - 1) > 1e-4:
        raise ValueError('Vector dimension/unit-norm mismatch')
    return d.cpu()


def validate_edits(student, edits):
    import torch
    if not isinstance(edits, list):
        raise ValueError('Submission/edit must be an ordered list')
    for e in edits:
        kind = e['type']
        if kind == 'local_finetune':
            if set(e) != {'type', 'delta', 'alpha'}:
                raise ValueError('Local finetune edit requires frozen delta and alpha')
            finite(e['alpha'])
            tensors = load_delta(e['delta'], student.relocation)
            for key, delta in tensors.items():
                layer, target = key.split(':')
                if delta.shape != module_at(student, int(layer), target).weight.shape:
                    raise ValueError('Delta tensor shape mismatch')
            continue
        if kind not in ('project_out', 'add_rank1', 'scale_neurons'):
            raise ValueError('Unknown P1 edit')
        student.layer_ids(e['layers'])
        if kind == 'scale_neurons':
            if set(e) != {'type', 'layers', 'neurons', 'c'}:
                raise ValueError('scale_neurons schema')
            finite(e['c'])
            for layer in e['layers']:
                width = module_at(student, layer, 'mlp.down_proj').weight.shape[1]
                ids = e['neurons'][str(layer)]
                if not ids or len(set(ids)) != len(ids) or any(type(i) is not int or not 0 <= i < width for i in ids):
                    raise ValueError('Invalid neuron indices')
            continue
        required = {'type', 'layers', 'targets', 'alpha'} | ({'direction_ids'} if kind == 'project_out' else {'a', 'b'})
        allowed = OUTPUTS if kind == 'project_out' else TARGETS
        if set(e) != required or not e['targets'] or len(set(e['targets'])) != len(e['targets']) or not set(e['targets']) <= set(allowed):
            raise ValueError('Projection edit schema')
        finite(e['alpha'])
        if kind == 'project_out' and (not isinstance(e['direction_ids'], list) or not 1 <= len(e['direction_ids']) <= 8
                                    or len(set(e['direction_ids'])) != len(e['direction_ids'])):
            raise ValueError('project_out needs 1..8 distinct direction IDs')
        for layer in e['layers']:
            for target in e['targets']:
                module = module_at(student, layer, target)
                if kind == 'add_rank1':
                    vector(student, e['a'], layer, module.weight.shape[0])
                    vector(student, e['b'], layer, module.weight.shape[1])
                else:
                    from transformers.models.gemma4.modeling_gemma4 import Gemma4RMSNorm
                    block = student.layers[layer]
                    if (getattr(block, 'enable_moe_block', False) or any(hasattr(block, n) for n in
                        ('post_feedforward_layernorm_1', 'post_feedforward_layernorm_2'))
                        or not isinstance(getattr(block, NORMS[target], None), Gemma4RMSNorm)):
                        raise ValueError('Expected native dense Gemma4 post-norm')
                    for ident in e['direction_ids']:
                        vector(student, dict(direction_id=ident), layer, module.weight.shape[0])


def rounded_weight(weight, original):
    """Round once for native execution; preserve all bytes of a numerical no-op."""
    import torch
    original = original.detach().cpu()
    rounded = weight.to(original.dtype)
    return original if torch.equal(rounded, original) else rounded


def rounding_stats(weight, original):
    import torch
    original = original.detach().cpu()
    rounded = rounded_weight(weight, original)
    base = original.float()
    norm = float(torch.linalg.vector_norm(base, dtype=torch.float64))
    delta_norm = float(torch.linalg.vector_norm(rounded.float() - base, dtype=torch.float64))
    # An uninstalled prefix can overflow the native dtype and later cancel in fp32.
    delta_norm = delta_norm if math.isfinite(delta_norm) else None
    fp32_changed = int(torch.count_nonzero(weight != base))
    changed = int(torch.count_nonzero(rounded != original))
    return dict(parameters=weight.numel(), native_dtype=str(original.dtype).removeprefix('torch.'),
        fp32_changed_parameters=fp32_changed, rounded_changed_parameters=changed,
        rounded_finite=bool(torch.isfinite(rounded).all()),
        original_frobenius=norm, rounded_delta_frobenius=delta_norm,
        rounded_relative_frobenius=delta_norm / norm if norm and delta_norm is not None else (0.0 if delta_norm == 0 else None),
        vanished=fp32_changed > 0 and changed == 0)


def rounding_summary(tensors):
    norm = math.hypot(*(t['original_frobenius'] for t in tensors.values()))
    deltas = [t['rounded_delta_frobenius'] for t in tensors.values()]
    delta_norm = math.hypot(*deltas) if all(d is not None for d in deltas) else None
    fp32_changed = sum(t['fp32_changed_parameters'] for t in tensors.values())
    changed = sum(t['rounded_changed_parameters'] for t in tensors.values())
    return dict(tensors=tensors, fp32_changed_parameters=fp32_changed, rounded_changed_parameters=changed,
        rounded_finite=all(t['rounded_finite'] for t in tensors.values()),
        original_frobenius=norm, rounded_delta_frobenius=delta_norm,
        rounded_relative_frobenius=delta_norm / norm if norm and delta_norm is not None else (0.0 if delta_norm == 0 else None),
        vanished=fp32_changed > 0 and changed == 0)


def composed_weights(student, edits, *, rounding=None):
    """Compose in CPU fp32; diagnostic prefix rounding never feeds back into edits."""
    import torch
    validate_edits(student, edits)
    weights = {}
    def current(layer, target):
        key = tensor_name(layer, target)
        if key not in weights:
            weights[key] = module_at(student, layer, target).weight.detach().cpu().float().clone()
        return weights[key]
    for index, e in enumerate(edits):
        kind = e['type']
        tensors = {}
        if kind == 'local_finetune':
            updates = load_delta(e['delta'], student.relocation).items()
        else:
            updates = ((tensor_name(layer, target), None) for layer in e['layers']
                       for target in e.get('targets', ['mlp.down_proj']))
        for key, delta in updates:
            layer, target = key.split(':')
            layer = int(layer)
            w = current(layer, target)
            if rounding is not None:
                before = w.clone()
            if kind == 'local_finetune':
                w.add_(delta, alpha=finite(e['alpha']))
            elif kind == 'scale_neurons':
                w[:, e['neurons'][str(layer)]] *= finite(e['c'])
            elif kind == 'add_rank1':
                a = vector(student, e['a'], layer, w.shape[0])
                b = vector(student, e['b'], layer, w.shape[1])
                w.add_(a[:, None] * b[None, :], alpha=finite(e['alpha']))
            else:
                ds = torch.stack([vector(student, dict(direction_id=i), layer, w.shape[0]) for i in e['direction_ids']], dim=1)
                g = getattr(student.layers[layer], NORMS[target]).weight.detach().cpu().float()
                scaled = g[:, None] * ds
                if not torch.isfinite(scaled).all() or torch.linalg.matrix_rank(scaled).item() != len(e['direction_ids']):
                    raise ValueError('Post-norm directions must be linearly independent')
                u = torch.linalg.qr(scaled, mode='reduced').Q
                w.sub_(finite(e['alpha']) * (u @ (u.T @ w)))
            if rounding is not None:
                original = module_at(student, layer, target).weight.detach().cpu()
                tensors[key] = dict(rounding_stats(w, original),
                    fp32_effect_changed_parameters=int(torch.count_nonzero(w != before)),
                    rounded_effect_changed_parameters=int(torch.count_nonzero(
                        rounded_weight(w, original) != rounded_weight(before, original))))
                del before
        if rounding is not None:
            fp32_effect = sum(t['fp32_effect_changed_parameters'] for t in tensors.values())
            rounded_effect = sum(t['rounded_effect_changed_parameters'] for t in tensors.values())
            rounding.append(dict(rounding_summary(tensors), index=index, type=kind,
                fp32_effect_changed_parameters=fp32_effect, rounded_effect_changed_parameters=rounded_effect,
                effect_vanished=fp32_effect > 0 and rounded_effect == 0))
    if any(not torch.isfinite(w).all() for w in weights.values()):
        raise ValueError('Nonfinite composed weights')
    return weights


@contextmanager
def edited(student, edits):
    import torch
    if edits is None:
        edits = []
    if student.active_edit is not None:
        raise ValueError('Nested weight edits forbidden')
    per_edit = []
    weights = composed_weights(student, edits, rounding=per_edit)
    originals, tensors = [], {}
    student.active_edit = deepcopy(edits)
    try:
        for key, w in weights.items():
            layer, target = key.split(':')
            module = module_at(student, int(layer), target)
            original = module.weight
            originals.append((module, original, tensor_hash(original)))
            native = original.detach().cpu()
            tensors[key] = rounding_stats(w, native)
            rounded = rounded_weight(w, native)
            if not torch.isfinite(rounded).all():
                raise ValueError('Nonfinite native-dtype composed weights')
            if torch.equal(rounded, native):
                continue  # alpha=0 preserves original arithmetic and signed zero bytes.
            module.weight = torch.nn.Parameter(rounded.to(original.device), requires_grad=False)
        count = sum(w.numel() for w in weights.values())
        total = sum(p.numel() for p in student.model.parameters())
        yield dict(edited_parameters=count, edited_fraction=count / total, fp32_composition=True,
                   native_execution=True, rounding=dict(rounding_summary(tensors), per_edit=per_edit),
                   edit_sha256=digest(edits), exact_rollback=True)
    finally:
        for module, original, sha in originals:
            module.weight = original
            if tensor_hash(module.weight) != sha:
                raise RuntimeError('Exact rollback verification failed')
        student.active_edit = None
