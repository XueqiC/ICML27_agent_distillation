"""Pinned HF runtime: P0 loading/determinism, optional one merged adapter."""
import os
from pathlib import Path

from repair.p0_pairs import MODEL, REVISION
from repair.p0_introspect import load_theta0
from repair.adapter_eval import training_identity
from sed.storage import file_hash
from surgery.common import read


def adapter_identity(path, *, e4=False):
    path = Path(path).resolve()
    value = dict(path=str(path), base_model=MODEL, revision=REVISION,
        adapter_weights_sha256=file_hash(path / 'adapter_model.safetensors'),
        adapter_config_sha256=file_hash(path / 'adapter_config.json'))
    if e4:
        training, inputs = training_identity(value)
        if training is None or not training['arm'].startswith('v0:'):
            raise ValueError('E4 requires a recorded V0 training-run checkpoint identity')
        value.update(training=training, training_inputs={p: file_hash(p) for p in inputs})
    return value


def load_runtime(base, adapter=None, expected=None, *, e4=False, max_new_tokens=1024, max_model_len=32768):
    required = dict(CUDA_DEVICE_ORDER='PCI_BUS_ID', TORCH_COMPILE_DISABLE='1', TORCHDYNAMO_DISABLE='1')
    if any(os.environ.get(k) != v for k, v in required.items()) or os.environ.get('CUDA_VISIBLE_DEVICES') not in tuple('01234567'):
        raise RuntimeError('Select one GPU and set P0 determinism environment variables')
    os.environ.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', CUBLAS_WORKSPACE_CONFIG=':4096:8')
    import torch
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError('Exactly one explicitly selected GPU required')
    base = Path(base)
    if base.name != REVISION:
        raise ValueError('Pinned E4B snapshot required')
    torch.manual_seed(0)
    torch.use_deterministic_algorithms(True)
    if adapter:
        recorded = adapter_identity(adapter, e4=e4)
        if expected:
            for key in ('base_model', 'revision', 'adapter_weights_sha256', 'adapter_config_sha256'):
                if recorded[key] != expected[key]:
                    raise ValueError('Configured adapter differs from frozen student')
        model, tokenizer, identity = load_theta0(base, adapter, recorded)
    else:
        if e4:
            raise ValueError('E4 needs an adapter')
        from transformers import Gemma4ForConditionalGeneration
        from train.dataset import load_tokenizer
        model = Gemma4ForConditionalGeneration.from_pretrained(base, local_files_only=True,
            trust_remote_code=False, dtype=torch.bfloat16, attn_implementation='sdpa', device_map={'': 'cuda:0'})
        tokenizer = load_tokenizer(base)
        from importlib.metadata import version
        identity = dict(base=MODEL, revision=REVISION, adapter=None, dtype='bfloat16', attention='sdpa',
            merge=None, greedy=True, base_files={p.name: file_hash(p) for p in sorted(base.iterdir())
                if p.is_file() and p.suffix in ('.json', '.jinja', '.safetensors')},
            dependency_versions={n: version(n) for n in ('torch', 'transformers', 'peft')})
    identity['runtime_device'] = dict(physical_gpu_index=int(os.environ['CUDA_VISIBLE_DEVICES']),
                                    device_name=torch.cuda.get_device_name(0))
    from surgery.student import Student
    return Student(model, tokenizer, identity=identity, max_new_tokens=max_new_tokens, max_model_len=max_model_len)


def identity_key(identity):
    """Paths/devices may relocate; numerical implementation and all bytes must agree."""
    from copy import deepcopy
    value = deepcopy(identity)
    value.pop('runtime_device', None)
    if value.get('adapter'):
        for key in ('path', 'training', 'training_inputs'):
            value['adapter'].pop(key, None)
    return value
