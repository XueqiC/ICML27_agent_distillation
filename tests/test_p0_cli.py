from types import SimpleNamespace

import pytest

from tools import p0_pilot


@pytest.mark.parametrize('physical', ['0', '3', '7', '0,1', '', '8', '-1', 'GPU-uuid', ' 0'])
def test_model_runtime_gpu_guard(monkeypatch, physical):
    import torch
    from repair import p0_introspect
    for key, value in dict(CUDA_DEVICE_ORDER='PCI_BUS_ID', TORCH_COMPILE_DISABLE='1',
                           TORCHDYNAMO_DISABLE='1', CUDA_VISIBLE_DEVICES=physical).items():
        monkeypatch.setenv(key, value)
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: True)
    monkeypatch.setattr(torch.cuda, 'device_count', lambda: 1)
    monkeypatch.setattr(torch.cuda, 'get_device_name', lambda i: 'Fake GPU' if i == 0 else pytest.fail('logical index'))
    loads = []
    monkeypatch.setattr(p0_introspect, 'load_theta0', lambda *a: loads.append(a) or (None, None, {'model': 'fake'}))
    monkeypatch.setattr(p0_introspect, 'StudentTools', lambda *a, **kw: SimpleNamespace(**kw))
    args = SimpleNamespace(base='fake-base', adapter='fake-adapter')
    pairs = dict(student={}, serving=dict(max_tokens=1, max_model_len=10))
    if physical in ('0', '3', '7'):
        result = p0_pilot.model_runtime(args, pairs)
        assert result.identity['runtime_device'] == dict(physical_gpu_index=int(physical), device_name='Fake GPU')
        assert len(loads) == 1
    else:
        with pytest.raises(RuntimeError, match='one index'):
            p0_pilot.model_runtime(args, pairs)
        assert not loads


@pytest.mark.parametrize('count', [0, 2])
def test_model_runtime_requires_one_torch_device(monkeypatch, count):
    import torch
    for key, value in dict(CUDA_DEVICE_ORDER='PCI_BUS_ID', TORCH_COMPILE_DISABLE='1',
                           TORCHDYNAMO_DISABLE='1', CUDA_VISIBLE_DEVICES='3').items():
        monkeypatch.setenv(key, value)
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: True)
    monkeypatch.setattr(torch.cuda, 'device_count', lambda: count)
    with pytest.raises(RuntimeError, match='exactly one'):
        p0_pilot.model_runtime(None, None)


@pytest.mark.parametrize('flag', ['CUDA_DEVICE_ORDER', 'TORCH_COMPILE_DISABLE', 'TORCHDYNAMO_DISABLE'])
def test_other_gpu_guards_remain_required(monkeypatch, flag):
    for key, value in dict(CUDA_DEVICE_ORDER='PCI_BUS_ID', TORCH_COMPILE_DISABLE='1',
                           TORCHDYNAMO_DISABLE='1', CUDA_VISIBLE_DEVICES='0').items():
        monkeypatch.setenv(key, value)
    monkeypatch.delenv(flag)
    with pytest.raises(RuntimeError, match='GPU operations require'):
        p0_pilot.model_runtime(None, None)


def test_evaluate_merge_does_not_load_model(tmp_path, monkeypatch):
    from repair import p0_eval, p0_agent
    calls = []
    monkeypatch.setattr(p0_pilot, 'load_pairs', lambda path: {'sha256': 'fake'})
    monkeypatch.setattr(p0_agent, 'load_submission', lambda *a, **kw: None)
    monkeypatch.setattr(p0_pilot, 'model_runtime', lambda *a: pytest.fail('model loaded during merge'))
    monkeypatch.setattr(p0_eval, 'merge_evaluation', lambda *a: calls.append(a) or {'complete': True})
    p0_pilot.main(['evaluate', '--pairs', 'fake', '--output', str(tmp_path), '--merge',
                   *sum((['--submission', f's{i}'] for i in range(5)), [])])
    assert len(calls) == 1
