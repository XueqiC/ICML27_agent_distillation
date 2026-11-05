from pathlib import Path
import pytest

from tools import p1_surgery
from surgery.runtime import adapter_identity, load_runtime
from sed.storage import file_hash


def test_e4_uses_recorded_training_identity(tmp_path, monkeypatch):
    path = tmp_path / 'update-000100/checkpoint'
    path.mkdir(parents=True)
    (path / 'adapter_model.safetensors').write_bytes(b'fake')
    (path / 'adapter_config.json').write_text('{}')
    seen = []
    def training(value):
        seen.append(value)
        return dict(arm='v0:E4', optimizer_updates=100), []
    monkeypatch.setattr('surgery.runtime.training_identity', training)
    value = adapter_identity(path, e4=True)
    assert seen[0]['adapter_weights_sha256'] == file_hash(path / 'adapter_model.safetensors')
    assert value['training']['optimizer_updates'] == 100
    monkeypatch.setattr('surgery.runtime.training_identity', lambda v: (None, []))
    with pytest.raises(ValueError, match='recorded'):
        adapter_identity(path, e4=True)


def test_runtime_cannot_load_on_cpu_test_environment():
    with pytest.raises(RuntimeError):
        load_runtime('/must-not-load')


def test_trial_overrides_need_explicit_trial(tmp_path, monkeypatch):
    monkeypatch.setattr(p1_surgery, 'runtime', lambda *a, **k: pytest.fail('Must fail before model'))
    with pytest.raises(SystemExit):
        p1_surgery.main(['diagnose', '--allow-teacher-api', '--tool-calls', '10', '--output', str(tmp_path / 'out')])


def test_new_launcher_uses_historical_code_and_e4b_config():
    source = Path('surgery/hpg_dev140.slurm').read_text()
    assert '46b8527' in source
    assert '${PROJECT_ROOT}/configs/base_e4b.yaml' in source
    assert '--merged-model-path "$MERGED_MODEL_PATH"' in source
    assert '--policy student --decoding greedy' in source
    assert '--workers 16' in source
    assert 'VLLM_BATCH_INVARIANT=1' in source
