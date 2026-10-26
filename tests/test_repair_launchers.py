from pathlib import Path
import subprocess

import pytest


@pytest.mark.parametrize('stage,gpus', [('coverage', 1), ('probes', 1)])
def test_stage_resources_and_no_overwrite_guard(stage, gpus, tmp_path):
    path = Path(f'scripts/hpg_repair_{stage}.slurm')
    text = path.read_text()
    assert f'#SBATCH --gpus={gpus}' in text
    assert 'repair_stage_env.sh' in text
    subprocess.run(['bash', '-n', str(path)], check=True)
    common = Path('scripts/repair_stage_env.sh').read_text()
    assert 'VLLM_BATCH_INVARIANT=1' in common
    for key in ['ALFWORLD_DATA', 'VLLM_CACHE_ROOT', 'TRITON_CACHE_DIR', 'HF_HOME', 'XDG_CACHE_HOME']:
        assert any(key in line.split()[1:] for line in common.splitlines() if line.startswith('export '))
    # Exercise guard before any GPU command or model import.
    env = tmp_path / 'hpg.env'
    env.write_text(f'PROJECT_ROOT={Path.cwd()}\n')
    out = tmp_path / 'existing'; out.mkdir()
    result = subprocess.run(['bash', '-c', 'source scripts/repair_stage_env.sh'],
        env={'PATH': '/usr/bin:/bin', 'HPG_CONFIG': str(env), 'REPAIR_OUT': str(out)}, capture_output=True, text=True)
    assert result.returncode != 0 and 'Output already exists' in result.stderr


def test_serving_stage_uses_existing_equivalence_gate_and_fresh_outputs():
    source = Path('scripts/repair_serve_stage.sh').read_text()
    assert source.count('tools/check_lora_serving.py') == 2
    assert '--hf-only' in source and '--hf-reference' in source
    assert 'harness/serve_student.py' in source and '--port 0' in source
    assert 'mkdir "$REPAIR_OUT"' in source
    assert 'repair_work "$endpoint"' in source
    assert not Path('scripts/hpg_repair_train_arm.slurm').exists()


def test_stage_preserves_caller_cache_exports_and_uses_this_worktree(tmp_path):
    root = Path.cwd()
    config = tmp_path / 'hpg.env'; config.write_text('# Caller supplies all paths\n')
    cache_keys = ['ALFWORLD_DATA', 'VLLM_CACHE_ROOT', 'TRITON_CACHE_DIR', 'HF_HOME',
                  'XDG_CACHE_HOME', 'SED_HASH_CACHE_DIR']
    caches = {key: str(tmp_path / key.lower()) for key in cache_keys}
    env = {'PATH': '/usr/bin:/bin', 'HPG_CONFIG': str(config), 'SLURM_SUBMIT_DIR': str(root),
           'REPAIR_OUT': str(tmp_path / 'new-run'), **caches}
    script = 'source scripts/repair_stage_env.sh\nprintf "%s\\n" "$PROJECT_ROOT" "$VLLM_BATCH_INVARIANT" ' + ' '.join('"$' + k + '"' for k in cache_keys)
    result = subprocess.run(['bash', '-c', script], env=env, capture_output=True, text=True, check=True)
    assert result.stdout.splitlines() == [str(root), '1', *caches.values()]
    del env['XDG_CACHE_HOME']
    result = subprocess.run(['bash', '-c', script], env=env, capture_output=True, text=True)
    assert result.returncode != 0 and 'Export OFF_HOME XDG_CACHE_HOME' in result.stderr
