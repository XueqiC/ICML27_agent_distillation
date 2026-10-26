from pathlib import Path
import json
import subprocess
import sys

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


@pytest.mark.parametrize('stage,branch,checkpoint', [
    ('coverage', 'P0_K20', 0), ('coverage', 'P1_U_mech', 0), ('coverage', 'P0_K0', 0),
    ('probes', 'P0_K0', 0), ('probes', 'P0_K0', 20), ('probes', 'P1_U_mech', 0)])
def test_rai_launchers_without_slurm_run_complete_fake_stage(tmp_path, stage, branch, checkpoint):
    from repair.artifacts import write_sealed
    from repair.branches import branch_fields
    root = Path.cwd()
    fake = tmp_path / 'fake-python'
    fake.write_text(f'#!{sys.executable}\n' + '''
import json, os, pathlib, signal, sys
a = sys.argv[1:]
if a[0] in ('-c', '-'):
    os.execv(sys.executable, [sys.executable, *a])
with open(os.environ['FAKE_CALLS'], 'a') as f:
    f.write(json.dumps({'args': a, 'cuda': os.environ.get('CUDA_VISIBLE_DEVICES'),
        'batch': os.environ.get('VLLM_BATCH_INVARIANT'), 'cwd': os.getcwd()}) + '\\n')
output = pathlib.Path(a[a.index('--output') + 1]); output.mkdir(parents=True)
if a[0] == 'harness/serve_student.py':
    (output / 'provenance.json').write_text('{}')
    (output / 'ready.json').write_text(json.dumps({'endpoint': 'http://127.0.0.1:1234/v1'}))
    signal.signal(signal.SIGTERM, lambda *a: sys.exit(0))
    signal.pause()
elif a[0] == 'tools/check_lora_serving.py':
    (output / ('reference.json' if '--hf-only' in a else 'report.json')).write_text('{}')
''')
    fake.chmod(0o755)
    cfg = tmp_path / 'rai.env'
    cfg.write_text('\n'.join(f'{key}={fake}' for key in ('ENV_PYTHON', 'TRAIN_PYTHON', 'VLLM_PYTHON')) + '\n')
    collection = tmp_path / 'collection.json'
    write_sealed(collection, branch_fields(branch))
    env = {'PATH': '/usr/bin:/bin', 'CUDA_VISIBLE_DEVICES': '', 'HPG_CONFIG': str(cfg),
           'REPAIR_OUT': str(tmp_path / 'stage'), 'MODEL_PATH': str(tmp_path / 'snapshot'),
           'FAKE_CALLS': str(tmp_path / 'calls.jsonl'), 'REPAIR_BRANCH': branch,
           'TASKS': str(tmp_path / 'selection.json'), 'THETA0': str(tmp_path / 'theta0'),
           'REPAIR_COLLECTION': str(collection), 'ARM': 'shared' if checkpoint == 0 else 'N',
           'CHECKPOINT': str(checkpoint), 'TRAINING_MANIFEST': str(tmp_path / 'training.json'),
           'ADAPTER': str(tmp_path / 'adapter'), 'MODEL_NAME': 'e4b-tsft-k20-e32-lr5e-5',
           'VERIFIED': str(tmp_path / 'verified'), 'READY_TIMEOUT': '5'}
    for key in ('ALFWORLD_DATA', 'VLLM_CACHE_ROOT', 'TRITON_CACHE_DIR', 'HF_HOME',
                'XDG_CACHE_HOME', 'SED_HASH_CACHE_DIR'):
        env[key] = str(tmp_path / key.lower())
    base = branch == 'P0_K0' and (stage == 'coverage' or checkpoint == 0)
    if base:
        for key in ('ADAPTER', 'THETA0', 'VERIFIED', 'MODEL_NAME'): env.pop(key)
    result = subprocess.run(['bash', str(root / f'scripts/hpg_repair_{stage}.slurm')],
        cwd=tmp_path, env=env, capture_output=True, text=True, timeout=20)
    assert result.returncode == 0, result.stderr
    calls = [json.loads(line) for line in (tmp_path / 'calls.jsonl').read_text().splitlines()]
    assert all(c['cuda'] == '' and c['batch'] == '1' and c['cwd'] == str(root) for c in calls)
    server = next(c['args'] for c in calls if c['args'][0] == 'harness/serve_student.py')
    driver = next(c['args'] for c in calls if c['args'][:2] == ['-m', f'tools.repair_{stage}'])
    assert ('--lora-adapter' in server) is not base
    assert ('--lora-check' in driver) is not base
    assert sum(c['args'][0] == 'tools/check_lora_serving.py' for c in calls) == (0 if base else 2)
    if branch == 'P1_U_mech':
        assert all('--repair-support-theta0' in c['args'] for c in calls
                   if c['args'][0] == 'tools/check_lora_serving.py')
    if base: assert driver[driver.index('--model-name') + 1] == 'google/gemma-4-E4B-it'
    if branch == 'P1_U_mech' and stage == 'coverage': assert '--task-set' in driver and 'U_mech' in driver
