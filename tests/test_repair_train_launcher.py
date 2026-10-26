"""SLURM argument/cache/fresh-output guards, executed with a fake interpreter."""
import json
import os
from pathlib import Path
import subprocess
import sys


def test_two_gpu_launch_preserves_caller_caches_and_fresh_output(tmp_path):
    fake = tmp_path / 'python'
    caches = ['ALFWORLD_DATA', 'VLLM_CACHE_ROOT', 'TRITON_CACHE_DIR', 'HF_HOME', 'XDG_CACHE_HOME', 'SED_HASH_CACHE_DIR']
    fake.write_text('#!' + sys.executable + '\nimport json,os,sys\n'
        'with open(os.environ["ARGUMENTS"], "w") as f:\n'
        ' json.dump({"argv":sys.argv[1:], "caches":{k:os.environ[k] for k in ' + repr(caches) + '}}, f)\n')
    fake.chmod(0o755)
    env = dict(os.environ, PROJECT_ROOT=str(Path.cwd()), TRAIN_PYTHON=str(fake), MODEL_PATH=str(tmp_path),
        CONFIG='configs/base_e4b.yaml', THETA0='theta0', REPAIR_COLLECTION='e2_training_collection.json',
        ARM='N', REPAIR_OUT=str(tmp_path / 'new'), ARGUMENTS=str(tmp_path / 'args.json'),
        ENABLE_THINKING='off', MAX_TOKENS='1024', MAX_LENGTH='32768', BATCH_INVARIANT='1',
        **{key: str(tmp_path / key) for key in caches})
    script = 'scripts/hpg_repair_train_arm.slurm'
    result = subprocess.run(['bash', script], env=env, capture_output=True, timeout=15)
    assert result.returncode == 0, result.stderr
    saved = json.loads((tmp_path / 'args.json').read_text())
    assert saved['caches'] == {key: env[key] for key in caches}
    assert saved['argv'][:5] == ['-m', 'torch.distributed.run', '--standalone', '--nproc_per_node', '2']
    assert saved['argv'][saved['argv'].index('--collection') + 1] == 'e2_training_collection.json'
    assert '#SBATCH --mem=64G' in Path(script).read_text()
    (tmp_path / 'new').symlink_to(tmp_path / 'absent')
    assert subprocess.run(['bash', script], env=env, capture_output=True, timeout=15).returncode == 2
    (tmp_path / 'new').unlink()
    del env['HF_HOME']
    assert subprocess.run(['bash', script], env=env, capture_output=True, timeout=15).returncode != 0
