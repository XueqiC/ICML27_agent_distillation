"""Offline launch/evaluation wiring: teacher stages cannot reserve GPU resources."""
from contextlib import contextmanager
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

from baselines.common import evaluation
from sed.storage import write_json
from sed.seed_evidence import artifact_hashes


@pytest.mark.parametrize('method,through,mode', [('adftp','collect','train'), ('adftp','train','train'),
                                               ('adftp','train','evaluate'), ('sad','train','train')])
def test_wrappers_select_stages(tmp_path, method, through, mode):
    fake = tmp_path / 'python'
    fake.write_text('#!' + sys.executable + '\nimport json,os,sys\n'
                    'open(os.environ["ARGS"],"w").write(json.dumps(sys.argv[1:]))\n')
    fake.chmod(0o755)
    env = os.environ.copy()
    for key in ('SLURM_JOB_GPUS', 'SLURM_GPUS_ON_NODE'):
        env.pop(key, None)
    env.update(PROJECT_ROOT=str(Path.cwd()), HPG_CONFIG=str(Path('configs/hpg.env').resolve()),
        ENV_PYTHON=str(fake), TRAIN_PYTHON=str(fake), VLLM_PYTHON=str(fake), MODEL_PATH='BASE', VERIFIED='VERIFIED',
        ARGS=str(tmp_path / 'args.json'), **{method.upper() + '_OUT':str(tmp_path/'out'),
        method.upper() + '_THROUGH':through, method.upper() + '_MODE':mode})
    result = subprocess.run(['bash', f'scripts/hpg_{method}.slurm'], env=env, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    args = json.loads((tmp_path / 'args.json').read_text())
    assert args[:2] == ['-m', f'baselines.{method}.pipeline']
    assert args[args.index('--through') + 1] == through
    if through == 'collect':
        assert '--snapshot' not in args
        env['SLURM_JOB_GPUS'] = '0'
        result = subprocess.run(['bash', 'scripts/hpg_adftp.slurm'], env=env, capture_output=True, text=True)
        assert result.returncode != 0 and 'must not hold a GPU' in result.stderr


@pytest.mark.parametrize('passing', [False, True])
def test_full_evaluation_requires_gate_before_dev140(tmp_path, monkeypatch, passing):
    checkpoint = tmp_path / 'checkpoint'
    write_json(checkpoint / 'weights.json', {})
    write_json(checkpoint / 'training_state.json', {'artifact_sha256': artifact_hashes(checkpoint)})
    write_json(tmp_path / 'models.json', [{'name':'sad-1of16', 'checkpoint':str(checkpoint)}])
    write_json(tmp_path / 'verified/manifest.json', {})
    args = SimpleNamespace(config='configs/base.yaml', snapshot='base', verified=str(tmp_path / 'verified'),
                           output=str(tmp_path), train_python='train', env_python='env', workers=1)
    calls = []
    @contextmanager
    def served(*args):
        yield 'http://127.0.0.1:1234/v1'
    monkeypatch.setattr(evaluation, 'served', served)
    def command(argv, log, **kwargs):
        calls.append(argv)
        if '--hf-reference' in argv and not passing:
            raise RuntimeError('stub gate rejection')
    monkeypatch.setattr(evaluation, 'run_command', command)
    if passing:
        evaluation.evaluate_models(args)
        assert calls[-1][1] == 'eval/run_suite.py'
        assert calls[-1][calls[-1].index('--tasks') + 1] == 'data/manifest/dev140.json'
    else:
        with pytest.raises(RuntimeError, match='gate rejection'):
            evaluation.evaluate_models(args)
        assert not any('eval/run_suite.py' in c for c in calls)
    assert '--hf-only' in calls[0] and '--hf-reference' in calls[1]
