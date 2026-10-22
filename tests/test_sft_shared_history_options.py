"""Offline launch commands, pipeline identity binding, and acceptance inputs."""
import importlib
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

from baselines.common import pipeline as common
from sed.storage import digest, write_json


@pytest.mark.parametrize('method', ['kang', 'adftp', 'smartad'])
@pytest.mark.parametrize('gpus', [1, 2, 4, 8])
@pytest.mark.parametrize('scoring', ['per_turn', 'shared_history'])
def test_pipeline_binds_launch_options(tmp_path, monkeypatch, method, gpus, scoring):
    pipeline = importlib.import_module(f'baselines.{method}.pipeline')
    received = []
    def prepare(args, config, *, method, recipe):
        identity = {'method': method, 'gpus': args.gpus, 'recipe': recipe, 'K': 5,
                    'unique_instances': 5, 'task_groups': 5}
        received.append((args, identity))
        return identity
    monkeypatch.setattr(pipeline, 'prepare', prepare)
    monkeypatch.setattr(pipeline, 'load_config', lambda _: {})
    monkeypatch.setattr(pipeline, 'update_disclosure', lambda *a: None)
    # Exercise command binding with fake inputs; full CPU preparation is tested
    # in test_training_prepare with real validation and a fake tokenizer.
    preparations = []
    monkeypatch.setattr(common, 'run_command', lambda command, log: preparations.append(command))
    if method == 'smartad':
        monkeypatch.setattr(pipeline, 'validate_h0', lambda *a: None)
        monkeypatch.setattr(pipeline, 'load_frozen', lambda *a: ({}, {}))
        monkeypatch.setattr(pipeline, 'load_config', lambda _: {'paths': {'manifest': '', 'split_freeze': ''}})
        monkeypatch.setattr(pipeline, 'validate_selection', lambda *a: None)
    args = ['--snapshot', 'BASE', '--verified', 'VERIFIED', '--output', str(tmp_path), '--mode', 'prepare',
            '--gpus', str(gpus), '--scoring', scoring]
    if method != 'kang':
        args += ['--through', 'train']
    pipeline.main(args)
    received_args, identity = received[0]
    assert identity['gpus'] == gpus and identity['recipe']['scoring'] == scoring
    assert identity['recipe']['dropout'] == .05
    assert identity['recipe']['dropout_equivalence'] == 'expected_gradient'
    assert identity['recipe']['rng_seed_policy'] == 'seed + rank'
    alternative = {**identity, 'recipe': {**identity['recipe'], 'scoring': 'per_turn' if scoring == 'shared_history' else 'shared_history'}}
    assert digest([identity, 'sft']) != digest([alternative, 'sft'])
    assert digest([identity, 'sft']) != digest([{**identity, 'gpus': 16}, 'sft'])
    command = pipeline.training_command(received_args, tmp_path / 'out')
    assert command[1:4] == ['-m', 'torch.distributed.run', '--standalone']
    assert command[command.index('--nproc_per_node') + 1] == str(gpus)
    assert command[command.index('--scoring') + 1] == scoring
    assert command[command.index('--objective') + 1] == ('smartad' if method == 'smartad' else 'sft')
    prepared, = preparations
    assert prepared[:3] == [received_args.train_python, '-m', 'baselines.common.fsdp_train']
    assert prepared[-1] == '--prepare-only'
    assert 'torch.distributed.run' not in prepared and '--nproc_per_node' not in prepared
    for flag in ('--scoring', '--objective'):
        assert prepared[prepared.index(flag) + 1] == command[command.index(flag) + 1]


@pytest.mark.parametrize('method', ['kang', 'adftp', 'smartad'])
def test_slurm_forwards_gpus_and_scoring(tmp_path, method):
    fake = tmp_path / 'python'
    fake.write_text('#!' + sys.executable + '\nimport json,os,sys\n'
                    'open(os.environ["ARGS"],"w").write(json.dumps(sys.argv[1:]))\n')
    fake.chmod(0o755)
    snapshot = tmp_path / 'base'
    snapshot.mkdir()
    env = dict(os.environ, PROJECT_ROOT=str(Path.cwd()), HPG_CONFIG=str(Path('configs/hpg.env').resolve()),
               ENV_PYTHON=str(fake), TRAIN_PYTHON=str(fake), VLLM_PYTHON=str(fake), MODEL_PATH=str(snapshot),
               VERIFIED='VERIFIED', ARGS=str(tmp_path / 'args.json'), NGPUS='4', SLURM_GPUS_ON_NODE='4',
               ENABLE_THINKING='off', MAX_TOKENS='1024', MAX_LENGTH='32768',
               **{method.upper() + '_OUT': str(tmp_path / 'out'), method.upper() + '_THROUGH': 'train',
                  method.upper() + '_SCORING': 'shared_history'})
    result = subprocess.run(['bash', f'scripts/hpg_{method}.slurm'], env=env, capture_output=True, text=True, timeout=15)
    assert result.returncode == 0, result.stderr
    args = json.loads((tmp_path / 'args.json').read_text())
    assert args[args.index('--gpus') + 1] == '4'
    assert args[args.index('--scoring') + 1] == 'shared_history'
    env['NGPUS'] = '3'
    result = subprocess.run(['bash', f'scripts/hpg_{method}.slurm'], env=env, capture_output=True, text=True, timeout=15)
    assert result.returncode == 2 and 'NGPUS must be' in result.stderr


@pytest.mark.parametrize('objective', ['sft', 'smartad'])
def test_acceptance_verified_loader_and_refusals(tmp_path, monkeypatch, objective):
    from tools import validate_shared_history_gpu as gate
    from test_sad import CharTokenizer, episode
    from baselines.common.conversation import conversation
    from baselines.smartad.data import encode_episode
    episodes = [episode('Thought: check\nAction: look') for _ in range(4)]
    collection = {'verified': list(range(4))}
    monkeypatch.setattr('sed.manifest.load_frozen', lambda *a: ({}, {}))
    monkeypatch.setattr('sed.seed_evidence.load_verified_collection', lambda *a: (collection, {}, tmp_path))
    monkeypatch.setattr('sed.seed_evidence.checked_episodes', lambda root, receipts: iter(episodes[i] for i in receipts))
    config = {'paths': {'manifest': '', 'split_freeze': ''}}
    rows, metadata = gate.load_sft_rows(tmp_path, config, CharTokenizer(), 4, objective)
    expected = (encode_episode(episodes[0], CharTokenizer()) if objective == 'smartad' else
                conversation(episodes[0], CharTokenizer(), source='trusted_teacher_seed', packing='turn'))
    assert rows == [expected] * 4 and metadata['encoded_rows_sha256'] == digest(rows)
    assert metadata['selected_indices'] == list(range(4))
    for n in (0, 5):
        with pytest.raises(ValueError, match='no silent undersampling'):
            gate.load_sft_rows(tmp_path, config, CharTokenizer(), n, objective)


def test_acceptance_slurm_runs_both_objectives_and_retains_failure(tmp_path):
    fake = tmp_path / 'python'
    fake.write_text('#!' + sys.executable + '\nimport json,os,sys\n'
        'with open(os.environ["ARGS"],"a") as f: f.write(json.dumps(sys.argv[1:])+"\\n")\n'
        'sys.exit(1 if "sft" in sys.argv else 0)\n')
    fake.chmod(0o755)
    env = dict(os.environ, PROJECT_ROOT=str(Path.cwd()), HPG_CONFIG=str(Path('configs/hpg.env').resolve()),
               TRAIN_PYTHON=str(fake), MODEL_PATH=str(tmp_path), VERIFIED=str(tmp_path),
               ARGS=str(tmp_path / 'args.jsonl'), OUTPUT_PREFIX=str(tmp_path / 'gpu'), SLURM_JOB_ID='test')
    result = subprocess.run(['bash', 'scripts/hpg_validate_sft_shared_history.slurm'], env=env,
                            capture_output=True, text=True, timeout=15)
    assert result.returncode == 1
    calls = [json.loads(line) for line in (tmp_path / 'args.jsonl').read_text().splitlines()]
    assert [c[c.index('--objective') + 1] for c in calls] == ['sft', 'smartad']
    assert all(c[c.index('--num-rows') + 1] == '4' for c in calls)


@pytest.mark.parametrize('objective', ['sft', 'smartad'])
def test_acceptance_cli_reports_missing_gpu(tmp_path, objective):
    from tools import validate_shared_history_gpu as gate
    output = tmp_path / 'rejected.json'
    assert gate.main(['--verified', str(tmp_path), '--objective', objective, '--output', str(output)]) == 1
    report = json.loads(output.read_text())
    assert report['settings']['objective'] == objective
    assert report['settings']['adaptation'] == 'kang_lora'
    assert report['settings']['lora_dropout'] == 0
    assert not report['gpu_acceptance'] and report['rows'] == []
    assert report['error_message'] == 'Exactly one visible GPU and one process required'
