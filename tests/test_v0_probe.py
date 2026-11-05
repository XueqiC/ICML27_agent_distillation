import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

from repair.v0_common import THETA0_CELLS, completed, read
from sed.storage import file_hash, write_json
from tools import v0_round as cli
from test_adapter_eval_v0 import adapter_at, save_run, training_identity, v0_run
from test_v0_pipeline import arguments, fake_adapter, fake_runtime
from test_v0_round import manifest_at, task_row


def contents(root):
    # Include locks: an isolated probe must not even create round.lock.
    return {str(p.relative_to(root)): p.read_bytes() for p in root.rglob('*') if p.is_file()}


def theta0_adapter(tmp_path, k):
    path = tmp_path / THETA0_CELLS[k] / 'checkpoint'
    path.mkdir(parents=True)
    write_json(path / 'training_state.json', dict(N_examples=k, final=True, seed=0,
                                                 epochs=64 if k == 30 else 32))
    (path / 'adapter_model.safetensors').write_bytes(b'fake weights, no model loaded')
    write_json(path / 'adapter_config.json', {})
    return dict(fake_adapter(k), **adapter_at(path))


def trained_adapter(v0_run, k):
    adapter, run, update, _ = v0_run
    run['K_trusted_seed_trajectories'] = k
    run['input_manifest'].update(K=k, theta0=dict(
        adapter_path=f'/recorded/theta0/{THETA0_CELLS[k]}/checkpoint', adapter_weights_sha256='a' * 64))
    save_run(adapter, run, update)
    return dict(fake_adapter(k), **adapter)


def wrapper_precheck(monkeypatch, root, seeds, adapter, k, stage='probe', isolated=True):
    # Execute the actual Python block only: no mkdir, server or environment setup.
    source = Path('tools/v0_round_rai.sh').read_text().split("<<'PY'\n", 1)[1].split('\nPY\n', 1)[0]
    monkeypatch.setenv('V0_STAGE', stage)
    monkeypatch.setenv('V0_ATTEMPT_DIR', str(root.parent / 'probe-out') if isolated else '')
    monkeypatch.setattr(sys, 'argv', ['-', str(root), str(Path(seeds).parent),
                                     THETA0_CELLS[k], adapter['path']])
    exec(compile(source, 'tools/v0_round_rai.sh:precheck', 'exec'), {})


def precheck_round(tmp_path, k):
    root = tmp_path / 'round'
    args = arguments(root, k)
    cli.round_root(args)
    manifest = manifest_at(root / 'synth', [task_row(probe=True)], k)
    seeds = tmp_path / 'verified/manifest.json'
    write_json(seeds, read(args.seeds))
    write_json(manifest.path, dict(manifest.value, source_seed_manifest_sha256=file_hash(seeds)))
    return root, seeds


@pytest.mark.parametrize('k', [10, 20, 30])
@pytest.mark.parametrize('trained', [False, True])
def test_probe_attempt_dir_isolates_all_outputs_and_binds_adapter(tmp_path, monkeypatch, k, trained, v0_run):
    root, out = tmp_path / 'round', tmp_path / 'other-adapter-probe'
    cli.round_root(arguments(root, k))
    manifest = manifest_at(root / 'synth', [task_row(probe=True)], k)
    existing = root / 'probe_attempt'
    existing.mkdir()
    (existing / 'sentinel').write_bytes(b'original probe state')
    original_serving = cli.serving
    fake_runtime(monkeypatch, manifest)
    monkeypatch.setattr(cli, 'serving', original_serving)
    adapter = trained_adapter(v0_run, k) if trained else theta0_adapter(tmp_path, k)
    adapter_dir = Path(adapter['path'])
    weights = adapter_dir / 'adapter_model.safetensors'
    training, inputs = training_identity(adapter)
    adapter['training'] = training
    def serving_spec(args, config):
        assert Path(args.output) == out  # Including any serving-validation logs.
        return {}, adapter, list(inputs)
    monkeypatch.setattr('repair.adapter_eval.serving_spec', serving_spec)
    before = contents(root)
    cli.main(['probe', '--output', str(root), '--attempt-dir', str(out),
              '--model-name', THETA0_CELLS[k]])
    assert contents(root) == before
    assert completed(out)
    assert len(read(out / 'records.json')) == 4
    assert list((out / 'episodes').glob('*/record.json'))
    for value in (read(out / 'plan.json')['identity'], read(out / 'summary.json'), read(out / 'completion.json')):
        assert value['adapter']['path'] == str(adapter_dir)
        assert value['adapter']['adapter_weights_sha256'] == file_hash(weights)
        assert value['adapter']['training'] == training
    if trained:
        assert training['arm'] == 'v0:' + v0_run[1]['dataset_sha256']
        assert training['optimizer_updates'] == 3
        assert set(inputs) <= read(out / 'plan.json')['identity']['inputs'].keys()
    assert str(weights) in read(out / 'plan.json')['identity']['inputs']
    after = contents(out)
    with pytest.raises(FileExistsError):
        cli.main(['probe', '--output', str(root), '--attempt-dir', str(out)])
    assert contents(root) == before and contents(out) == after


@pytest.mark.parametrize('k', [10, 30])
@pytest.mark.parametrize('trained', [False, True])
def test_isolated_probe_wrapper_accepts_matching_adapter(tmp_path, monkeypatch, k, trained, v0_run):
    root, seeds = precheck_round(tmp_path, k)
    adapter = trained_adapter(v0_run, k) if trained else theta0_adapter(tmp_path, k)
    before = contents(tmp_path)
    wrapper_precheck(monkeypatch, root, seeds, adapter, k)
    assert contents(tmp_path) == before


@pytest.mark.parametrize('k', [10, 30])
@pytest.mark.parametrize('tamper', ['theta0', 'source_k', 'run_k', 'weights', 'schema', 'unknown_path'])
def test_isolated_probe_refuses_mismatched_training_adapter(tmp_path, monkeypatch, k, tamper, v0_run):
    root, seeds = precheck_round(tmp_path, k)
    adapter = trained_adapter(v0_run, k)
    _, run, update, _ = v0_run
    other_k = 30 if k == 10 else 10
    if tamper == 'theta0':
        run['input_manifest']['theta0']['adapter_path'] = f'/other/{THETA0_CELLS[other_k]}/checkpoint'
    elif tamper == 'source_k':
        run['input_manifest']['K'] = other_k
    elif tamper == 'run_k':
        run['K_trusted_seed_trajectories'] = other_k
    elif tamper == 'weights':
        (Path(adapter['path']) / 'adapter_model.safetensors').write_bytes(b'changed weights')
    elif tamper == 'schema':
        run.update(schema='v2-training-run/1', arm='CONTINUE')
    else:
        unknown = tmp_path / 'unrecognized-checkpoint'
        shutil.copytree(adapter['path'], unknown)
        adapter['path'] = str(unknown)
    save_run(v0_run[0], run, update)
    with pytest.raises(ValueError):
        wrapper_precheck(monkeypatch, root, seeds, adapter, k)
    monkeypatch.setattr('repair.adapter_eval.serving_spec', lambda *args: ({}, adapter, []))
    args = arguments(root, k, K=k, command='probe', attempt_dir=tmp_path / 'probe-out')
    with pytest.raises(ValueError):
        cli.serving(args, cli.config_for(args))


@pytest.mark.parametrize('k', [10, 30])
@pytest.mark.parametrize('stage,isolated', [('attempt', False), ('review', False), ('expand', False),
                                         ('expand-synth', False), ('probe', False), ('attempt', True)])
def test_nonisolated_stages_keep_theta0_only(tmp_path, monkeypatch, k, stage, isolated, v0_run):
    root, seeds = precheck_round(tmp_path, k)
    adapter = trained_adapter(v0_run, k)
    with pytest.raises(ValueError, match='requires MODEL_NAME and ADAPTER'):
        wrapper_precheck(monkeypatch, root, seeds, adapter, k, stage, isolated)
    monkeypatch.setattr('repair.adapter_eval.serving_spec', lambda *args: ({}, adapter, []))
    args = arguments(root, k, K=k, command=stage, attempt_dir=tmp_path / 'out' if isolated else None)
    with pytest.raises(ValueError, match='requires MODEL_NAME and ADAPTER'):
        cli.serving(args, cli.config_for(args))


@pytest.mark.parametrize('kind', ['directory', 'file', 'dangling_symlink'])
def test_probe_attempt_dir_refuses_existing_before_reading_round(tmp_path, kind):
    out = tmp_path / 'exists'
    if kind == 'directory':
        out.mkdir()
    elif kind == 'file':
        out.touch()
    else:
        out.symlink_to(tmp_path / 'missing')
    with pytest.raises(FileExistsError):
        cli.main(['probe', '--output', str(tmp_path / 'nonexistent-round'), '--attempt-dir', str(out)])


@pytest.mark.parametrize('flag', ['--resume', '--report-only'])
def test_probe_attempt_dir_cannot_reuse_directory(tmp_path, flag):
    with pytest.raises(SystemExit):
        cli.main(['probe', '--output', str(tmp_path / 'round'), '--attempt-dir', str(tmp_path / 'out'), flag])
    assert not (tmp_path / 'out').exists()


def test_probe_default_path_unchanged(tmp_path, monkeypatch):
    root = tmp_path / 'round'
    cli.round_root(arguments(root))
    manifest = manifest_at(root / 'synth', [task_row(probe=True)])
    fake_runtime(monkeypatch, manifest)
    cli.main(['probe', '--output', str(root)])
    assert completed(root / 'probe_attempt')


@pytest.mark.parametrize('k', [10, 20, 30])
def test_rai_launcher_places_serving_and_probe_under_exclusive_bundle(tmp_path, k):
    project, out, root = tmp_path / 'project', tmp_path / 'probe-bundle', tmp_path / 'round'
    (project / 'tools').mkdir(parents=True)
    (project / 'scripts').mkdir()
    root.mkdir()
    launcher = project / 'tools/v0_round_rai.sh'
    shutil.copyfile('tools/v0_round_rai.sh', launcher)
    # Only exercise shell wiring; these stubs launch no process/model/server.
    # The preflight's Python validation is covered separately; stub it here.
    (project / 'scripts/repair_stage_env.sh').write_text('CONFIG=fake\nENV_PYTHON=true\n')
    (project / 'scripts/repair_serve_stage.sh').write_text('''mkdir -p "$REPAIR_OUT"
printf '%s' "$CUDA_VISIBLE_DEVICES" > "$REPAIR_OUT/gpu"
printf '%s' "$VERIFIED" > "$REPAIR_OUT/verified"
run_work() { printf '%s\\n' "$@" > "$REPAIR_OUT/args"; }
repair_work fake-endpoint fake-provenance fake-check
''')
    env = dict(os.environ, V0_STAGE='probe', V0_ATTEMPT_DIR=str(out), V0_ROUND_OUT=str(root),
               ADAPTER='fake-adapter', GPU='3', V0_RESUME='0')
    env.pop('MODEL_NAME', None)
    env.pop('VERIFIED', None)
    if k != 20:
        env['MODEL_NAME'] = THETA0_CELLS[k]
    subprocess.run(['bash', str(launcher)], env=env, check=True, capture_output=True)
    assert not list(root.iterdir())
    assert (out / 'serve/gpu').read_text() == '3'
    args = (out / 'serve/args').read_text().splitlines()
    assert args[args.index('--attempt-dir') + 1] == str(out / 'probe')
    assert args[args.index('--output') + 1] == str(root)
    assert args[args.index('--model-name') + 1] == THETA0_CELLS[k]
    assert (out / 'serve/verified').read_text().endswith(f'/subsets/k{k}')
    before = contents(out)
    assert subprocess.run(['bash', str(launcher)], env=env, capture_output=True).returncode != 0
    assert contents(out) == before and not list(root.iterdir())


@pytest.mark.parametrize('k', [10, 30])
@pytest.mark.parametrize('variant', ['D-data', 'D-param'])
@pytest.mark.parametrize('stage', ['expand', 'expand-synth'])
@pytest.mark.parametrize('resume', ['0', '1'])
def test_rai_expansion_passes_k_seed_view_and_local_tokenizer(tmp_path, k, variant, stage, resume):
    project = tmp_path / 'project'
    (project / 'tools').mkdir(parents=True)
    (project / 'scripts').mkdir()
    launcher = project / 'tools/v0_round_rai.sh'
    shutil.copyfile('tools/v0_round_rai.sh', launcher)
    (project / 'scripts/repair_stage_env.sh').write_text('''CONFIG=fake
ENV_PYTHON=fake_python
ALFWORLD_DATA_ROOT=fake-games
fake_python() {
    if [[ "$1" == -m ]]; then
        printf '%s\\n' "$@" > "$V0_ROUND_OUT/args"
        printf '%s' "$CUDA_VISIBLE_DEVICES" > "$V0_ROUND_OUT/cuda"
    fi
}
''')
    (project / 'scripts/repair_serve_stage.sh').write_text('''mkdir -p "$REPAIR_OUT"
run_work() { printf '%s\\n' "$@" > "$REPAIR_OUT/args"; }
repair_work fake-endpoint fake-provenance fake-check
''')
    root = tmp_path / 'round'
    root.mkdir()
    env = dict(PATH=os.defpath, V0_STAGE=stage, V0_RESUME=resume, V0_ROUND_OUT=str(root),
        V0_SOURCE_ROOT=str(tmp_path / 'source'), ADAPTER='fake', MODEL_NAME=THETA0_CELLS[k],
        V0_VARIANT=variant, V0_DATASET='fake-dataset', V0_GROUPS='fake-groups', V0_SNAPSHOT='fake-snapshot')
    subprocess.run(['bash', str(launcher)], env=env, check=True, capture_output=True)
    if stage == 'expand-synth':
        assert not list(root.glob('serve-*'))  # Sourcing the server stub creates this directory.
        assert (root / 'cuda').read_text() == ''
        args = (root / 'args').read_text().splitlines()
        assert args[:4] == ['-m', 'tools.v0_round', 'expand', '--synth-only']
        assert not {'--endpoint', '--server-provenance', '--lora-check'}.intersection(args)
    else:
        args = (root / ('serve-expand-' + variant) / 'args').read_text().splitlines()
        assert '--synth-only' not in args
    assert ('--resume' in args) == (resume == '1')
    assert args[args.index('--variant') + 1] == variant
    assert args[args.index('--dataset') + 1] == 'fake-dataset'
    assert args[args.index('--data-root') + 1] == 'fake-games'
    assert args[args.index('--main-checkout') + 1] == str(tmp_path / 'source')
    assert args[args.index('--config') + 1] == 'fake'
    assert args[args.index('--output') + 1] == str(root)
    assert args[args.index('--seed') + 1] == '0'
    assert args[args.index('--seeds') + 1].endswith(f'/subsets/k{k}/manifest.json')
    assert args[args.index('--model-name') + 1] == THETA0_CELLS[k]
    assert args[args.index('--snapshot') + 1] == 'fake-snapshot'
    assert ('--groups' in args) == (variant == 'D-data')
