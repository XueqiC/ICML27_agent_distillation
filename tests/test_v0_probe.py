import os
from pathlib import Path
import shutil
import subprocess

import pytest

from repair.v0_common import THETA0_CELLS, completed, read
from sed.storage import file_hash, write_json
from tools import v0_round as cli
from test_v0_pipeline import arguments, fake_adapter, fake_runtime
from test_v0_round import manifest_at, task_row


def contents(root):
    # Include locks: an isolated probe must not even create round.lock.
    return {str(p.relative_to(root)): p.read_bytes() for p in root.rglob('*') if p.is_file()}


@pytest.mark.parametrize('k', [10, 20, 30])
def test_probe_attempt_dir_isolates_all_outputs_and_binds_adapter(tmp_path, monkeypatch, k):
    root, out = tmp_path / 'round', tmp_path / 'other-adapter-probe'
    cli.round_root(arguments(root, k))
    manifest = manifest_at(root / 'synth', [task_row(probe=True)], k)
    existing = root / 'probe_attempt'
    existing.mkdir()
    (existing / 'sentinel').write_bytes(b'original probe state')
    original_serving = cli.serving
    fake_runtime(monkeypatch, manifest)
    monkeypatch.setattr(cli, 'serving', original_serving)
    adapter_dir = tmp_path / THETA0_CELLS[k]
    adapter_dir.mkdir()
    write_json(adapter_dir / 'training_state.json', dict(N_examples=k, final=True, seed=0,
                                                       epochs=64 if k == 30 else 32))
    weights = adapter_dir / 'adapter_model.safetensors'
    weights.write_bytes(b'fake weights, no model loaded')
    adapter = dict(fake_adapter(k), path=str(adapter_dir), adapter_weights_sha256=file_hash(weights))
    def serving_spec(args, config):
        assert Path(args.output) == out  # Including any serving-validation logs.
        return {}, adapter, []
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
    assert str(weights) in read(out / 'plan.json')['identity']['inputs']
    after = contents(out)
    with pytest.raises(FileExistsError):
        cli.main(['probe', '--output', str(root), '--attempt-dir', str(out)])
    assert contents(root) == before and contents(out) == after


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
def test_rai_expansion_passes_k_seed_view_and_local_tokenizer(tmp_path, k, variant):
    project = tmp_path / 'project'
    (project / 'tools').mkdir(parents=True)
    (project / 'scripts').mkdir()
    launcher = project / 'tools/v0_round_rai.sh'
    shutil.copyfile('tools/v0_round_rai.sh', launcher)
    (project / 'scripts/repair_stage_env.sh').write_text('CONFIG=fake\nENV_PYTHON=true\nALFWORLD_DATA_ROOT=fake-games\n')
    (project / 'scripts/repair_serve_stage.sh').write_text('''mkdir -p "$REPAIR_OUT"
run_work() { printf '%s\\n' "$@" > "$REPAIR_OUT/args"; }
repair_work fake-endpoint fake-provenance fake-check
''')
    root = tmp_path / 'round'
    env = dict(PATH=os.defpath, V0_STAGE='expand', V0_ROUND_OUT=str(root),
        V0_SOURCE_ROOT=str(tmp_path / 'source'), ADAPTER='fake', MODEL_NAME=THETA0_CELLS[k],
        V0_VARIANT=variant, V0_DATASET='fake-dataset', V0_GROUPS='fake-groups', V0_SNAPSHOT='fake-snapshot')
    subprocess.run(['bash', str(launcher)], env=env, check=True, capture_output=True)
    args = (root / ('serve-expand-' + variant) / 'args').read_text().splitlines()
    assert args[args.index('--seeds') + 1].endswith(f'/subsets/k{k}/manifest.json')
    assert args[args.index('--model-name') + 1] == THETA0_CELLS[k]
    assert args[args.index('--snapshot') + 1] == 'fake-snapshot'
    assert ('--groups' in args) == (variant == 'D-data')
