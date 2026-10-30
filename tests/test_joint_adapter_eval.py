"""E0.6 uses fake weights, transports, students and environments only."""
from copy import deepcopy
from functools import partial
import json
from pathlib import Path
import shutil
from types import SimpleNamespace

import pytest

from repair.adapter_eval import (NoTeacherBudget, PREFIX_VERIFICATION, assert_no_teacher_calls,
                                 leakage_check, serving_spec, training_identity)
from repair.joint import execute_joint
from sed.config import load_config
from sed.storage import digest, file_hash, write_json
from tools import joint_guidance
from tools.cg_probe import serving_spec as theta0_serving_spec
from tools.teacher_takeover import main as takeover_main
from test_cg_probe import GROUPS, ScriptEnvironment, coverage_fixture, original
from test_joint import CaptureStudent, case
from test_lora_serving import adapter_bank, bound_report


@pytest.fixture
def trained_adapter(tmp_path, bound_report):
    report, server = deepcopy(bound_report)
    adapter = server['config']['server']['lora_adapters'][0]
    checkpoint = tmp_path / 'v2-run' / 'update-03' / 'checkpoint'
    shutil.copytree(adapter['path'], checkpoint)
    adapter.update(path=str(checkpoint), served_path=str(checkpoint))
    update = dict(optimizer_updates=3, update=3, sha256={
        'adapter_model.safetensors': adapter['adapter_weights_sha256'],
        'adapter_config.json': adapter['adapter_config_sha256']})
    run = dict(schema='v2-training-run/1', arm='OFFLINE', training_exposure=48,
               accounting={'optimizer_updates': 500}, input_manifests=[
                   dict(schema='v2-train-interchange/1', training_exposure_tasks=['train/unrelated']),
                   dict(schema='v2-train-interchange/1', training_exposure_tasks=['train/replay'])])
    write_json(checkpoint.parent / 'manifest.json', update)
    write_json(checkpoint.parent.parent / 'manifest.json', run)
    report['server_provenance_sha256'] = digest(server)
    server_path, report_path = tmp_path / 'server.json', tmp_path / 'check.json'
    write_json(server_path, server)
    write_json(report_path, report)
    args = SimpleNamespace(endpoint=server['config']['server']['endpoint'], model_name=adapter['name'],
                           server_provenance=str(server_path), lora_check=str(report_path),
                           dry_run=False, output=str(tmp_path / 'evaluation'))
    return args, adapter, run, update


def cli_spec(tmp_path, original):
    coverage, _ = coverage_fixture(tmp_path, original)
    spec = tmp_path / 'spec.json'
    assert takeover_main(['spec-from-cg', '--failures', str(coverage / 'failures.json'),
        '--coverage-root', str(coverage), '--groups', GROUPS[5], '--dry-run', '--output', str(spec)]) == 0
    return spec


def cli_args(spec, args):
    return ['run', '--spec', str(spec), '--config', 'configs/base.yaml', '--dry-run', '--eval-adapter',
            '--modes', 'U', '--seeds', '--endpoint', args.endpoint, '--model-name', args.model_name,
            '--server-provenance', args.server_provenance, '--lora-check', args.lora_check]


@pytest.mark.parametrize('modes', [['G'], ['G1'], ['U', 'G'], ['U', 'G1']])
def test_guided_eval_refused_before_inputs_or_output(tmp_path, modes):
    with pytest.raises(ValueError, match='requires --modes U'):
        joint_guidance.main(['run', '--eval-adapter', '--modes', *modes, '--spec', 'missing',
                             '--output', str(tmp_path / 'out')])
    assert not (tmp_path / 'out').exists()


def test_default_still_requires_budget(tmp_path, capsys):
    with pytest.raises(SystemExit) as error:
        joint_guidance.main(['run', '--modes', 'U', '--spec', 'missing', '--output', str(tmp_path / 'out')])
    assert error.value.code == 2 and '--budget-ledger' in capsys.readouterr().err


def test_adapter_gate_and_checkpoint_provenance(trained_adapter):
    args, adapter, _, _ = trained_adapter
    config = load_config()
    spec, identity, inputs = serving_spec(args, config)
    assert spec['model'] == 'sft' and not spec['enable_thinking']
    for key in ('name', 'path', 'adapter_weights_sha256', 'adapter_config_sha256'):
        assert identity[key] == adapter[key]
    training = identity['training']
    assert training['arm'] == 'OFFLINE' and training['optimizer_updates'] == 3
    assert training['manifest_sha256'] == file_hash(training['manifest_path'])
    assert training['checkpoint_manifest_sha256'] == file_hash(training['checkpoint_manifest_path'])
    assert training['training_exposure'] == dict(task_ids=[], group_ids=['train/replay', 'train/unrelated'])
    assert {training['manifest_path'], training['checkpoint_manifest_path']} <= set(inputs)
    # The original helper still rejects these weights/name even for the same base.
    with pytest.raises(ValueError, match='theta0'):
        theta0_serving_spec(args, load_config(), [])


@pytest.mark.parametrize('tamper', ['base', 'revision', 'server', 'report', 'weights', 'checkpoint'])
def test_eval_gate_fails_closed(trained_adapter, tamper):
    args, adapter, _, update = trained_adapter
    config = load_config()
    if tamper in ('base', 'revision'):
        config['models']['student']['requested_repo' if tamper == 'base' else 'revision'] = 'other'
    elif tamper == 'server':
        server = json.loads(Path(args.server_provenance).read_text())
        server['config']['server']['served_model_name'] = 'other'
        write_json(args.server_provenance, server)
    elif tamper == 'report':
        report = json.loads(Path(args.lora_check).read_text())
        report['passed'] = False
        write_json(args.lora_check, report)
    elif tamper == 'weights':
        (Path(adapter['path']) / 'adapter_model.safetensors').write_bytes(b'changed')
    else:
        update['sha256']['adapter_config.json'] = 'changed'
        write_json(Path(adapter['path']).parent / 'manifest.json', update)
    with pytest.raises(ValueError):
        serving_spec(args, config)


@pytest.mark.parametrize('exposure', [
    'task', 'alternate_trial', 'group', 'bare_group', 'bundle', 'replay_bundle', 'checkpoint'])
@pytest.mark.parametrize('from_start', [False, True])
def test_leakage_refused_before_creating_run(trained_adapter, original, tmp_path, exposure, from_start):
    args, adapter, run, update = trained_adapter
    spec = cli_spec(tmp_path, original)
    group = str(Path(original.task_id).parent)
    if exposure == 'task':
        run['training_exposure'] = dict(task_ids=[original.task_id], group_ids=[])
    elif exposure == 'alternate_trial':
        run['training_exposure'] = dict(tasks=[group + '/trial_other'])
    elif exposure in ('group', 'bare_group'):
        run['training_exposure'] = dict(groups=[group if exposure == 'group' else GROUPS[5]])
    elif exposure == 'checkpoint':
        update['training_exposure'] = dict(group_ids=[group])
    else:
        run['input_manifests'][int(exposure == 'replay_bundle')]['training_exposure_tasks'] = [group]
    write_json(Path(adapter['path']).parent.parent / 'manifest.json', run)
    write_json(Path(adapter['path']).parent / 'manifest.json', update)
    with pytest.raises(ValueError, match='overlaps adapter training exposure'):
        joint_guidance.main([*cli_args(spec, args), '--output', args.output,
                             *(['--from-start'] if from_start else [])])
    assert not Path(args.output).exists()


@pytest.mark.parametrize('ledger_kind', ['omitted', 'absent', 'existing'])
def test_cli_identity_zero_teacher_resume_and_summary(trained_adapter, original, tmp_path, monkeypatch, ledger_kind):
    args, adapter, _, _ = trained_adapter
    spec = cli_spec(tmp_path, original)
    def forbidden(*args, **kwargs):
        pytest.fail('Teacher budget/reviewer must never be constructed')
    monkeypatch.setattr(joint_guidance, 'SharedTeacherBudget', forbidden)
    monkeypatch.setattr(joint_guidance, 'make_reviewer', forbidden)
    ledger = tmp_path / 'external' / 'budget.jsonl'
    if ledger_kind == 'existing':
        ledger.parent.mkdir()
        ledger.write_bytes(b'External budget must not even be parsed.\n')
    common = [*cli_args(spec, args), '--output', args.output]
    if ledger_kind != 'omitted':
        common += ['--budget-ledger', str(ledger), '--max-teacher-input-tokens', '7',
                   '--max-teacher-output-tokens', '9']
    assert joint_guidance.main(common) == 0
    output = Path(args.output)
    plan = json.loads((output / 'plan.json').read_text())
    identity = plan['identity']
    assert identity['eval_adapter'] and identity['teacher_calls_allowed'] == 0
    assert identity['leakage_check']['status'] == 'passed'
    assert identity['leakage_check']['overlapping_group_ids'] == []
    assert identity['prefix_verification'] == PREFIX_VERIFICATION
    if ledger_kind != 'omitted':
        assert identity['caps'] == dict(input_tokens=7, output_tokens=9)
    records = {p.name: p.read_bytes() for p in (output / 'records').glob('*.json')}
    assert len(records) == 1
    row = json.loads(next(iter(records.values())))
    assert row['evaluated_adapter'] == identity['evaluated_adapter']
    assert row['prefix_verified'] and row['training_exposure'] == 0
    assert row['teacher_usage']['calls'] == [] and row['metrics']['review_calls'] == 0
    assert joint_guidance.main([*common, '--resume']) == 0
    assert records == {p.name: p.read_bytes() for p in (output / 'records').glob('*.json')}
    summary_path = tmp_path / 'summary'
    assert joint_guidance.main(['summarize', '--rollouts', str(output), '--output', str(summary_path)]) == 0
    summary = json.loads((summary_path / 'summary.json').read_text())
    assert summary['global_budget']['reserved_attempts'] == 0
    assert summary['global_budget']['caps'] == identity['caps']
    assert summary['all_attempt_usage'] == dict(input_tokens=0, output_tokens=0, reasoning_tokens=0)
    assert not summary['missing_records'] and summary['unknown_usage_attempts'] == 0
    assert not list(output.glob('calls/*/*/usage.jsonl'))
    if ledger_kind == 'existing':
        assert ledger.read_bytes() == b'External budget must not even be parsed.\n'
        assert list(ledger.parent.iterdir()) == [ledger]
    else:
        assert not ledger.parent.exists()
    # Training metadata is part of resume identity, not merely descriptive text.
    manifest = Path(adapter['path']).parent.parent / 'manifest.json'
    updated = json.loads(manifest.read_text())
    updated['arm'] = 'JOINT'
    write_json(manifest, updated)
    with pytest.raises(ValueError, match='Resume requires identical'):
        joint_guidance.main([*common, '--resume'])


@pytest.mark.parametrize('from_start', [False, True])
@pytest.mark.parametrize('mismatch', [False, True])
def test_prefix_replay_never_regenerates_theta0(trained_adapter, original, tmp_path, from_start, mismatch):
    _, adapter, _, _ = trained_adapter
    training, _ = training_identity(adapter)
    identity = dict(adapter, training=training)
    student = CaptureStudent(['Thought: A different policy.\nAction: take apple 1 from cabinet 2'])
    row = execute_joint(case(original), original, mode='U', decoding='greedy', seed=None,
        env_factory=partial(ScriptEnvironment, mismatch=mismatch), student_factory=lambda _: student,
        reviewer_factory=lambda _: pytest.fail('Teacher constructed'), output=tmp_path / 'execution',
        run_sha256='unit', budget=NoTeacherBudget(), evaluated_adapter=identity, from_start=from_start)
    if mismatch and not from_start:
        assert row['void'] and not row['prefix_verified'] and not student.prompts
    else:
        assert row['prefix_verified'] and row['success'] and not row['void']
        n = 0 if from_start else 10
        assert len(student.prompts) == 1  # Only the first suffix decision is generated.
        assert len(student.prompts[0]) == 2 + 2 * n
        assert row['execution']['steps'][:n] == original.to_json()['steps'][:n]
        assert row['steps'][0]['proposal_reply'] != original.steps[n]['action_raw']
        assert 'manifest_sha256' not in json.dumps(student.prompts)
    assert row['evaluated_adapter'] == identity
    assert_no_teacher_calls([row])
    corrupted = deepcopy(row)
    corrupted['teacher_usage']['interrupted_requests'] = 1
    with pytest.raises(AssertionError, match='Teacher usage'):
        assert_no_teacher_calls([corrupted])


def test_synthetic_eval_pool_and_from_start(original, tmp_path):
    spec = cli_spec(tmp_path, original)
    output = tmp_path / 'out'
    assert joint_guidance.main(['run', '--spec', str(spec), '--eval-adapter', '--modes', 'U',
        '--dry-run', '--from-start', '--seeds', '0', '--workers', '2', '--output', str(output)]) == 0
    rows = [json.loads(p.read_text()) for p in (output / 'records').glob('*.json')]
    assert len(rows) == 2 and all(r['branch_index'] == 0 for r in rows)
    assert all(r['evaluated_adapter']['synthetic'] and r['prefix_verified'] for r in rows)


def test_non_v2_adapter_discloses_missing_exposure_metadata(bound_report, tmp_path, original):
    _, server = bound_report
    adapter = server['config']['server']['lora_adapters'][0]
    assert training_identity(adapter) == (None, [])
    report = leakage_check([dict(spec=case(original))], dict(training=None))
    assert report['status'] == 'no_exposure_ids_declared'


def test_missing_v2_manifest_is_not_silently_ignored(trained_adapter):
    _, adapter, _, _ = trained_adapter
    (Path(adapter['path']).parent.parent / 'manifest.json').unlink()
    with pytest.raises(FileNotFoundError):
        training_identity(adapter)
