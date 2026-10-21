"""Offline SAD evaluation selection, provenance, and checkpoint-stage reuse."""
from contextlib import contextmanager
import json
from pathlib import Path

import pytest

from baselines.common import evaluation
from baselines.sad import pipeline
from sed.seed_evidence import artifact_hashes
from sed.storage import file_hash, write_json


@pytest.fixture
def sad_evaluation(tmp_path, monkeypatch):
    models = []
    for denominator in (16, 8, 4, 2, 1):
        checkpoint = tmp_path / f'sad-1of{denominator}'
        write_json(checkpoint / 'weights.json', {})
        write_json(checkpoint / 'training_state.json', {'artifact_sha256': artifact_hashes(checkpoint)})
        models.append({'name': checkpoint.name, 'checkpoint': str(checkpoint), 'fraction': f'1/{denominator}'})
    write_json(tmp_path / 'models.json', models)
    write_json(tmp_path / 'verified/manifest.json', {})
    monkeypatch.delenv('SAD_EVAL_FRACTIONS', raising=False)
    served_models, calls = [], []

    @contextmanager
    def served(args, checkpoint, attempt):
        served_models.append(str(checkpoint))
        yield 'http://127.0.0.1:1234/v1'

    def forbidden(*args, **kwargs):
        raise AssertionError('Evaluation must never prepare or train')

    monkeypatch.setattr(evaluation, 'served', served)
    monkeypatch.setattr(evaluation, 'run_command', lambda argv, log, **kwargs: calls.append(argv))
    monkeypatch.setattr(pipeline, 'prepare', forbidden)
    monkeypatch.setattr(pipeline, 'training_command', forbidden)
    argv = ['--snapshot', 'BASE', '--verified', str(tmp_path / 'verified'), '--output', str(tmp_path),
            '--mode', 'evaluate', '--train-python', 'train', '--env-python', 'env']
    return argv, models, served_models, calls


@pytest.mark.parametrize('source', ['cli', 'env'])
@pytest.mark.parametrize('fractions,indices', [(None, [0, 1, 2, 3, 4]), ('', [0, 1, 2, 3, 4]),
    ('1', [4]), ('1/1', [4]), ('1, 1/2', [3, 4])])
def test_sad_evaluation_fractions_and_provenance(tmp_path, monkeypatch, sad_evaluation, source, fractions, indices):
    argv, models, served_models, calls = sad_evaluation
    if fractions is not None:
        if source == 'cli':
            argv += ['--eval-fractions', fractions]
        else:
            monkeypatch.setenv('SAD_EVAL_FRACTIONS', fractions)
    pipeline.main(argv)
    selected = [models[i] for i in indices]
    assert served_models == [model['checkpoint'] for model in selected]
    assert len([cmd for cmd in calls if 'eval/run_suite.py' in cmd]) == len(selected)
    provenance_path, = (tmp_path / 'evaluations').glob('*/provenance.json')
    provenance = json.loads(provenance_path.read_text())
    assert provenance['details'] == {
        'requested_fractions': [part.strip() for part in fractions.split(',')] if fractions else None,
        'selected_models': selected, 'decodings': ['greedy']}
    assert provenance['inputs'][str(tmp_path / 'models.json')] == file_hash(tmp_path / 'models.json')
    assert json.loads((tmp_path / 'models.json').read_text()) == models


@pytest.mark.parametrize('source', ['cli', 'env'])
@pytest.mark.parametrize('fractions', ['1/3', '1,1/3', '1,'])
def test_unknown_fractions_fail_before_any_checkpoint_work(tmp_path, monkeypatch, sad_evaluation, source, fractions):
    argv, _, served_models, calls = sad_evaluation
    if source == 'cli':
        argv += ['--eval-fractions', fractions]
    else:
        monkeypatch.setenv('SAD_EVAL_FRACTIONS', fractions)
    with pytest.raises(ValueError, match='Unknown evaluation fractions'):
        pipeline.main(argv)
    assert not served_models and not calls
    assert not (tmp_path / 'stages').exists()


def test_cli_selection_overrides_environment_and_ignores_unselected_artifacts(monkeypatch, sad_evaluation):
    argv, models, served_models, _ = sad_evaluation
    monkeypatch.setenv('SAD_EVAL_FRACTIONS', 'unknown')
    Path(models[0]['checkpoint'], 'training_state.json').unlink()
    pipeline.main(argv + ['--eval-fractions', '1'])
    assert served_models == [models[-1]['checkpoint']]


def test_selection_changes_reuse_completed_checkpoint_stages(tmp_path, sad_evaluation):
    argv, models, served_models, calls = sad_evaluation
    pipeline.main(argv + ['--eval-fractions', '1'])
    pipeline.main(argv)
    pipeline.main(argv + ['--eval-fractions', '1'])
    assert served_models == [model['checkpoint'] for model in [models[-1], *models[:-1]]]
    assert len(calls) == 15  # One HF reference, serving check, and evaluation per checkpoint.
    assert len(list((tmp_path / 'evaluations').glob('*/provenance.json'))) == 3


def test_fraction_selection_includes_retained_attempts_and_final_label_one(tmp_path, sad_evaluation):
    argv, models, served_models, _ = sad_evaluation
    retry = {**models[-1], 'name': 'sad-1of1-attempt-001', 'fraction': '1'}
    models.append(retry)
    write_json(tmp_path / 'models.json', models)
    pipeline.main(argv + ['--eval-fractions', '1'])
    assert served_models == [models[-2]['checkpoint'], retry['checkpoint']]
