"""Merged training routes retain each method's input restrictions."""
from types import SimpleNamespace

import pytest

from baselines.common import pipeline, fsdp_train
from sed.seed_selection import ALL_TRAIN_RULE


@pytest.mark.parametrize('selection,options,accepted', [
    ({'K': 3}, {}, False),
    ({'K': 3, 'selection_rule': ALL_TRAIN_RULE}, {}, True),
    ({'K': 3}, {'unrestricted_verified_size': True}, True),
    ({'K': 3}, {'registered_k': None}, True),
    ({'K': 3}, {'registered_k': (3,)}, True),
    ({'K': 3}, {'registered_k': (5,)}, False),
    ({'K': 3}, {'registered_k': (5,), 'unrestricted_verified_size': True}, True),
    ({'K': 5}, {}, True),
])
def test_support_guards_keep_full_train_and_both_overrides(tmp_path, monkeypatch, selection, options, accepted):
    class Accepted(Exception):
        pass
    monkeypatch.setattr(pipeline, 'load_frozen', lambda *a: ({}, {}))
    monkeypatch.setattr(pipeline, 'load_verified_collection', lambda *a: (
        {'verified': []}, dict(selection, seed=0), tmp_path))
    def accepted_tasks(*args):
        raise Accepted()
    monkeypatch.setattr(pipeline, 'training_tasks', accepted_tasks)
    args = SimpleNamespace(output=tmp_path / 'out', verified='fixture')
    config = {'paths': {'raw': str(tmp_path / 'raw'), 'verified': str(tmp_path / 'verified'),
                       'manifest': 'manifest', 'split_freeze': 'freeze'}}
    with pytest.raises(Accepted if accepted else ValueError):
        pipeline.prepare(args, config, method='fixture', recipe={}, **options)


@pytest.mark.parametrize('extra', [[], ['--processed-token-budget', '10'], ['--method-data', 'fixture']])
def test_had_route_preserves_sad_only_budget_and_steca_data_guards(monkeypatch, extra):
    from baselines.had import training
    received = []
    monkeypatch.setattr(training, 'run', lambda args: received.append(args))
    argv = ['--objective', 'had', '--snapshot', 'base', '--base-snapshot', 'base',
        '--verified', 'fixture', '--output', 'unused', '--had-data', 'data', '--had-identity', 'identity',
        '--had-round', '0', '--start-update', '0', '--learning-rate', '1e-5', '--scheduler', 'cosine',
        '--warmup-ratio', '0', '--warmup-steps', '15', '--original-batch', '60',
        '--weight-decay', '.01', '--beta', '.5', '--epochs', '1']
    if extra:
        with pytest.raises(SystemExit) as exc:
            fsdp_train.main(argv + extra)
        assert exc.value.code == 2 and not received
    else:
        fsdp_train.main(argv)
        assert len(received) == 1 and received[0].objective == 'had'
