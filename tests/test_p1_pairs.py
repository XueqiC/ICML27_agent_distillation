from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import random

import pytest

from tests.p0_fakes import fake_pair, TinyTokenizer
from repair.records import TrajectoryRecord
from repair.p0_pairs import CATEGORIES
from sed.storage import digest, write_json, file_hash
from surgery.common import Relocation, output_guard
from surgery.pairs import split_a, extract_a, extract_b, load_pairs, collateral_b


def split_fixture():
    return [dict(fake_pair(i + c * 12, category=category), group_id=f'g-{c}-{i}',
                 task_id=f't-{c}-{i}', state=dict(sha256=f's-{c}-{i}'))
            for c, category in enumerate(CATEGORIES) for i in range(12)]


def test_split_counts_deterministic_preserves_fields():
    source = split_fixture()
    original = deepcopy(source)
    output = split_a(source)
    assert source == original
    assert output == split_a(source)
    for c in CATEGORIES:
        assert sum(p['split'] == 'diagnosis' and p['category'] == c for p in output) == 6
    assert not {p['group_id'] for p in output if p['split'] == 'diagnosis'} & {p['group_id'] for p in output if p['split'] == 'test'}
    for a, b in zip(original, output):
        assert {k: v for k, v in a.items() if k != 'split'} == {k: v for k, v in b.items() if k != 'split'}


def test_actual_frozen_split_and_sha(tmp_path):
    source = Path('reports/p0_tooling/extraction_v3/pairs.json')
    if not source.exists():
        pytest.skip('Read-only original P0 freeze unavailable')
    value = extract_a(source)
    assert value['counts']['diagnosis']['pairs'] == 55
    assert value['counts']['test']['pairs'] == 58
    assert list(value['counts']['diagnosis']['categories'].values()) == [13, 18, 9, 6, 9]
    assert value['split_sha256'] == '5dfde976ea4820bbc9b97e5f6c0f6f2b3c92eca31a80bf01590ce5d02a815715'
    path = tmp_path / 'pairs.json'
    write_json(path, value)
    assert load_pairs(path)['sha256'] == value['sha256']
    value['pairs'][0]['split'] = 'diagnosis' if value['pairs'][0]['split'] == 'test' else 'test'
    write_json(path, value)
    with pytest.raises(ValueError):
        load_pairs(path)


def test_split_refuses_unattainable_group_counts():
    source = split_fixture()
    for p in source:
        p['group_id'] = p['category']
    with pytest.raises(ValueError, match='Exact'):
        split_a(source)


def test_b_extraction_teacher_forced_history_and_matching():
    pair = fake_pair()
    record = TrajectoryRecord.from_json(pair['audit']['original'])
    seen = []
    def greedy(state):
        seen.append(state)
        action = record.steps[0]['action_parsed'] if len(seen) == 1 else 'inventory'
        return dict(action=action, reply='Action: ' + action, prompt_tokens=1, generated_tokens=1)
    student = SimpleNamespace(tokenizer=TinyTokenizer(), greedy=greedy, identity={'fake': True}, usage={},
                              max_new_tokens=4, max_model_len=8192)
    value = extract_b(student, [(record, 'group')])
    assert len(value['matched_pool']) == 1
    assert len(value['pairs']) == len(record.steps) - 1
    assert all(p['split'] == 'diagnosis' and p['category'] == 'none' and p['a_f'] == 'inventory' for p in value['pairs'])
    assert record.steps[0]['action_raw'] in seen[1]['text']
    assert 'Action: inventory' not in seen[1]['text']
    assert value['pairs'][0]['a_r'] == record.steps[1]['action_parsed']
    assert len(value['decisions']) == len(record.steps)


def test_b_null_parse_preserved():
    record = TrajectoryRecord.from_json(fake_pair()['audit']['original'])
    student = SimpleNamespace(tokenizer=TinyTokenizer(), greedy=lambda s: dict(action=None, reply='bad format'),
                              identity={}, usage={}, max_new_tokens=4, max_model_len=8192)
    value = extract_b(student, [(record, 'g')])
    assert len(value['pairs']) == len(record.steps)
    assert all(p['a_f'] is None for p in value['pairs'])


def test_b_collateral_fixed_random_sample_and_short_pool():
    pool = [dict(pair_id=str(i)) for i in range(210)]
    assert collateral_b(dict(matched_pool=pool)) == random.Random(0).sample(pool, 200)
    with pytest.raises(ValueError):
        collateral_b(dict(matched_pool=pool[:199]))


def test_relocation_component_single_substitution_and_hash(tmp_path):
    old, new, third = tmp_path / 'old', tmp_path / 'new', tmp_path / 'third'
    new.mkdir()
    (new / 'file.json').write_text('{}')
    relocation = Relocation([f'{old}={new}', f'{new}={third}'])
    assert relocation.path(old / 'file.json') == new / 'file.json'
    assert relocation.path(str(old) + 'extra') == Path(str(old) + 'extra')
    relocation.inputs({str(old / 'file.json'): file_hash(new / 'file.json')})
    assert relocation.path('file.json', root=new) == new / 'file.json'


@pytest.mark.parametrize('rules', [['/a=/b', '/a/c=/d'], ['relative=/b'], ['/a/../x=/b'], ['/a=/b/../c'], ['invalid']])
def test_bad_relocation(rules):
    with pytest.raises(ValueError):
        Relocation(rules)


def test_relocation_symlink_escape(tmp_path):
    old, new, outside = tmp_path / 'old', tmp_path / 'new', tmp_path / 'outside'
    new.mkdir(); outside.mkdir()
    (new / 'link').symlink_to(outside, target_is_directory=True)
    with pytest.raises(ValueError, match='escape'):
        Relocation([f'{old}={new}']).path(old / 'link/file')


@pytest.mark.parametrize('path', ['/home/xueqi/hq/projects/ICML27_v0_runs/new', 'reports/p0_runs/new', 'reports/p0_tooling/new'])
def test_output_guard(path):
    with pytest.raises(ValueError):
        output_guard(path)


def test_seed_version_translation_matches_p0_and_refuses_changed_environment(monkeypatch, tmp_path):
    import importlib.metadata
    from harness.alfworld_env import Environment
    from repair.execute import environment_version
    from repair.p0_pairs import ROOT
    from surgery.pairs import seed_replay_version
    versions = dict(alfworld='fixture-alf', textworld='fixture-tw', numpy='fixture-np')
    monkeypatch.setattr(importlib.metadata, 'version', lambda n: versions[n])
    source = dict(dependencies=versions, code_sha256={p: file_hash(ROOT / p) for p in
        ('harness/alfworld_env/environment.py', 'src/sed/react.py', 'src/sed/visibility.py')})
    # Constructor does not reset/load a real environment.
    assert seed_replay_version(source) == environment_version(Environment(tmp_path))
    source['code_sha256']['src/sed/react.py'] = 'changed'
    with pytest.raises(ValueError, match='code differs'):
        seed_replay_version(source)
