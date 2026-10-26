from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path

import pytest

from repair.dry_fakes import FakeEnvironment, N, O
from repair.probes import probe_point, probe_case, combined_table, quick_eval, run_probes
from repair.records import RepairRecord
from repair.execute import execute_actions
from sed.storage import digest, write_json
from test_repair_collect import run, fake_review
from test_lora_serving import adapter_bank, bound_report


@pytest.fixture
def corpus(tmp_path):
    return run(tmp_path / 'collection', reviewer=fake_review), []


def test_witness_actions_exclude_successful_earlier_state_changes(corpus):
    data, _ = corpus
    case = data['cases'][0]
    original, point, witnesses, excluded = probe_point(case)
    assert point == 3 and list(witnesses) == [N[3]]
    far = case['far_candidates'][0]['execution']
    assert far['success'] is True and far['trajectory_id'] in excluded
    assert 'go to table 1' not in witnesses
    assert original.success is False
    # A successful route can contain a parser-feedback turn; that turn has no
    # next action and cannot populate A-plus.
    near = RepairRecord.from_json(case['repair']).executed_trajectory
    steps = deepcopy(near.steps)
    steps[point]['action_parsed'] = None
    invalid = replace(near, steps=steps)
    case['candidates'].append({'execution': invalid.to_json()})
    _, _, actions, rejected = probe_point(case)
    assert list(actions) == [N[3]] and invalid.trajectory_id in rejected


def test_real_prefix_replay_eight_samples_two_autonomous_continuations(corpus, tmp_path):
    case = corpus[0]['cases'][0]
    calls = []
    def policy(seed, tid, path):
        def act(history, visible):
            assert 'internal_state' not in json.dumps([history, visible])
            calls.append((seed, len(history), deepcopy(history), visible))
            return 'Action: ' + N[len(history)]
        return act
    row = probe_case(case, env_factory=FakeEnvironment, policy_factory=policy, output=tmp_path / 'probe')
    assert row['m'] == 1 and len(row['samples']) == 8
    assert [s['seed'] for s in row['samples']] == list(range(8))
    original = RepairRecord.from_json(case['repair']).original_trajectory
    for seed, length, history, visible in calls[:8]:
        assert length == 3
        assert history == original.decision_context(3)['history']
        assert visible == original.decision_context(3)['visible_context']
    assert all(c['success'] is True and c['prefix_turns'] == 3 and c['remaining_H'] == 37 for c in row['continuations'])
    assert all(c['new_failure_position'] is None for c in row['continuations'])
    # Three replay steps once for decision; two full six-step real executions.
    assert row['budget']['interaction']['evaluation_steps'] == 15


def test_prefix_divergence_prevents_all_policy_calls(corpus, tmp_path, monkeypatch):
    def reset(self, tid):
        return self.state('Changed initial observation.')
    monkeypatch.setattr(FakeEnvironment, 'reset', reset)
    def factory(*a):
        return lambda *a: pytest.fail('Policy called on a non-replayed history')
    row = probe_case(corpus[0]['cases'][0], env_factory=FakeEnvironment, policy_factory=factory, output=tmp_path / 'bad')
    assert row['m'] is None and row['samples'] == []
    assert all(c['success'] == 'unknown' for c in row['continuations'])
    assert row['decision_replay']['error_type'].startswith('handback_prefix_check')


def test_continuation_prefix_consumes_horizon_and_parser_failures_count(corpus, tmp_path):
    policy = lambda *a: lambda h, v: 'malformed\nreply'
    row = probe_case(corpus[0]['cases'][0], env_factory=FakeEnvironment, policy_factory=policy, output=tmp_path / 'parse')
    assert row['m'] == 0
    for c in row['continuations']:
        assert c['success'] is False and len(c['execution']['steps']) == 40
        assert c['new_failure_position'] == 3
        assert c['execution']['environment_steps'] == 3  # parser turns consume H but no environment transition


def test_context_overflow_is_known_failure_and_transport_unknown(corpus, tmp_path):
    from students.vllm_student import ContextOverflow
    for kind in ['context', 'transport']:
        def policy(*a):
            def act(*a):
                if kind == 'context': raise ContextOverflow({})
                raise RuntimeError('PRIVATE_SECRET')
            return act
        row = probe_case(corpus[0]['cases'][0], env_factory=FakeEnvironment, policy_factory=policy, output=tmp_path / kind)
        assert row['m'] is None
        assert all(c['success'] is False if kind == 'context' else c['success'] == 'unknown' for c in row['continuations'])
        assert 'PRIVATE_SECRET' not in json.dumps(row)


def table_fixture():
    result = []
    for arm, checkpoint, m, quick in [('shared', 0, .25, [False, True]), ('N', 1, .5, None),
            ('F', 1, .375, None), ('N', 20, .75, [True, True]), ('F', 20, .5, [False, True])]:
        result.append({'arm': arm, 'checkpoint': checkpoint, 'collection_sha256': 'frozen',
            'split_sha256': 'splits', 'seeds': list(range(8)), 'H': 40, 'temperature': .7, 'top_p': .95,
            'cases': [{'task_id': 'probe', 'm': m, 'continuations': [{'success': True}, {'success': False}]}],
            'quick_eval': {'tasks': [{'task_id': str(i), 'success': s, 'status': 'ok'} for i, s in enumerate(quick)]}
                if quick is not None else None})
    return result


def test_combined_delta_pair_counts_and_missing_points(tmp_path):
    table = combined_table(table_fixture(), tmp_path / 'combined')
    assert table['checkpoints'][1]['delta_m_N_minus_delta_m_F'] == .125
    assert table['checkpoints'][2]['delta_m_N_minus_delta_m_F'] is None
    assert table['checkpoints'][3]['delta_m_N_minus_delta_m_F'] == .25
    assert table['quick_paired_counts']['theta0_vs_N20']['right_only'] == 1
    assert table['quick_paired_counts']['N20_vs_F20']['left_only'] == 1
    assert table['missing_checkpoints'] == [['N', 5], ['F', 5]]
    bad = table_fixture(); bad[1]['seeds'] = [9]
    with pytest.raises(ValueError, match='conditions differ'):
        combined_table(bad, tmp_path / 'bad')
    with pytest.raises(ValueError, match='duplicate'):
        combined_table([*table_fixture(), table_fixture()[0]], tmp_path / 'duplicate')


def test_unknown_observations_do_not_turn_into_zero_or_disappear(tmp_path):
    rows = table_fixture()
    rows[1]['cases'][0]['m'] = None
    rows[-1]['quick_eval']['tasks'][0].update(success=True, status='error')
    value = combined_table(rows, tmp_path / 'table')
    assert value['checkpoints'][1]['delta_m_N_minus_delta_m_F'] is None
    assert value['quick_paired_counts']['N20_vs_F20']['unknown'] == 1


def test_quick_eval_calls_existing_suite_greedy_and_keeps_paired_records(tmp_path, monkeypatch):
    import sed.manifest as mf
    import eval.run_suite as suite
    tasks = [{'task_id': f'train/quick-{i}/trial', 'group_id': f'train/quick-{i}',
              'task_type': 'pick_and_place_simple', 'split': 'train', 'sealed': False} for i in range(24)]
    ids = [t['task_id'] for t in tasks]
    listing = {'task_ids': ids, 'list_sha256': digest(ids)}
    monkeypatch.setattr(mf, 'load_frozen', lambda *a: ({}, {}))
    monkeypatch.setattr(suite, 'frozen_tasks', lambda *a, **k: (listing, tasks))
    seen = []
    def fake_suite(**kw):
        seen.append(kw)
        for i, task in enumerate(tasks):
            path = Path(kw['output']) / 'episodes' / f"{i:03d}-{digest(task['task_id'])[:12]}"
            path.mkdir(parents=True)
            write_json(path / 'episode.json', {'status': 'ok', 'success': i % 2 == 0})
        return {'successes': 12}
    result = quick_eval({'paths': {'manifest': 'fake', 'split_freeze': 'fake'}},
        {'sets': {'D_unseen_quick': listing}}, tmp_path / 'quick', {'model': 'adapter'}, suite_runner=fake_suite)
    assert seen[0]['spec']['decoding'] == 'greedy' and seen[0]['policy'] == 'student'
    assert len(result['tasks']) == 24 and sum(t['success'] for t in result['tasks']) == 12


def test_only_shared_zero_and_final_get_quick_eval(corpus, tmp_path, monkeypatch):
    import repair.probes as p
    data, _ = corpus
    data['checkpoint']['identity'] = {'adapter_config_sha256': 'c', 'adapter_weights_sha256': 'w',
                                     'base_model': 'm', 'revision': 'r'}
    monkeypatch.setattr(p, 'validate_serving', lambda *a: data['checkpoint']['identity'])
    calls = []
    def quick(*a, **kw): calls.append(a); return {'tasks': []}
    from sed.config import load_config
    config = load_config('configs/base_e4b.yaml')
    for arm, cp in [('shared', 0), ('N', 1), ('N', 5), ('N', 20)]:
        p.run_probes(data, {'content_sha256': 'fake'}, config, tmp_path / f'cp{cp}', arm=arm, checkpoint=cp,
            spec={}, env_factory=FakeEnvironment, policy_factory=lambda *a: lambda h, v: 'Action: ' + N[len(h)],
            quick_runner=quick)
    assert len(calls) == 2
    with pytest.raises(ValueError):
        p.run_probes(data, {}, config, tmp_path / 'bad0', arm='N', checkpoint=0, spec={}, env_factory=FakeEnvironment)


def test_served_factory_uses_m0b1_sampled_decoding_without_teacher(monkeypatch, tmp_path):
    from students.sampled_student import SampledStudent
    from repair.probes import served_policy_factory
    from types import SimpleNamespace
    calls = []
    def generate(self, history, task, visible):
        calls.append((self.spec, self.request_sampling(history, task)))
        return SimpleNamespace(raw_text='Action: look')
    monkeypatch.setattr(SampledStudent, 'generate', generate)
    factory = served_policy_factory({'endpoint': 'http://127.0.0.1:12345/v1', 'model': 'adapter',
        'revision': 'fake', 'enable_thinking': False, 'max_tokens': 1024})
    assert factory(7, 'train/t', tmp_path)([{}, {}, {}], {}) == 'Action: look'
    assert calls[0][1] == {'temperature': .7, 'top_p': .95, 'n': 1,
                           'seed': int(digest([7, 'train/t', 3])[:8], 16)}
    assert calls[0][0]['max_tokens'] == 1024


def test_quick_pair_ids_must_match(tmp_path):
    rows = table_fixture()
    rows[-1]['quick_eval']['tasks'][0]['task_id'] = 'different-task'
    with pytest.raises(ValueError, match='task identities'):
        combined_table(rows, tmp_path / 'bad')


def test_serving_gate_uses_real_report_schema_and_server_artifact_path(bound_report, adapter_bank):
    from repair.probes import validate_serving
    report, server = bound_report
    path, _, student = adapter_bank
    assert 'path' not in report['identity']['adapter']
    server['config']['server']['batch_invariant'] = True
    report['server_provenance_sha256'] = digest(server)
    config = {'server_provenance': server, 'lora_equivalence': report, 'models': {'student': student}}
    spec = {'model': 'sft', 'endpoint': 'http://127.0.0.1:34567/v1'}
    adapter = validate_serving(config, spec)
    assert Path(adapter['path']) == path
    server['config']['server']['batch_invariant'] = False
    with pytest.raises(ValueError, match='BATCH_INVARIANT=1'):
        validate_serving(config, spec)
