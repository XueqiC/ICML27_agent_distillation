from copy import deepcopy
from functools import partial
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from repair.artifacts import checked_document
from repair.budgets import GlobalTeacherBudget
from repair.collect import collect, finalize_cases, freeze_probe_split, public_review, review_reader
from repair.dry_fakes import FakeEnvironment, FakeTeacher, FakeTokenizer, fake_review, O, F
from repair.execute import execute_actions
from repair.ledger import BudgetLedger
from repair.runtime import CaseSession, CaseStopped, make_teachers, RepairTeacher, CanaryRepairTeacher, Limits
from repair.prompts import Prompt
from repair.schemas import parse_diagnosis
from sed.config import load_config
from sed.storage import digest


def inputs(n=6):
    tasks = [{'task_id': f'train/synthetic-{i}/trial', 'group_id': f'train/synthetic-{i}',
        'split': 'train', 'sealed': False, 'task_type': 'pick_clean_then_place_in_recep',
        'instruction': 'Put a clean apple on table 1.', 'parent_seed_ids': [f'seed-{i}']} for i in range(n)]
    failures = [execute_actions(FakeEnvironment, t['task_id'],
        ['Thought: Use the visible information.\nAction: ' + a for a in O], 40) for t in tasks]
    return failures, tasks


def run(tmp_path, **kw):
    failures, tasks = inputs()
    return collect(failures, tasks, load_config('configs/base_e4b.yaml'), tmp_path,
        env_factory=FakeEnvironment, teacher_factory=kw.pop('teacher_factory', lambda *a, **k: FakeTeacher()),
        tokenizer=FakeTokenizer(), tokenizer_id='synthetic', checkpoint={'name': 'synthetic'},
        dry_run=True, K=6, **kw)


def test_freeze_precedes_teacher_factory_and_complete_dry_collection(tmp_path):
    out = tmp_path / 'collect'
    class ReasoningTeacher(FakeTeacher):
        def generate(self, *args):
            result = super().generate(*args)
            result.metadata['usage']['reasoning_tokens'] = 12
            return result
    def factory(*args, **kwargs):
        split = checked_document(out / 'repair_split.json')
        assert len(split['D_repair_probe']) == 2
        assert not (out / 'cases.jsonl').exists()
        return ReasoningTeacher()
    value = run(out, teacher_factory=factory, reviewer=fake_review)
    assert value['status'] == 'complete' and len(value['pairs']) == 4
    assert not set(value['D_repair_train']) & set(value['D_repair_probe'])
    assert value['D_repair_train'] == value['split']['training_candidates']
    assert checked_document(out / 'collection.json')['pairs'] == value['pairs']
    manifest = json.loads((out / 'run_manifest.json').read_text())
    assert manifest['K_trusted_seed_trajectories'] == 6
    assert manifest['teacher_responses'] and all(r['returned_model_id'] == 'fake' for r in manifest['teacher_responses'])
    assert manifest['code_revision'] and manifest['training_exposure'] == 0
    totals = manifest['global_budget']['reported_usage']
    for key in totals:
        assert totals[key] == sum(c['budget']['supervision']['teacher_' + key] for c in value['cases'])
    assert totals['reasoning_tokens'] > 0
    assert totals['output_tokens'] == totals['reasoning_tokens'] + totals['visible_tokens']
    assert manifest['global_budget']['unknown_quantities'] == {}
    assert 'near' in (out / 'human_review.tsv').read_text() and 'far' in (out / 'human_review.tsv').read_text()
    assert all(c['e1'] and c['candidates'] and c['parent_seed_ids'] for c in value['cases'])
    with pytest.raises(FileExistsError):
        run(out)


def test_pending_reviews_keep_far_labels_offline_finalize_without_teacher(tmp_path, monkeypatch):
    value = run(tmp_path / 'pending')
    assert not value['pairs'] and all(c['labelled_candidates'] for c in value['cases'])
    reviews = json.loads((tmp_path / 'pending' / 'review_requests.json').read_text())
    assert len(reviews) == 12
    def forbidden(*a, **kw):
        raise AssertionError('Offline finalization cannot call teacher')
    monkeypatch.setattr(FakeTeacher, 'generate', forbidden)
    accepted = {key: fake_review(public) for key, public in reviews.items()}
    pairs, exclusions = finalize_cases(value['cases'], value['split'], reviewer=review_reader(accepted))
    assert len(pairs) == 4 and all(not e['reasons'] for e in exclusions)
    from tools.repair_collect import main
    rp = tmp_path / 'reviews.json'; rp.write_text(json.dumps(accepted))
    assert main(['--branch', 'P0_K20', '--finalize', str(tmp_path / 'pending' / 'collection.json'), '--reviews', str(rp),
                 '--output', str(tmp_path / 'final')]) == 0
    assert len(checked_document(tmp_path / 'final' / 'collection.json')['pairs']) == 4


@pytest.mark.parametrize('cap,reason', [('max_teacher_input_tokens', 'global_input_tokens_cap'),
                                       ('max_teacher_output_tokens', 'global_output_tokens_cap')])
def test_global_cap_stops_before_call_preserves_exclusions(tmp_path, cap, reason):
    class Never(FakeTeacher):
        def generate(self, *a):
            raise AssertionError('Must reserve BEFORE teacher call')
    value = run(tmp_path / cap, teacher_factory=lambda *a, **k: Never(), **{cap: 0})
    assert value['status'] == reason and not value['pairs']
    assert value['global_budget']['reserved_attempts'] == 0
    assert len(value['cases']) == 1 and len(value['exclusions']) == 6


def test_global_budget_carries_across_cases_and_far_branch(tmp_path):
    value = run(tmp_path / 'cap', reviewer=fake_review, max_teacher_output_tokens=16500)
    assert value['status'] == 'global_output_tokens_cap'
    # Near proposal reservation 16384 fits after diagnosis; Far must reserve anew.
    assert value['global_budget']['reserved_attempts'] > 1
    assert value['global_budget']['tokens_or_unsettled_bounds']['output_tokens'] <= 16500
    assert len(value['cases']) < 6


def test_old_teacher_route_is_not_far_and_probe_choice_is_not_reassigned(tmp_path):
    value = run(tmp_path / 'duplicates', reviewer=fake_review, seed_actions={t['task_id']: [F] for t in inputs()[1]})
    assert not value['pairs']
    assert any('far_duplicates_original_teacher_seed' in e['reasons'] for e in value['exclusions'])
    assert value['D_repair_probe'] == freeze_probe_split(inputs()[1])['D_repair_probe']


def test_canary_one_diagnosis_no_proposal_and_no_format_retry(tmp_path):
    class Count(FakeTeacher):
        def __init__(self): self.calls = []
        def generate(self, h, t, v):
            from repair.prompts import unpack_request
            self.calls.append(unpack_request(v).kind)
            return super().generate(h, t, v)
    teacher = Count()
    value = run(tmp_path / 'canary', teacher_factory=lambda *a, **k: teacher, canary=True,
                max_teacher_output_tokens=8192)
    assert teacher.calls == ['diagnosis'] and len(value['cases']) == 1 and not value['pairs']
    constructed = make_teachers(load_config('configs/base_e4b.yaml'), tmp_path / 'clients',
                                run_id='fake', single_diagnosis=True)
    assert set(constructed) == {'diagnosis'}
    assert constructed['diagnosis'].teacher.max_attempts == 1
    assert constructed['diagnosis'].teacher.sampling['max_output_tokens'] == 8192


@pytest.mark.parametrize('canary', [False, True])
def test_global_caps_include_http_retry_and_canary_never_retries(tmp_path, task, canary):
    from teachers.capped import CappedTeacher
    from teachers.openai_teacher import TeacherCallError
    from test_capped_teacher import SequenceClient, http_429
    from test_stage2c_teacher import FakeClock
    from repair_teacher_fakes import DIAGNOSIS
    clock = FakeClock()
    sdk = SequenceClient([http_429()])
    cap = 8192
    teacher = CappedTeacher(counter_path=tmp_path / 'counter.json', cap=2,
        cache_dir=tmp_path / 'cache', usage_ledger=tmp_path / 'usage.jsonl', run_id='repair-global',
        client=sdk, teacher_class=CanaryRepairTeacher if canary else RepairTeacher,
        sampling={'max_output_tokens': cap}, clock=clock, sleep=clock.sleep, jitter=lambda lo, hi: 0)
    budget = GlobalTeacherBudget(tmp_path / 'global.jsonl', output_tokens=cap)
    session = CaseSession(task=task, teacher=teacher, ledger=BudgetLedger(tmp_path / 'case.jsonl'),
        config=load_config('configs/base_e4b.yaml'), global_budget=budget,
        output_caps={'diagnosis': cap}, limits=Limits(format_retry_calls=0))
    with pytest.raises((CaseStopped, TeacherCallError)):
        session.request(Prompt('diagnosis', {'task': 'public'}), lambda s: parse_diagnosis(s, 'original', 3))
    assert len(sdk.requests) == 1
    assert budget.used['output_tokens'] == cap
    assert budget.summary()['reported_usage']['reasoning_tokens'] == 0
    failed = next(r for r in session.ledger.rows() if r['event'] == 'teacher_failure')
    assert failed['supervision']['teacher_reasoning_tokens'] is None
    assert failed['supervision']['teacher_visible_tokens'] is None
    assert 'PRIVATE_PROVIDER_EXCEPTION' not in (tmp_path / 'case.jsonl').read_text()


@pytest.mark.parametrize('caps,failure_cap', [
    ({'diagnosis': 8192, 'proposal': 16384, 'far': 16384, 'rationale': 1024}, 64000),
    ({'diagnosis': 4096, 'proposal': 8192, 'far': 12288, 'rationale': 768}, 80000),
])
def test_cli_dry_run_full_pipeline(tmp_path, caps, failure_cap):
    from tools.repair_collect import main
    options = [] if failure_cap == 64000 else [
        arg for kind, cap in caps.items() for arg in (f'--{kind}-max-output-tokens', str(cap))]
    if options:
        options += ['--max-failure-output-tokens', str(failure_cap)]
    assert main(['--branch', 'P0_K20', '--dry-run', '--output', str(tmp_path / 'cli'), *options]) == 0
    assert len(checked_document(tmp_path / 'cli' / 'collection.json')['pairs']) == 4
    manifest = json.loads((tmp_path / 'cli/run_manifest.json').read_text())
    config = manifest['config']
    effective = config['repair_collection']
    assert effective['teacher_output_caps'] == caps
    assert effective['case_limits']['output_tokens'] == failure_cap
    assert effective['case_limits']['input_tokens'] == 250000
    assert manifest['global_budget']['caps'] == {'input_tokens': 5_000_000, 'output_tokens': 500_000}
    clients = make_teachers(config, tmp_path / 'clients', run_id='configured')
    assert {kind: client.teacher.sampling['max_output_tokens'] for kind, client in clients.items()} == caps
    seen = set()
    for path in (tmp_path / 'cli/cases').glob('*/budget.jsonl'):
        for row in BudgetLedger(path).rows():
            assert 'teacher_reasoning_tokens' in row['supervision']
            assert 'teacher_visible_tokens' in row['supervision']
            if row['event'] == 'teacher_request':
                kind = row['details']['request']['kind']
                seen.add(kind)
                assert row['details']['output_cap'] == caps[kind]
    assert seen == set(caps)


@pytest.mark.parametrize('truncated', [False, True])
def test_canary_reports_reasoning_and_visible_usage_even_without_text(tmp_path, truncated):
    from teachers.capped import CappedTeacher
    from test_capped_teacher import SequenceClient
    from repair_teacher_fakes import DIAGNOSIS
    output = 8192 if truncated else 425
    reasoning = 8192 if truncated else 384
    response = {'output_text': '' if truncated else json.dumps(DIAGNOSIS),
        'status': 'incomplete' if truncated else 'completed', 'model': 'fake-model',
        'id': 'fake', '_request_id': 'fake-request',
        'incomplete_details': {'reason': 'max_output_tokens'} if truncated else None,
        'usage': {'input_tokens': 3799, 'output_tokens': output,
                  'output_tokens_details': {'reasoning_tokens': reasoning}}}
    sdk = SequenceClient([SimpleNamespace(status_code=200, headers={}, parse=lambda: response)])
    def factory(config, directory, *, run_id):
        return CappedTeacher(counter_path=directory / 'counter.json', cap=1,
            cache_dir=directory / 'cache', usage_ledger=directory / 'usage.jsonl', run_id=run_id,
            client=sdk, teacher_class=CanaryRepairTeacher,
            sampling={'max_output_tokens': config['repair_collection']['teacher_output_caps']['diagnosis']})
    value = run(tmp_path / 'canary', canary=True, teacher_factory=factory)
    assert value['status'] == ('canary_failed' if truncated else 'canary_complete')
    assert len(sdk.requests) == 1 and sdk.requests[0]['max_output_tokens'] == 8192
    assert len(value['cases']) == 1 and not value['pairs']
    totals = {'input_tokens': 3799, 'output_tokens': output,
              'reasoning_tokens': reasoning, 'visible_tokens': output - reasoning}
    manifest = json.loads((tmp_path / 'canary/run_manifest.json').read_text())
    assert manifest['global_budget']['reported_usage'] == totals
    assert manifest['global_budget']['tokens_or_unsettled_bounds']['output_tokens'] == output
    case = value['cases'][0]['budget']['supervision']
    assert {k: case['teacher_' + k] for k in totals} == totals
    global_rows = [json.loads(line) for line in
                   (tmp_path / 'canary/global_teacher_budget.jsonl').read_text().splitlines()]
    assert all('reasoning_tokens' in row['reported_usage'] for row in global_rows)
    assert {key: global_rows[-1]['usage'][key] for key in totals} == totals


@pytest.mark.parametrize('caps,global_cap,expected', [
    ({'diagnosis': 16384}, 500000, 8192),
    ({'diagnosis': 4096}, 500000, 4096),
    ({'diagnosis': 8192}, 8000, None),
])
def test_canary_overrides_remain_bounded(tmp_path, caps, global_cap, expected):
    value = run(tmp_path / 'canary', canary=True, output_caps=caps,
                max_teacher_output_tokens=global_cap)
    assert value['global_budget']['reserved_attempts'] == int(expected is not None)
    assert value['status'] == ('canary_complete' if expected else 'global_output_tokens_cap')
    manifest = json.loads((tmp_path / 'canary/run_manifest.json').read_text())
    config = manifest['config']
    clients = make_teachers(config, tmp_path / 'clients', run_id='canary', single_diagnosis=True)
    assert set(clients) == {'diagnosis'}
    assert clients['diagnosis'].teacher.sampling['max_output_tokens'] == min(caps['diagnosis'], 8192)


def test_global_reported_usage_tracks_unknown_reasoning_without_releasing_output(tmp_path):
    budget = GlobalTeacherBudget(tmp_path / 'budget.jsonl', output_tokens=20)
    budget.reserve(20, 20)
    budget.settle(20, 20, {'input_tokens': 10, 'output_tokens': 12})
    assert budget.summary()['reported_usage'] == {
        'input_tokens': 10, 'output_tokens': 12, 'reasoning_tokens': 0, 'visible_tokens': 0}
    assert budget.summary()['unknown_quantities'] == {'reasoning_tokens': 1, 'visible_tokens': 1}
    assert budget.used['output_tokens'] == 12
    with pytest.raises(CaseStopped, match='global_output_tokens_cap'):
        budget.reserve(1, 9)


def test_global_budget_cannot_restart_or_bypass_a_stop(tmp_path):
    path = tmp_path / 'global.jsonl'
    budget = GlobalTeacherBudget(path, output_tokens=10)
    with pytest.raises(CaseStopped): budget.reserve(1, 11)
    with pytest.raises(CaseStopped): budget.reserve(1, 1)
    with pytest.raises(ValueError, match='already exists'): GlobalTeacherBudget(path)


def test_wrong_public_review_hash_never_approves_an_episode(tmp_path):
    value = run(tmp_path / 'pending')
    pairs, exclusions = finalize_cases(value['cases'], value['split'],
        reviewer=review_reader({'wrong-hash': fake_review({})}))
    assert not pairs
    assert any('near:public_quality_review_missing' in e['reasons'] for e in exclusions)


def test_two_probes_required_and_canary_caps_apply_to_library_too(tmp_path):
    with pytest.raises(ValueError, match='exactly two probes'):
        freeze_probe_split(inputs(1)[1])
    value = run(tmp_path / 'tiny-canary', canary=True)
    assert value['global_budget']['caps'] == {'input_tokens': 16384, 'output_tokens': 8192}
    assert value['global_budget']['reserved_attempts'] == 1
    assert value['status'] == 'canary_complete'
    manifest = json.loads((tmp_path / 'tiny-canary/run_manifest.json').read_text())
    assert manifest['config']['repair_collection']['case_limits']['format_retry_calls'] == 0


from test_repair_coverage import coverage_bank, ActionBackend


def test_real_coverage_artifacts_feed_collector_preflight_and_reject_tampering(coverage_bank, tmp_path):
    from harness.alfworld_env import Environment
    from sed.runner import ScriptedPolicy
    from tools.repair_coverage import run_coverage
    from tools.repair_collect import load_failures
    m, f, tasks, config, spec, splits = coverage_bank
    factory = lambda: Environment(config['environment']['data_root'], max_episode_steps=2, backend_factory=ActionBackend)
    coverage = tmp_path / 'coverage'
    run_coverage(tasks[:2], config, coverage, spec=spec, env_factory=lambda _: factory(),
                 policy_factory=lambda *a, **kw: ScriptedPolicy(['look', 'look']))
    preflight = tmp_path / 'preflight'; preflight.mkdir()
    failures, selected = load_failures(coverage / 'failures.json', config, splits, m, f, factory, preflight)
    frozen = json.loads((coverage / 'failures.json').read_text())
    assert [t['task_id'] for t in selected] == [r['task_id'] for r in frozen['failures']]
    assert len(failures) == 2 and all(r.success is False for r in failures)
    assert set(failures[0].token_costs) == {'prompt_tokens', 'completion_tokens', 'thinking_tokens', 'thinking_chars'}
    single = tmp_path / 'canary-preflight'; single.mkdir()
    failures_one, tasks_all = load_failures(coverage / 'failures.json', config, splits, m, f, factory,
                                           single, canary=True)
    assert len(failures_one) == 1 and len(tasks_all) == 2
    ep = coverage / frozen['failures'][0]['episode_dir'] / 'episode.json'
    ep.write_text(ep.read_text() + ' ')
    another = tmp_path / 'changed'; another.mkdir()
    with pytest.raises(ValueError, match='evidence changed'):
        load_failures(coverage / 'failures.json', config, splits, m, f, factory, another)
