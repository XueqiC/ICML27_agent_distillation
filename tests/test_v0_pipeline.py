"""End-to-end V0 orchestration and expansion boundaries using existing fakes."""
from collections import Counter
from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from repair.dry_fakes import FakeEnvironment, N, O
from repair.execute import execute_actions
from repair.joint import FakeReviewClient, FakeStudent
from repair.supervision import build_trajectory, seal_dataset
from repair.v0_common import CAPS, THETA0_CELLS, SyntheticManifest, artifacts, finish, phase, read
from repair.v0_data import load_dataset, publish_dataset
from repair.v0_expand import expand, expansion_plan, generate
from repair.v0_synth import edit_kind
from sed.storage import digest, file_hash, write_json
from tools import v0_round as cli
from test_v0_round import SEED, TASK, task_row, review_value, seed_ids
from test_synth_tasks import small_game, FakeEnv, template_seeds
from repair import synth_tasks as factory


def arguments(root, k=20, **kwargs):
    seeds = root.parent / f'seeds-k{k}.json'
    write_json(seeds, dict(K=k, verified=[dict(task_id=tid) for tid in seed_ids(k)]))
    return SimpleNamespace(output=str(root), seed=0, config='configs/base_e4b.yaml', resume=False,
        seeds=str(seeds), synthetic_manifest=None, endpoint='http://localhost:1/v1', model_name=THETA0_CELLS[k],
        server_provenance='fake', lora_check='fake', task_ids=None, action_only=True,
        splits='data/manifest/method_v1_splits.json', verified_root=None, theta0_training_dir=None,
        snapshot=None, replay=None, byte_token_cap=True, **kwargs)


def fake_adapter(k=20):
    return dict(name=THETA0_CELLS[k], adapter_weights_sha256='a' * 64,
                adapter_config_sha256='b' * 64, revision='fake-revision')


def setup_synth(root, k=20):
    directory = root / 'synth'
    phase(directory, dict(command='synth-fake', inputs={}))
    seeds = seed_ids(k)
    row = task_row()
    row.update(fingerprint=factory.fingerprint(SEED, small_game()), task_text='Put a clean apple on table 1.')
    write_json(directory / 'split_freeze.json', dict(groups=[], sources=[]))
    write_json(directory / 'manifest.json', dict(schema='v0-synthetic-tasks/1', tasks=[row], seed_task_ids=seeds, K=k))
    finish(directory)
    return SyntheticManifest(directory / 'manifest.json')


def replay_bundle(path, seeds):
    class Policy:
        def check(self, tid, source):
            assert tid in seeds
            return tid.rsplit('/', 1)[0], 'pick_and_place_simple'
    rows = []
    for tid in sorted(seeds):
        execution = execute_actions(FakeEnvironment, tid, N, 40)
        episode = dict(schema_version=2, task_id=tid, success=True, complete=True, status='ok',
            split='train', sealed=False, initial=execution.initial, steps=execution.steps)
        rows.append(build_trajectory(episode, arm='CONTINUE', source='k20_replay', episode_ref='seed/' + tid,
                                     policy=Policy()))
    path.mkdir(parents=True)
    seal_dataset(path, rows, policy=Policy(), arm='CONTINUE', inputs={}, splits_sha256='c' * 64,
                 theta0=fake_adapter(len(seeds)), excluded=Counter(), code_revision='fake')
    return path


class WholeClient(FakeReviewClient):
    calls = 0
    def create(self, **kwargs):
        self.calls += 1
        payload = json.loads(kwargs['input'][-1]['content'])
        value = dict(plan=[], preservable_choices=[],
            breaking_points=[dict(step=3, category='missing_precondition', reason='Cleaning is missing')],
            patch=dict(original_trajectory_id=payload['original_trajectory_id'], blocks=[dict(
                block_id='clean', op='replace', start=3, end_exclusive=4, actions=[N[3]])]))
        result = SimpleNamespace(output_text=json.dumps(value), model='FAKE_NO_MODEL', id='fake', status='completed',
            usage={'input_tokens': 100, 'output_tokens': 80, 'output_tokens_details': {'reasoning_tokens': 0}})
        return SimpleNamespace(status_code=200, headers={}, parse=lambda: result)


def fake_runtime(monkeypatch, manifest, *, successful=False):
    monkeypatch.setattr(cli, 'serving', lambda args, config: ({}, fake_adapter(manifest.k), []))
    monkeypatch.setattr(SyntheticManifest, 'environment', lambda self, tid, **kw: FakeEnvironment())
    def student(spec, tid, decoding, seed, log_dir):
        # One successful sampled attempt provides S; the rest provide failures.
        actions = N if successful or seed == 2 else O
        return lambda: FakeStudent(replies=['Action: ' + a for a in actions] + ([] if actions == N else ['Action: look']))
    monkeypatch.setattr(cli, 'student_factory', student)
    from repair import v0_review
    original = v0_review.WholeTransport
    client = WholeClient()
    monkeypatch.setattr(v0_review, 'WholeTransport', lambda *a, **kw: original(*a, **kw, client=client))
    return client


@pytest.mark.parametrize('k', [10, 20, 30])
def test_full_fake_round_build_and_resume(tmp_path, monkeypatch, k):
    root = tmp_path / 'round'
    args = arguments(root, k)
    cli.round_root(args)
    manifest = setup_synth(root, k)
    client = fake_runtime(monkeypatch, manifest)
    attempts = cli.run_attempts(args, root, manifest)
    assert len(attempts) == 4
    estimate = cli.review_dry_run(args, root, manifest, root / 'attempt')
    assert estimate['failures'] == 2 and estimate['teacher_calls'] == 0 and client.calls == 0
    assert estimate['review_output_cap'] == estimate['rationale_output_cap'] == 16384
    assert estimate['review_retry_output_cap'] == estimate['rationale_retry_output_cap'] == 32768
    assert estimate['maximum_review_output_tokens'] == 2 * 3 * 2 * (16384 + 32768)
    assert estimate['maximum_label_output_tokens'] == 2 * 40 * (16384 + 32768)
    assert estimate['projected_review_input_at_current_request_sizes'] == 12 * estimate['first_round_input_byte_bound']
    repairs = cli.run_reviews(args, root, manifest, root / 'attempt')
    assert len(repairs) == 2 and all(r['verified'] for r in repairs) and client.calls == 2
    label_calls = []
    if k != 20:
        args.action_only = False
        def labeler(*unused):
            def generate(prompt):
                label_calls.append(prompt)
                return json.dumps(dict(reasoning='Use public evidence.', evidence_steps=[-1], unsupported=False))
            return generate
        monkeypatch.setattr('repair.v0_review.make_rationale_labeler', labeler)
    rows = cli.run_labels(args, root, manifest, root / 'attempt', root / 'review')
    assert {r['v0']['bucket'] for r in rows} == {'R', 'S'}
    assert sum(r['v0']['edit_decision'] for r in rows) == 2
    if k != 20:
        assert len(label_calls) == 2
        assert all(r['evidence']['label_source'] == 'teacher_rationale' for r in rows if r['v0']['edit_decision'])
    args.replay = str(replay_bundle(tmp_path / 'continue', manifest.seeds))
    replay_before = artifacts(args.replay)
    built = cli.build_dataset(args, root, manifest, root / 'label',
        extra_inputs=cli.source_inputs(root / 'synth', root / 'attempt', root / 'review'))
    assert built['K'] == k and built['counts']['D']['episodes'] == k and built['counts']['G']['rows'] == 0
    assert read(root / 'round.json')['K'] == k
    assert artifacts(root / 'dataset/replay') == replay_before
    assert built['round_teacher_budget']['reported_usage']['output_tokens'] == 160
    _, data = load_dataset(root / 'dataset')
    assert len(data) == len(rows)
    required = {'arm', 'evidence', 'example_id', 'group_id', 'kind', 'messages', 'source',
                'supervised_assistant_indices', 'task_id', 'task_type', 'weight', 'v0'}
    assert all(set(r) == required for r in data)
    args.resume = True
    assert cli.run_attempts(args, root, manifest) == attempts
    assert cli.run_reviews(args, root, manifest, root / 'attempt') == repairs
    assert cli.run_labels(args, root, manifest, root / 'attempt', root / 'review') == rows
    cli.build_dataset(args, root, manifest, root / 'label',
        extra_inputs=cli.source_inputs(root / 'synth', root / 'attempt', root / 'review'))
    assert client.calls == 2


@pytest.mark.parametrize('new_items', [0, 1])
@pytest.mark.parametrize('k', [10, 20, 30])
def test_expand_new_dataset_never_writes_input_and_parent_is_seed(tmp_path, monkeypatch, new_items, k):
    root = tmp_path / 'round'
    args = arguments(root, k)
    cli.round_root(args)
    manifest = setup_synth(root, k)
    fake_runtime(monkeypatch, manifest)
    cli.run_attempts(args, root, manifest)
    cli.run_reviews(args, root, manifest, root / 'attempt')
    cli.run_labels(args, root, manifest, root / 'attempt', root / 'review')
    args.replay = str(replay_bundle(tmp_path / 'continue', manifest.seeds))
    cli.build_dataset(args, root, manifest, root / 'label',
        extra_inputs=cli.source_inputs(root / 'synth', root / 'attempt', root / 'review'))
    dataset = root / 'dataset'
    before = artifacts(dataset)
    from repair import v0_expand
    # Replace only the offline game-generation subprocess. All subsequent
    # attempt -> review -> label -> NEW-dataset steps run the actual driver.
    original_run = v0_expand.subprocess.run
    def generate_fake(command, **kwargs):
        if '-m' not in command or 'repair.v0_expand' not in command:
            return original_run(command, **kwargs)
        request = read(command[-1])
        assert request['plan']['requested_items'] == 1
        unit = request['plan']['units'][0]
        assert unit['preferred_seed'] in manifest.seeds
        row = dict(task_row('train/synth/new'), expansion_unit=unit,
            parent_seed_task_id=unit['preferred_seed'], asset_root=request['out'])
        write_json(Path(request['out']) / 'generated.json', [row] if new_items else [])
        write_json(Path(request['out']) / 'attempts.json', [row])
        write_json(Path(request['out']) / 'allocation_execution.json', [dict(unit=unit,
            status='filled' if new_items else 'unfilled_after_fallback')])
    monkeypatch.setattr(v0_expand.subprocess, 'run', generate_fake)
    seeds_path = tmp_path / 'seeds.json'
    write_json(seeds_path, {'fixture': True})
    args.dataset, args.variant = str(dataset), 'D-param'
    args.groups = args.weights = args.attempts = args.reviews = None
    args.seeds, args.data_root, args.main_checkout = str(seeds_path), str(tmp_path / 'games'), str(tmp_path / 'main')
    fake_runtime(monkeypatch, manifest, successful=True)
    result = expand(args, root, manifest)
    assert result['requested_items'] == 1
    assert result['generated_items'] == result['counts']['G']['episodes'] == new_items
    assert result['unfilled_count'] == 1 - new_items
    assert result['fallback_count'] == 0 and result['strict_fill'] is False
    if not new_items:
        assert read(root / 'expand-D-param/attempt/records.json') == []
    assert artifacts(dataset) == before
    new_manifest, rows = load_dataset(result['dataset'])
    assert new_manifest['K'] == k and new_manifest['counts']['D']['episodes'] == k
    assert {r['task_id'] for r in rows if r['v0']['bucket'] == 'G'} == ({'train/synth/new'} if new_items else set())
    assert (Path(result['dataset']) / 'replay/examples.jsonl').read_bytes() == (dataset / 'replay/examples.jsonl').read_bytes()


def test_expansion_generator_matches_edit_kind_and_fresh_fingerprint(tmp_path, monkeypatch):
    source = tmp_path / 'source'
    parent = factory.Seed(SEED, small_game(), ['finish'], {})
    path = source / SEED / 'game.tw-pddl'
    write_json(path, parent.game)
    parent.provenance = dict(game_path=str(path), game_sha256=file_hash(path))
    monkeypatch.setattr(factory, 'load_seeds', lambda *args: ([parent], {}))
    class Env(FakeEnv):
        def __init__(self, root, **kwargs):
            super().__init__()
            self.root = Path(root)
        def reset(self, task):
            result = super().reset(task)
            game = self.root / task / 'game.tw-pddl'
            if game.exists():
                result[0]['observation'] = 'Room.\n' + factory.task_text(read(game))
            return result
    restriction = dict(generator='G1', bin='near', destination_receptacle_type='Fridge')
    units = [dict(unit=i, group='category', source_episode_id='repair', source_synthetic_task_id=TASK,
                  preferred_seed=SEED, restriction=restriction) for i in range(2)]
    request = dict(seeds='unused', data_root=str(source), seed_task_ids=[SEED], seed=0,
        heldout=[], out=str(tmp_path / 'expand'), excluded_fingerprints=[],
        plan=dict(units=units, requested_items=2))
    before = artifacts(source)
    rows = generate(request, env_factory=Env)
    assert len(rows) == 2 and len({r['item_fingerprint'] for r in rows}) == 2
    assert all(r['parent_seed_task_id'] == SEED and edit_kind(r) == restriction for r in rows)
    assert artifacts(source) == before
    assert generate(request, env_factory=Env) == rows


def test_budget_fail_loud_blocks_pipeline_before_a_teacher_call(tmp_path, monkeypatch):
    root = tmp_path / 'round'
    args = arguments(root)
    cli.round_root(args)
    manifest = setup_synth(root)
    client = fake_runtime(monkeypatch, manifest)
    cli.run_attempts(args, root, manifest)
    budget = cli.budget_for(root)
    budget.reserve('used', CAPS['input_tokens'], CAPS['output_tokens'])
    budget.settle('used', dict(input_tokens=CAPS['input_tokens'], output_tokens=CAPS['output_tokens']))
    with pytest.raises(Exception):
        cli.run_reviews(args, root, manifest, root / 'attempt')
    assert client.calls == 0
    assert not (root / 'review/completion.json').exists()
