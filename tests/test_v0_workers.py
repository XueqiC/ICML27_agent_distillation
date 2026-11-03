"""Real spawn pools, fake models/environments, and fixed audit clocks/IDs.

Byte comparisons use the SAME paths and fixed audit metadata: real timestamps,
UUIDs and latency already differ between two sequential runs. No production
evidence is normalized or removed to obtain the equality asserted here.
"""
from contextlib import contextmanager
from functools import partial
import itertools
import json
import os
from pathlib import Path
import shutil
import socket
import threading
import time
from types import SimpleNamespace

import pytest

from repair.dry_fakes import FakeEnvironment, N, O
from repair.joint import FakeStudent
from repair.takeover import SharedTeacherBudget
from repair.v0_common import SyntheticManifest, completed, read
from sed.storage import digest, write_json
from tools import v0_round as cli
from test_v0_pipeline import arguments, fake_adapter, WholeClient
from test_v0_round import manifest_at, task_row

ATTEMPT_UNIT, REVIEW_UNIT = cli._attempt_unit, cli._review_unit
AUDIT = threading.local()


def audit_uuid():
    return digest([AUDIT.key, next(AUDIT.counter)])[:32]


def fixed_time():
    return '2000-01-01T00:00:00Z'


def forbidden(*args, **kwargs):
    raise AssertionError('Network/API access forbidden in spawned fake worker')


@contextmanager
def fixed_audit(key):
    """Set up every spawned process explicitly; pytest fixtures do not cross spawn."""
    import repair.execute
    import repair.ledger
    import repair.takeover
    import sed.provenance
    import teachers.capped
    import teachers.openai_teacher
    from teachers.rate_limit import RateLimiter
    with pytest.MonkeyPatch.context() as patch:
        patch.setenv('CUDA_VISIBLE_DEVICES', '')
        patch.delenv('OPENAI_API_KEY', raising=False)
        patch.delenv('TEACHER_RATE_LIMIT_DIR', raising=False)
        patch.setenv('SED_SECRETS_FILE', '/tmp/v0-workers-no-secret.env')
        for name in ('connect', 'connect_ex', 'sendto'):
            patch.setattr(socket.socket, name, forbidden)
        patch.setattr(socket, 'create_connection', forbidden)
        patch.setattr(socket, 'getaddrinfo', forbidden)
        from openai.resources.responses.responses import Responses
        patch.setattr(Responses, 'create', forbidden)
        for module in (repair.execute, repair.ledger, repair.takeover, teachers.openai_teacher):
            patch.setattr(module, 'uuid', SimpleNamespace(uuid4=audit_uuid))
        for module in (repair.ledger, repair.takeover, sed.provenance, teachers.capped,
                       teachers.openai_teacher):
            patch.setattr(module, 'utc_now', fixed_time)
        patch.setattr('repair.v0_common.utc_now', fixed_time)
        patch.setattr(teachers.openai_teacher, 'time', SimpleNamespace(monotonic=lambda: 0.))
        original_init = RateLimiter.__init__
        def initialize(self, *args, **kwargs):
            original_init(self, *args, **dict(kwargs, clock=lambda: 1.))
        patch.setattr(RateLimiter, '__init__', initialize)
        AUDIT.key, AUDIT.counter = key, itertools.count()
        yield patch


def fake_student(spec, tid, decoding, seed, log_dir):
    assert spec['endpoint'] == 'http://fake-served-endpoint/v1'
    marker = Path(spec['test_markers'])
    # A durable construction trace proves completed units are not run on resume.
    def create():
        marker.mkdir(exist_ok=True)
        with (marker / (digest(str(log_dir)) + '.txt')).open('a') as handle:
            handle.write(str(os.getpid()) + '\n')
        if spec.get('test_failure') == seed and spec.get('inject_failure'):
            # Let another process complete first, then retain a sanitized error.
            time.sleep(.3)
            raise RuntimeError('PRIVATE_FAKE_EXCEPTION_DO_NOT_PERSIST')
        actions = N if seed == 2 else O + ['look']
        if 'failures' in log_dir.parts:
            actions = N[3:] if spec.get('test_mode') == 'handback' else ['look']
        return FakeStudent(replies=['Action: ' + action for action in actions])
    return create


def fake_attempt_unit(**job):
    key = job['directory'].name + ('-retry' if (job['directory'] / 'worker_error.json').exists() else '')
    with fixed_audit(key) as patch:
        patch.setattr(SyntheticManifest, 'environment', lambda *a, **kw: FakeEnvironment())
        patch.setattr(cli, 'student_factory', fake_student)
        # Force out-of-order completion without changing any response/evidence.
        time.sleep(.03 if job['seed'] is None else .001)
        return ATTEMPT_UNIT(**job)


class ReviewClient(WholeClient):
    def __init__(self, mode, budget_path, markers):
        super().__init__()
        self.mode, self.budget_path, self.markers = mode, Path(budget_path), Path(markers)

    def create(self, **kwargs):
        self.markers.mkdir(exist_ok=True)
        (self.markers / AUDIT.key).write_text(str(os.getpid()))
        if self.mode == 'timeout' and AUDIT.key == self.truncation_target:
            raise TimeoutError('PRIVATE_FAKE_TIMEOUT')
        if self.mode == 'cap_race':
            # One admitted transport remains in flight while another reservation
            # hits the global cap. The admitted call must still be settled.
            deadline = time.monotonic() + 15
            while time.monotonic() < deadline:
                if json.loads(self.budget_path.read_text().splitlines()[-1])['stop_reason']:
                    break
                time.sleep(.01)
            else:
                raise AssertionError('No concurrent cap stop')
        response = super().create(**kwargs)
        result = response.parse()
        if self.mode == 'truncated_twice' and AUDIT.key == self.truncation_target:
            result.output_text, result.status = '', 'incomplete'
            result.incomplete_details = {'reason': 'max_output_tokens'}
            result.usage.update(output_tokens=kwargs['max_output_tokens'],
                output_tokens_details={'reasoning_tokens': kwargs['max_output_tokens']})
        if self.mode in ('handback', 'three_rounds'):
            value = json.loads(result.output_text)
            value['patch']['blocks'][0]['actions'] = ['look']
            result.output_text = json.dumps(value)
        return response


def fake_review_unit(**job):
    from repair import v0_review
    transport = v0_review.WholeTransport
    spec = job['spec']
    if spec.get('test_mode') == 'cap_race' and job['attempt']['attempt_id'] != spec['admit']:
        # Ensure the admitted worker has reached transport, rather than stopping
        # between its budget reservation and its pre-transport quota check.
        deadline = time.monotonic() + 15
        while not (Path(spec['test_markers']) / spec['admit']).exists():
            assert time.monotonic() < deadline, 'Admitted transport did not start'
            time.sleep(.01)
    with fixed_audit(job['directory'].name) as patch:
        patch.setattr(SyntheticManifest, 'environment', lambda *a, **kw: FakeEnvironment())
        patch.setattr(cli, 'student_factory', fake_student)
        client = ReviewClient(job['spec'].get('test_mode'), job['budget'].path,
                              job['spec']['test_markers'])
        client.truncation_target = spec.get('truncation_target')
        def make_transport(*a, **kw):
            if spec.get('review_failure') == job['attempt']['attempt_id']:
                time.sleep(.5)
                raise RuntimeError('PRIVATE_FAKE_EXCEPTION_DO_NOT_PERSIST')
            result = transport(*a, **kw, client=client)
            result.teacher.teacher.sleep = lambda seconds: None
            result.teacher.teacher.jitter = lambda lo, hi: 0
            return result
        patch.setattr(v0_review, 'WholeTransport', make_transport)
        return REVIEW_UNIT(**job)


@pytest.fixture
def runtime(tmp_path, monkeypatch):
    root = tmp_path / 'round'
    args = arguments(root, workers=1, label_workers=1)
    cli.round_root(args)
    manifest = manifest_at(root, [task_row('train/synth/' + name, probe=probe)
        for probe, names in ((False, ['z', 'a']), (True, ['pz', 'pa'])) for name in names])
    spec = dict(endpoint='http://fake-served-endpoint/v1', test_markers=str(tmp_path / 'markers'))
    monkeypatch.setattr(cli, 'serving', lambda *a: (dict(spec), fake_adapter(), []))
    monkeypatch.setattr(cli, '_attempt_unit', fake_attempt_unit)
    monkeypatch.setattr(cli, '_review_unit', fake_review_unit)
    with fixed_audit('parent'):
        yield args, root, manifest, spec


def contents(directory):
    return {str(p.relative_to(directory)): p.read_bytes() for p in directory.rglob('*')
            if p.is_file() and not p.name.endswith('.lock')}


def reset_budget(root):
    (root / 'teacher_budget.jsonl').unlink(missing_ok=True)
    shutil.rmtree(root / 'joint_teacher_rate_limit', ignore_errors=True)


@pytest.mark.parametrize('probes', [False, True])
def test_attempt_probe_worker_count_identical_records_and_seals(runtime, probes):
    args, root, manifest, spec = runtime
    directory = root / ('probe_attempt' if probes else 'attempt')
    sequential = cli.run_attempts(args, root, manifest, probes=probes)
    expected = contents(directory)
    shutil.rmtree(directory)
    shutil.rmtree(spec['test_markers'])
    args.workers = 4
    assert cli.run_attempts(args, root, manifest, probes=probes) == sequential
    assert contents(directory) == expected
    assert completed(directory)
    pids = {p.read_text().strip() for p in Path(spec['test_markers']).iterdir()}
    assert len(pids) > 1 and str(os.getpid()) not in pids
    assert [(r['task_id'], r['decoding'], r['sampling_seed']) for r in sequential] == [
        (tid, decoding, seed) for tid, task in manifest.tasks.items() if task['heldout_probe'] == probes
        for decoding, seed in cli.DECODINGS]


@pytest.mark.parametrize('mode', ['suffix', 'handback', 'three_rounds', 'truncated_twice', 'timeout'])
def test_review_worker_count_identical_records_and_seals(runtime, mode):
    args, root, manifest, spec = runtime
    spec['test_mode'] = mode
    attempts = cli.run_attempts(args, root, manifest)
    if mode in ('truncated_twice', 'timeout'):
        from repair.v0_execution import selected_failures
        spec['truncation_target'] = selected_failures(attempts)[0]['attempt_id']
    sequential = cli.run_reviews(args, root, manifest, root / 'attempt')
    expected = contents(root / 'review')
    assert len(sequential) == 4
    assert all(len(r['rounds']) == (3 if mode == 'three_rounds' else 1) for r in sequential)
    if mode == 'handback':
        assert all('handback-student' in r['executors'] for r in sequential)
    if mode in ('truncated_twice', 'timeout'):
        assert sequential[0]['rounds'][0]['status'] == ('teacher_timeout' if mode == 'timeout' else 'teacher_output_truncated')
        assert [r['verified'] for r in sequential] == [False, True, True, True]
    budget = cli.budget_for(root).summary()
    shutil.rmtree(root / 'review')
    shutil.rmtree(spec['test_markers'])
    reset_budget(root)
    args.workers = 4
    assert cli.run_reviews(args, root, manifest, root / 'attempt') == sequential
    assert contents(root / 'review') == expected
    assert completed(root / 'review')
    assert cli.budget_for(root).summary() == budget
    limiter = read(next((root / 'joint_teacher_rate_limit').glob('*.json')))
    assert len(limiter['reservations']) == budget['reserved_attempts']
    pids = {p.read_text().strip() for p in Path(spec['test_markers']).iterdir() if p.suffix != '.txt'}
    assert len(pids) > 1 and str(os.getpid()) not in pids


@pytest.mark.parametrize('phase_name', ['attempt', 'probe', 'review'])
def test_resume_partial_parallel_phase_skips_completed_units(runtime, phase_name):
    args, root, manifest, spec = runtime
    args.workers = 4
    if phase_name == 'review':
        cli.run_attempts(args, root, manifest)
        run = lambda: cli.run_reviews(args, root, manifest, root / 'attempt')
        directory, unit_dir = root / 'review', 'failures'
    else:
        run = lambda: cli.run_attempts(args, root, manifest, probes=phase_name == 'probe')
        directory = root / ('probe_attempt' if phase_name == 'probe' else 'attempt')
        unit_dir = 'episodes'
    expected = run()
    # Simulate a process interruption after all unit records, before publication.
    for name in ('completion.json', 'records.json'):
        (directory / name).unlink()
    before = contents(directory / unit_dir)
    constructions = contents(Path(spec['test_markers']))
    args.resume, args.workers = True, 1
    assert run() == expected
    assert contents(directory / unit_dir) == before
    assert contents(Path(spec['test_markers'])) == constructions
    assert completed(directory)


def test_parallel_attempt_exception_retains_evidence_and_can_resume(runtime, monkeypatch):
    args, root, manifest, spec = runtime
    args.workers = 4
    spec.update(inject_failure=True, test_failure=0)
    with pytest.raises(RuntimeError):
        cli.run_attempts(args, root, manifest)
    directory = root / 'attempt'
    assert not (directory / 'completion.json').exists()
    assert not (directory / 'records.json').exists()
    retained = list((directory / 'episodes').glob('*/record.json'))
    assert retained
    errors = list((directory / 'episodes').glob('*/worker_error.json'))
    assert errors and all(read(p)['error_type'] == 'RuntimeError' for p in errors)
    assert all(b'PRIVATE_FAKE_EXCEPTION' not in data for data in contents(directory).values())
    old = {p: p.read_bytes() for p in retained}
    # Clear the injected fault without changing the bound serving configuration.
    # In a real run the transient server/factory recovers at the same endpoint.
    original = cli._attempt_unit
    def recovered(**job):
        job['spec'] = dict(job['spec'], inject_failure=False)
        return original(**job)
    monkeypatch.setattr(cli, '_attempt_unit', recovered)
    args.resume, args.workers = True, 1
    rows = cli.run_attempts(args, root, manifest)
    assert len(rows) == 8 and completed(directory)
    assert all(p.read_bytes() == data for p, data in old.items())


def test_shared_review_cap_stops_all_workers_and_settles_inflight(runtime, monkeypatch):
    from repair.v0_execution import selected_failures
    args, root, manifest, spec = runtime
    spec['test_mode'] = 'cap_race'
    attempts = cli.run_attempts(args, root, manifest)
    spec['admit'] = selected_failures(attempts)[0]['attempt_id']
    from repair.v0_review import OUTPUT_CAP
    caps = dict(input_tokens=1_000_000, output_tokens=OUTPUT_CAP)
    budget = SharedTeacherBudget(root / 'teacher_budget.jsonl', **caps)
    monkeypatch.setattr(cli, 'budget_for', lambda root: budget)
    args.workers = 4
    with pytest.raises(Exception):
        cli.run_reviews(args, root, manifest, root / 'attempt')
    summary = budget.summary()
    assert summary['stop_reason'] == 'global_output_tokens_cap'
    assert summary['reserved_attempts'] == 1 and summary['unsettled_attempts'] == 0
    assert summary['reported_usage']['output_tokens'] == 80
    assert len([p for p in Path(spec['test_markers']).iterdir() if p.suffix != '.txt']) == 1
    assert not (root / 'review/completion.json').exists()
    with pytest.raises(Exception):
        budget.reserve('after-stop', 1, 1)


def test_review_worker_exception_retains_partial_records_and_resumes(runtime, monkeypatch):
    from repair.v0_execution import selected_failures
    args, root, manifest, spec = runtime
    attempts = cli.run_attempts(args, root, manifest)
    spec['review_failure'] = selected_failures(attempts)[0]['attempt_id']
    args.workers = 4
    with pytest.raises(RuntimeError):
        cli.run_reviews(args, root, manifest, root / 'attempt')
    directory = root / 'review'
    retained = {p: p.read_bytes() for p in directory.glob('failures/*/record.json')}
    assert retained and len(retained) < 4
    assert not (directory / 'completion.json').exists()
    assert all(b'PRIVATE_FAKE_EXCEPTION' not in value for value in contents(directory).values())
    original = cli._review_unit
    def recovered(**job):
        job['spec'] = dict(job['spec'], review_failure=None)
        return original(**job)
    monkeypatch.setattr(cli, '_review_unit', recovered)
    args.resume, args.workers = True, 1
    assert len(cli.run_reviews(args, root, manifest, root / 'attempt')) == 4
    assert all(p.read_bytes() == data for p, data in retained.items())
    assert cli.budget_for(root).summary()['reserved_attempts'] == 4
    assert completed(directory)


@pytest.mark.parametrize('action_only', [False, True])
def test_label_worker_count_identical_rows_seals_and_resume(runtime, monkeypatch, action_only):
    from repair.supervision import FakeLabelClient
    from repair import v0_review
    from test_build_v2_label_workers import ExclusiveTokenizer
    args, root, manifest, spec = runtime
    cli.run_attempts(args, root, manifest)
    cli.run_reviews(args, root, manifest, root / 'attempt')
    # Labels use real capped transport with a fake response and thread-local IDs.
    labeler_factory = v0_review.make_rationale_labeler
    constructed = []
    barrier = None
    class Client(FakeLabelClient):
        def create(self, **kw):
            if barrier is not None:
                barrier.wait(timeout=15)
            return super().create(**kw)
    def labeler(config, tid, directory, budget):
        AUDIT.key, AUDIT.counter = directory.parent.name, itertools.count()
        constructed.append(directory.parent.name)
        return labeler_factory(config, tid, directory, budget, client=Client())
    monkeypatch.setattr(v0_review, 'make_rationale_labeler', labeler)
    tokenizer = ExclusiveTokenizer()
    monkeypatch.setattr('repair.supervision.ByteBoundTokenizer', lambda: tokenizer)
    args.action_only = action_only
    # Keep the initial review usage ledger, exactly, for each label run.
    budget_before = (root / 'teacher_budget.jsonl').read_bytes()
    sequential = cli.run_labels(args, root, manifest, root / 'attempt', root / 'review')
    expected = contents(root / 'label')
    usage = cli.budget_for(root).summary()
    shutil.rmtree(root / 'label')
    reset_budget(root)
    (root / 'teacher_budget.jsonl').write_bytes(budget_before)
    args.label_workers = 4
    barrier = threading.Barrier(4)
    assert cli.run_labels(args, root, manifest, root / 'attempt', root / 'review') == sequential
    assert contents(root / 'label') == expected
    assert cli.budget_for(root).summary() == usage
    assert (tokenizer.calls == 0) if action_only else (tokenizer.calls > 0)
    assert completed(root / 'label')
    # Incomplete phase, durable episode rows: no duplicate label calls on resume.
    (root / 'label/completion.json').unlink()
    (root / 'label/rows.json').unlink()
    calls = len(constructed)
    args.resume, args.label_workers = True, 1
    assert cli.run_labels(args, root, manifest, root / 'attempt', root / 'review') == sequential
    assert len(constructed) == calls


def test_label_exception_fail_fast_without_seal(runtime, monkeypatch):
    args, root, manifest, spec = runtime
    cli.run_attempts(args, root, manifest)
    cli.run_reviews(args, root, manifest, root / 'attempt')
    args.label_workers = 4
    from repair import v0_data
    original = v0_data.decision_rows
    calls, lock = [], threading.Lock()
    def fail(episode, *a, **kw):
        with lock:
            calls.append(episode['episode_id'])
            first = len(calls) == 1
        if first:
            write_json(Path(kw['episode_ref']).parent / 'fake_failure.json', {'retained': True})
            raise RuntimeError('fake label failure')
        return original(episode, *a, **kw)
    monkeypatch.setattr(v0_data, 'decision_rows', fail)
    with pytest.raises(RuntimeError, match='fake label failure'):
        cli.run_labels(args, root, manifest, root / 'attempt', root / 'review')
    assert len(calls) <= 4
    assert list((root / 'label').rglob('fake_failure.json'))
    assert list((root / 'label').rglob('worker_error.json'))
    assert not (root / 'label/completion.json').exists()
    assert not (root / 'label/rows.json').exists()


@pytest.mark.parametrize('name', ['workers', 'label_workers'])
@pytest.mark.parametrize('value', [0, -1])
def test_worker_counts_reject_nonpositive(name, value):
    with pytest.raises(ValueError, match='must be >= 1'):
        cli.worker_count(SimpleNamespace(**{name: value}), name)


def test_cli_worker_defaults_and_explicit_values(tmp_path, monkeypatch):
    captured = []
    monkeypatch.setattr(cli, 'round_root', lambda args: captured.append(args) or tmp_path)
    def stop(*a):
        raise RuntimeError('parsed')
    monkeypatch.setattr(cli, 'manifest_for', stop)
    for command, flag in [('attempt', '--workers'), ('probe', '--workers'), ('review', '--workers'),
                          ('label', '--label-workers'), ('build', '--label-workers'), ('expand', '--workers')]:
        extra = ['--seeds', 's', '--data-root', 'd', '--main-checkout', 'm', '--dataset', 'x', '--variant', 'D-param'] if command == 'expand' else []
        for workers in (None, 4):
            with pytest.raises(RuntimeError, match='parsed'):
                cli.main([command, '--output', str(tmp_path), *extra,
                          *([] if workers is None else [flag, str(workers)])])
            assert getattr(captured[-1], flag[2:].replace('-', '_')) == (workers or 1)
