"""Concurrent L builds use fabricated evidence and the existing fake transport."""
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
import json
from pathlib import Path
import threading
import time
from types import SimpleNamespace

import pytest

from repair.runtime import CaseStopped
from repair.supervision import FakeLabelClient, InvalidExample, RationaleLabeler
from sed.storage import canonical_bytes, digest, file_hash, write_json
from teachers.openai_teacher import TeacherQuotaError
from tools import build_v2_dataset as builder
from test_supervision import K20, cli_args, dataset_inputs


def records(d, count):
    (d['run'] / 'records/test.json').unlink()
    paths = []
    for index in range(count):
        row = deepcopy(d['row'])
        row['run_key'] = f'{index:03d}'
        path = d['run'] / 'records' / f'{index:03d}.json'
        write_json(path, row)
        paths.append(path)
    return paths


class ExclusiveTokenizer:
    """Fail if any method, call or property overlaps another tokenizer use."""
    def __init__(self):
        self.busy = threading.Lock()
        self.calls = 0

    def encode(self, text, **kwargs):
        assert self.busy.acquire(blocking=False), 'concurrent tokenizer use'
        try:
            time.sleep(0.001)  # Release the GIL so an unprotected call can overlap.
            self.calls += 1
            return list(text.encode())
        finally:
            self.busy.release()

    __call__ = encode

    @property
    def vocab_size(self):
        return len(self.encode('abc'))


def comparable_manifest(output, relative):
    manifest = json.loads((output / relative).read_text())
    assert manifest['effective_config'].pop('label_workers') in (1, 8)
    # Audit bytes include timestamps, latency, output paths, content-addressed
    # blobs and scheduling-dependent ledger order. Check their actual hashes;
    # compare their label semantics and usage separately below.
    artifacts = manifest.pop('label_artifact_sha256')
    assert artifacts
    for name, expected in artifacts.items():
        assert file_hash(output / name) == expected
    if 'replay' in manifest:
        assert manifest['replay'].pop('manifest_sha256') == file_hash(output / 'replay/manifest.json')

    def normalize(value):
        if isinstance(value, dict):
            return {k: normalize(v) for k, v in value.items() if k not in ('created', 'timestamp')}
        if isinstance(value, list):
            return [normalize(v) for v in value]
        return value.replace(str(output), '<OUTPUT>') if isinstance(value, str) else value
    return normalize(manifest)


def test_worker_builds_match_sequential_and_keep_lazy_labelers(dataset_inputs, tmp_path, monkeypatch):
    d = dataset_inputs
    paths = records(d, 12)
    for path, update in zip(paths[-2:], ({'void': True}, {'mode': 'takeover'})):
        row = json.loads(path.read_text())
        row.update(update)
        write_json(path, row)
    expected_refs = [str(p) for p in paths[:-2] for _ in range(2)]
    constructed, seals, outputs = [], [], []
    context = threading.local()
    stats_lock = threading.Lock()
    active = peak = arrived = 0
    barrier = None
    original_create = FakeLabelClient.create
    original_seal = builder.seal_dataset

    class TrackedLabeler(RationaleLabeler):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            constructed.append(self.directory.name)
            context.record, context.counter = self.directory.name, 0

    def deterministic_uuid():
        # Stable per-record fake request IDs let manifests compare exactly even
        # when the real transport's UUID allocation would interleave differently.
        context.counter += 1
        return digest([context.record, context.counter])

    def overlapping_create(self, **kwargs):
        nonlocal active, peak, arrived
        with stats_lock:
            active += 1
            peak = max(peak, active)
            arrived += 1
            synchronize = barrier is not None and arrived <= 8
        try:
            if synchronize:
                barrier.wait(timeout=10)
            return original_create(self, **kwargs)
        finally:
            with stats_lock:
                active -= 1

    def capture_seal(directory, examples, **kwargs):
        if Path(directory).name != 'replay':
            seals.append([e['evidence']['episode_ref'] for e in examples])
        return original_seal(directory, examples, **kwargs)

    monkeypatch.setattr(builder, 'RationaleLabeler', TrackedLabeler)
    monkeypatch.setattr('teachers.openai_teacher.uuid', SimpleNamespace(uuid4=deterministic_uuid))
    monkeypatch.setattr(FakeLabelClient, 'create', overlapping_create)
    monkeypatch.setattr(builder, 'seal_dataset', capture_seal)
    tokenizer = ExclusiveTokenizer()
    monkeypatch.setattr(builder, 'local_tokenizer', lambda *a: (tokenizer, {'status': 'synthetic'}))

    def no_executor(*args, **kwargs):
        raise AssertionError('Single-worker builds must stay synchronous')

    for workers in (None, 1, 8):
        output = tmp_path / f'workers-{workers}'
        outputs.append(output)
        constructed.clear()
        active = peak = arrived = 0
        barrier = threading.Barrier(8) if workers == 8 else None
        monkeypatch.setattr(builder, 'ThreadPoolExecutor', ThreadPoolExecutor if workers == 8 else no_executor)
        args = cli_args(d, output) + ([] if workers is None else ['--label-workers', str(workers)])
        assert builder.main(args) == 0
        assert sorted(constructed) == sorted(digest(str(p)) for p in paths[:-2])
        assert peak == (8 if workers == 8 else 1)
        assert seals[-1] == expected_refs
        manifest = json.loads((output / 'manifest.json').read_text())
        assert manifest['effective_config']['label_workers'] == (workers or 1)
        # Existing usage_summary counts both completion and state-link receipts.
        assert manifest['synthetic_label_calls'] == 20
        assert manifest['global_teacher_budget']['reserved_attempts'] == 10
        assert manifest['global_teacher_budget']['reported_usage']['input_tokens'] == 1000
        assert manifest['global_teacher_budget']['reported_usage']['output_tokens'] == 300
        assert manifest['global_teacher_budget']['unsettled_attempts'] == 0
        assert manifest['excluded'] == [dict(reason='void_or_unverified_episode', count=1),
                                        dict(reason='wrong_mode', count=1)]
    assert tokenizer.calls > 0
    assert all(seal == expected_refs for seal in seals)
    for output in outputs[1:]:
        for name in ('examples.jsonl', 'examples.sha256', 'replay/examples.jsonl', 'replay/examples.sha256'):
            assert (output / name).read_bytes() == (outputs[0] / name).read_bytes()
        for name in ('manifest.json', 'replay/manifest.json'):
            assert comparable_manifest(output, name) == comparable_manifest(outputs[0], name)
        labels = lambda root: sorted(canonical_bytes(json.loads(line))
                                    for line in (root / 'labels.jsonl').read_text().splitlines())
        assert labels(output) == labels(outputs[0])


@pytest.mark.parametrize('error_type', [CaseStopped, TeacherQuotaError, InvalidExample])
def test_first_failure_stops_submission_waits_and_is_reraised(dataset_inputs, tmp_path, monkeypatch, error_type):
    paths = records(dataset_inputs, 12)
    failure = error_type('synthetic label failure')
    barrier, failed = threading.Barrier(4), threading.Event()
    submitted, finished = [], []
    first_four = {digest(str(path)): index for index, path in enumerate(paths[:4])}

    class ObservedExecutor(ThreadPoolExecutor):
        def submit(self, fn, *args, **kwargs):
            submitted.append(args[0])
            future = super().submit(fn, *args, **kwargs)
            def notify_failure(done):
                if done.exception() is failure:
                    failed.set()
            future.add_done_callback(notify_failure)
            return future

    class FailingLabeler(RationaleLabeler):
        def __call__(self, prompt):
            index = first_four[self.directory.name]
            barrier.wait(timeout=10)
            if index == 3:
                raise failure  # A later input fails while the first is blocked.
            assert failed.wait(timeout=10)
            try:
                if index == 0:
                    raise RuntimeError('later failure must not replace the first')
                return super().__call__(prompt)
            finally:
                finished.append(index)

    monkeypatch.setattr(builder, 'ThreadPoolExecutor', ObservedExecutor)
    monkeypatch.setattr(builder, 'RationaleLabeler', FailingLabeler)
    output = tmp_path / 'failed'
    with pytest.raises(error_type) as caught:
        builder.main(cli_args(dataset_inputs, output) + ['--label-workers', '4'])
    assert caught.value is failure
    assert submitted == [str(p) for p in paths[:4]]
    assert sorted(finished) == [0, 1, 2]  # build() waited for the active workers.
    assert not (output / 'manifest.json').exists()
    assert not (output / 'replay/manifest.json').exists()


def test_all_inputs_authorized_before_parallel_teacher_construction(dataset_inputs, tmp_path, monkeypatch):
    paths = records(dataset_inputs, 12)
    row = json.loads(paths[-1].read_text())
    row['task_id'] = K20
    write_json(paths[-1], row)
    def forbidden(*args, **kwargs):
        raise AssertionError('teacher constructed before all inputs authorized')
    monkeypatch.setattr(builder, 'RationaleLabeler', forbidden)
    output = tmp_path / 'unauthorized'
    with pytest.raises(InvalidExample, match='task_outside_frozen_view'):
        builder.main(cli_args(dataset_inputs, output) + ['--label-workers', '8'])
    assert not output.exists()


@pytest.mark.parametrize('value', ['0', '-1'])
def test_nonpositive_label_workers_rejected_before_io(tmp_path, value):
    output = tmp_path / 'invalid-workers'
    with pytest.raises(ValueError, match='--label-workers must be >= 1'):
        builder.main(['build', '--arm', 'JOINT', '--inputs', 'nonexistent',
                      '--output', str(output), '--dry-run', '--label-workers', value])
    assert not output.exists()


def test_tokenizer_proxy_serializes_calls_methods_and_properties():
    raw = ExclusiveTokenizer()
    tokenizer = builder.LockedTokenizer(raw)
    barrier = threading.Barrier(8)
    def use(index):
        barrier.wait(timeout=10)
        if index % 3 == 0:
            return tokenizer('abc')
        if index % 3 == 1:
            return tokenizer.encode('abc', add_special_tokens=False)
        return tokenizer.vocab_size
    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(use, range(8)))
    assert results == [3 if i % 3 == 2 else list(b'abc') for i in range(8)]
    assert raw.calls == 8
