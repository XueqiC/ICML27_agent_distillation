"""Build and seal E0.4 v2-train-interchange/1 datasets and separate K20 replay."""
import argparse
from collections import Counter, deque
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from hashlib import sha256
import json
from pathlib import Path
import subprocess
import sys
import threading

# Support both the requested script command and python -m tools.build_v2_dataset.
ROOT = Path(__file__).resolve().parents[1]
if __package__ in (None, ''):
    sys.path[:0] = [str(ROOT), str(ROOT / 'src')]

from repair.artifacts import checked_document, read_json
from repair.supervision import (ARMS, SCHEMA, InvalidExample, RationaleLabeler, SplitPolicy,
    build_decisions, build_trajectory, seal_dataset)
from repair.takeover import SharedTeacherBudget
from sed.config import load_config
from sed.provenance import utc_now, versions, training_support_manifest
from sed.seed_evidence import checked_episode
from sed.storage import canonical_bytes, digest, file_hash, JsonlLedger, write_json

THETA0_NAME = 'e4b-tsft-k20-e32-lr5e-5'
LABEL_CODE_PATHS = ('repair/supervision.py', 'repair/prompts.py', 'repair/labels.py',
                    'repair/facts.py', 'repair/schemas.py', 'src/sed/react.py', 'src/sed/visibility.py')
LABEL_CONFIG = dict(rationale_template='repair/prompts.py:RATIONALE', label_token_cap=64,
                    label_output_cap_including_reasoning=1024)


class LabelReuseError(ValueError):
    """Sanitized reuse failures, safe to show at the CLI boundary."""


class LabelReuseSource:
    """Verify a sealed source before output creation and index completed calls."""
    def __init__(self, directory, *, dry_run, splits_sha256, theta0, teacher):
        self.directory = Path(directory).resolve()
        manifest_path = self.directory / 'manifest.json'
        if not manifest_path.is_file():
            raise LabelReuseError('Label reuse requires a sealed source manifest.json')
        try:
            content = manifest_path.read_bytes()
            manifest = json.loads(content)
        except (OSError, ValueError):
            raise LabelReuseError('Label reuse source manifest.json is unreadable or invalid') from None
        if not isinstance(manifest, dict):
            raise LabelReuseError('Label reuse source manifest.json must be an object')
        self.manifest_sha256 = sha256(content).hexdigest()
        expected = dict(schema=SCHEMA, splits_sha256=splits_sha256, theta0=theta0,
                        teacher_alias=teacher['provider_model_id'], dry_run=dry_run)
        for field, value in expected.items():
            if field not in manifest or manifest[field] != value:
                raise LabelReuseError(f'Label reuse source {field} mismatch')
        if manifest.get('arm') not in ARMS[:2]:
            raise LabelReuseError('Label reuse source arm must be JOINT or JOINT_COMPLETE_ONLY')
        if not dry_run and manifest.get('training_eligible') is not True:
            raise LabelReuseError('Real label reuse requires a training_eligible source')
        effective = manifest.get('effective_config')
        for field, value in dict(teacher=teacher, **LABEL_CONFIG).items():
            if not isinstance(effective, dict) or effective.get(field) != value:
                raise LabelReuseError(f'Label reuse source effective_config.{field} mismatch')
        code = manifest.get('code_sha256')
        for name in LABEL_CODE_PATHS:
            if not isinstance(code, dict) or code.get(name) != file_hash(ROOT / name):
                raise LabelReuseError(f'Label reuse source code_sha256 mismatch: {name}')
        artifacts = manifest.get('label_artifact_sha256')
        if not isinstance(artifacts, dict):
            raise LabelReuseError('Label reuse source has no label_artifact_sha256 mapping')
        paths = sorted((self.directory / 'labels').glob('*/requests.jsonl'))
        names = {str(p.relative_to(self.directory)) for p in paths}
        recorded = {name for name in artifacts if len(Path(name).parts) == 3
                    and Path(name).parts[0] == 'labels' and Path(name).name == 'requests.jsonl'}
        if names != recorded:
            raise LabelReuseError('Label reuse source requests.jsonl files differ from label_artifact_sha256')
        self.entries, self.reused_calls, self.total_calls = {}, 0, 0
        self.lock = threading.Lock()
        for path in paths:
            name = str(path.relative_to(self.directory))
            try:
                content = path.read_bytes()
            except OSError:
                raise LabelReuseError(f'Label reuse source requests file is unreadable: {name}') from None
            if sha256(content).hexdigest() != artifacts[name]:
                raise LabelReuseError(f'Label reuse source label_artifact_sha256 mismatch: {name}')
            for line, raw_row in enumerate(content.splitlines(), 1):
                try:
                    row = json.loads(raw_row)
                    if row['status'] != 'completed':
                        continue
                    if not isinstance(row['request'], dict) or not isinstance(row['raw_output'], str):
                        raise ValueError
                    key = (path.parent.name, canonical_bytes(row['request']))
                except (ValueError, KeyError, TypeError):
                    raise LabelReuseError(f'Label reuse source invalid request row: {name}:{line}') from None
                self.entries.setdefault(key, deque()).append((row['raw_output'], name, line))
                self.total_calls += 1
        try:
            self.examples_sha256 = file_hash(self.directory / 'examples.jsonl')
        except OSError:
            raise LabelReuseError('Label reuse source examples.jsonl is missing or unreadable') from None

    def summary(self):
        with self.lock:
            return dict(source_dataset=str(self.directory), source_manifest_sha256=self.manifest_sha256,
                        source_examples_sha256=self.examples_sha256, reused_calls=self.reused_calls,
                        unused_source_calls=self.total_calls - self.reused_calls)


class ReusedLabeler:
    """Consume each matching source completion once and retain its exact origin."""
    def __init__(self, source, directory):
        self.source, self.directory = source, Path(directory)
        self.ledger = JsonlLedger(self.directory / 'requests.jsonl')

    def __call__(self, prompt):
        request = prompt.to_json()
        key = (self.directory.name, canonical_bytes(request))
        with self.source.lock:
            queue = self.source.entries.get(key)
            if not queue:
                raise LabelReuseError('L halted: no remaining reused label for '
                                      f'{self.directory.name}; request_sha256={digest(request)}')
            raw_output, requests_file, line = queue.popleft()
            self.ledger.append(dict(timestamp=utc_now(), request=request, status='reused',
                raw_output=raw_output, source=dict(dataset=str(self.source.directory),
                                                   requests_file=requests_file, line=line)))
            self.source.reused_calls += 1
            return raw_output


class LockedTokenizer:
    """Serialize tokenizer calls, methods and property reads on one lock."""
    def __init__(self, tokenizer):
        self._tokenizer, self._lock = tokenizer, threading.Lock()

    def __call__(self, *args, **kwargs):
        with self._lock:
            return self._tokenizer(*args, **kwargs)

    def __getattr__(self, name):
        with self._lock:
            value = getattr(self._tokenizer, name)
        if not callable(value):
            return value
        def call(*args, **kwargs):
            with self._lock:
                return value(*args, **kwargs)
        return call


def label_results(records, build_record, workers):
    """Bound in-flight records, fail fast, and yield in original input order."""
    if workers == 1:
        for ref, record in records:
            yield build_record(ref, record)
        return
    lock, failures = threading.Lock(), []
    def run(ref, record):
        try:
            return build_record(ref, record)
        except BaseException as exc:
            # Coordinate with submission, including failures during initial fill.
            with lock:
                if not failures:
                    failures.append(exc)
            raise
    results, pending = [None] * len(records), {}
    remaining = iter(enumerate(records))
    exhausted = False
    with ThreadPoolExecutor(max_workers=workers) as pool:
        try:
            while pending or not exhausted:
                while not exhausted and len(pending) < workers:
                    with lock:
                        if failures:
                            raise failures[0]
                        item = next(remaining, None)
                        if item is None:
                            exhausted = True
                        else:
                            index, (ref, record) = item
                            pending[pool.submit(run, ref, record)] = index
                if pending:
                    done, _ = wait(pending, return_when=FIRST_COMPLETED)
                    with lock:
                        if failures:
                            raise failures[0]
                        for future in done:
                            results[pending.pop(future)] = future.result()
        finally:
            # Cancel unstarted records; the executor waits for active calls even
            # on failure. The original exception escapes before any sealing.
            for future in pending:
                future.cancel()
    yield from results


def read_bound(path, inputs, expected=None):
    path = Path(path).resolve()
    actual = file_hash(path)
    if expected is not None and actual != expected:
        raise InvalidExample('input_hash_mismatch')
    if str(path) in inputs and inputs[str(path)] != actual:
        raise InvalidExample('input_changed_during_build')
    inputs[str(path)] = actual
    return read_json(path)


def frozen_collection(splits, root, inputs):
    """Check the frozen collection and K20 view without rescanning other tasks."""
    original = Path(splits['rebuild_arguments']['verified_root'])
    def frozen(relative):
        key = str(original / relative)
        expected = splits['input_sha256'].get(key)
        if expected is None:
            raise InvalidExample('collection_not_in_split_freeze')
        return read_bound(root / relative, inputs, expected)
    collection = frozen('manifest.json')
    replay = frozen('subsets/k20/manifest.json')
    selection = frozen('subsets/k20/selection.json')
    identity = read_bound(root / 'subsets/k20/identity.json', inputs)
    if (collection['status'] != 'complete' or replay['status'] != 'complete' or
            replay['K'] != 20 or len(replay['verified']) != 20 or selection['K'] != 20 or
            replay['selection_sha256'] != digest(selection) or replay['identity_sha256'] != digest(identity) or
            identity['selection'] != selection or replay['source_collection']['manifest_sha256'] != digest(collection) or
            [r['task_id'] for r in replay['verified']] != selection['task_order'] or
            len(set(selection['task_order'])) != 20):
        raise InvalidExample('k20_collection_identity')
    receipts = {digest(r): r for r in collection['verified']}
    if any(digest(r) not in receipts for r in replay['verified']):
        raise InvalidExample('k20_receipt_not_in_source')
    return collection, replay, selection, receipts


def load_teacher_episode(root, receipt, inputs):
    directory = (root / receipt['episode_dir']).resolve()
    if not directory.is_relative_to(root) or receipt.get('won') is not True:
        raise InvalidExample('invalid_teacher_receipt')
    episode = checked_episode(root, receipt)
    if episode['task_id'] != receipt['task_id']:
        raise InvalidExample('teacher_episode_identity')
    # Includes raw event and audit-blob hashes, retained exclusively as evidence.
    for relative, expected in receipt['episode_hashes'].items():
        path = (directory / relative).resolve()
        if not path.is_relative_to(directory) or file_hash(path) != expected:
            raise InvalidExample('teacher_evidence_hash')
        inputs[str(path)] = expected
    return episode, str(directory / 'episode.json')


def theta0_identity(project, path, replay_root, inputs, config):
    if path is None:
        candidates = [project / 'reports/runs' / runs / THETA0_NAME / 'stages/train-sft/attempt-000/output'
                      for runs in ('baselines_loni', 'baselines')]
        path = next((p for p in candidates if (p / 'checkpoint/training_state.json').is_file()), None)
        if path is None:
            raise ValueError('Supply --theta0-training-dir with theta0 training provenance and checkpoint')
    path = Path(path).resolve()
    provenance = read_bound(path / 'provenance.json', inputs)
    state = read_bound(path / 'checkpoint/training_state.json', inputs)
    if provenance['config']['models']['student'] != config['models']['student']:
        raise InvalidExample('theta0_student_mismatch')
    support = training_support_manifest(provenance)
    if provenance['inputs'].get(str(support)) != file_hash(replay_root / 'manifest.json'):
        raise InvalidExample('theta0_k20_support_mismatch')
    if state.get('N_examples') != 20 or state.get('final') is not True:
        raise InvalidExample('theta0_checkpoint_not_final_k20')
    checkpoint = path / 'checkpoint'
    for name in ('adapter_config.json', 'adapter_model.safetensors'):
        expected = state['artifact_sha256'][name]
        if file_hash(checkpoint / name) != expected:
            raise InvalidExample('theta0_checkpoint_hash')
        inputs[str(checkpoint / name)] = expected
    return dict(adapter_path=str(checkpoint), adapter_weights_sha256=state['artifact_sha256']['adapter_model.safetensors']), provenance


def local_tokenizer(args, provenance, inputs):
    """Local files only, same pinned snapshot metadata; no model weights loaded."""
    training = provenance['config'].get('baseline_training', provenance['config'].get('sft', {}))
    recorded = training.get('base_snapshot', {})
    model = provenance['config']['models']['student']
    candidates = [Path(args.snapshot)] if args.snapshot else [
        Path(training.get('snapshot', '/nonexistent')),
        Path.home() / '.cache/huggingface/hub' / ('models--' + model['requested_repo'].replace('/', '--')) /
        'snapshots' / model['revision']]
    snapshot = next((p for p in candidates if (p / 'tokenizer.json').is_file()), None)
    if snapshot is None:
        if args.snapshot:
            raise ValueError('Explicit --snapshot has no local tokenizer')
        return None, {'status': 'unavailable', 'student_revision': model['revision']}
    for name, expected in recorded.get('metadata_sha256', {}).items():
        if file_hash(snapshot / name) != expected:
            raise InvalidExample('tokenizer_snapshot_mismatch')
        inputs[str((snapshot / name).resolve())] = expected
    if not recorded.get('metadata_sha256') and snapshot.name != model['revision']:
        raise InvalidExample('unpinned_tokenizer')
    from train.dataset import load_tokenizer
    tokenizer = load_tokenizer(snapshot)
    from repair.artifacts import tokenizer_identity
    info = dict(path=str(snapshot.resolve()), student_revision=model['revision'], **tokenizer_identity(snapshot, tokenizer))
    for name, value in info['files'].items():
        inputs[str((snapshot / name).resolve())] = value
    return tokenizer, info


def input_records(directories, inputs, *, dry_run, policy, theta0):
    """Read every selected task identity before any rationale call is possible."""
    records, seen = [], set()
    for directory in sorted({Path(p).resolve() for p in directories}):
        plan = read_bound(directory / 'plan.json', inputs)
        if plan['run_sha256'] != digest(plan['identity']):
            raise InvalidExample('run_identity_hash')
        if plan['identity'].get('dry_run') and not dry_run:
            raise InvalidExample('synthetic_run_requires_dry_run')
        provenance = read_bound(directory / 'provenance.json', inputs)
        read_bound(directory / 'run_manifest.json', inputs)
        adapters = provenance.get('config', {}).get('server_provenance', {}).get('config', {}).get('server', {}).get('lora_adapters', [])
        if adapters:
            matches = [a for a in adapters if a.get('name') == THETA0_NAME]
            if len(matches) != 1 or matches[0]['adapter_weights_sha256'] != theta0['adapter_weights_sha256']:
                raise InvalidExample('run_theta0_mismatch')
        paths = sorted((directory / 'records').glob('*.json'))
        if not paths:
            raise InvalidExample('no_episode_records')
        for path in paths:
            row = read_bound(path, inputs)
            if row['run_sha256'] != plan['run_sha256']:
                raise InvalidExample('foreign_episode_record')
            policy.check(row['task_id'], 'joint_local')
            key = (row['run_sha256'], row['run_key'])
            if key in seen:
                raise InvalidExample('duplicate_input_record')
            seen.add(key)
            records.append((str(path), row))
    return records


def build(args):
    reuse_from = getattr(args, 'reuse_labels_from', None)
    if reuse_from is not None:
        if args.arm not in ARMS[:2]:
            raise LabelReuseError('--reuse-labels-from requires --arm JOINT or JOINT_COMPLETE_ONLY')
        if (args.rationale_teacher or args.budget_ledger is not None or
                args.max_teacher_input_tokens is not None or args.max_teacher_output_tokens is not None):
            raise LabelReuseError('--reuse-labels-from is mutually exclusive with --rationale-teacher, '
                                  '--budget-ledger and --max-teacher-* caps')
    if args.label_workers < 1:
        raise ValueError('--label-workers must be >= 1')
    if args.arm in ARMS[:3] and not args.inputs:
        raise ValueError('Decision arms require --inputs run directories')
    if args.rationale_teacher and not args.dry_run and not all(
            v is not None for v in (args.budget_ledger, args.max_teacher_input_tokens, args.max_teacher_output_tokens)):
        raise ValueError('Real L calls require --budget-ledger and both explicit hard token caps')
    output = Path(args.output).resolve()
    if output.exists():
        raise FileExistsError('Dataset directory already exists')
    splits = checked_document(args.splits)
    inputs = {str(Path(args.splits).resolve()): file_hash(args.splits)}
    root = Path(args.verified_root or splits['rebuild_arguments']['verified_root']).resolve()
    collection, replay_collection, selection, receipts = frozen_collection(splits, root, inputs)
    config = load_config(args.config)
    inputs[str(Path(args.config).resolve())] = file_hash(args.config)
    theta0, training_provenance = theta0_identity(root.parents[2], args.theta0_training_dir,
                                                root / 'subsets/k20', inputs, config)
    reuse = None
    if reuse_from is not None:
        reuse = LabelReuseSource(reuse_from, dry_run=args.dry_run, splits_sha256=file_hash(args.splits),
            theta0=theta0, teacher={k: config['models']['teacher'][k]
                                   for k in ('requested_name', 'provider_model_id', 'sampling')})
        inputs[str(reuse.directory / 'manifest.json')] = reuse.manifest_sha256
    task_types = {r['task_id']: r['task_type'] for r in collection['verified']}
    policy = SplitPolicy(splits, selection['task_order'], task_types)
    # Resolve/authorize every input before constructing any teacher transport.
    records = input_records(args.inputs, inputs, dry_run=args.dry_run, policy=policy, theta0=theta0) if args.arm in ARMS[:3] else []
    if args.arm not in ARMS[:3]:
        # OFFLINE/CONTINUE always use frozen receipts. Extra source directories
        # may only name that collection, its K20 view, or theta0's training dir.
        allowed = {root, root / 'subsets/k20', Path(theta0['adapter_path']).parent}
        if any(Path(p).resolve() not in allowed for p in args.inputs):
            raise ValueError('Trajectory --inputs must name the frozen collection, K20 view or theta0 training directory')
    tokenizer, tokenizer_info = local_tokenizer(args, training_provenance, inputs)
    if args.label_workers > 1 and tokenizer is not None:
        tokenizer = LockedTokenizer(tokenizer)
    # A failed build is retained with an explicit report but no sealing manifest.
    # Gather every mandatory trajectory rejection before considering L calls.
    # Admissibility/grounding are decision-only rejection criteria; trajectories
    # retain main-grid rendering and receive informational manifest diagnostics.
    output.mkdir(parents=True, exist_ok=False)
    replay, trajectory_failures = [], []
    def trajectory(episode, ref, source):
        try:
            return build_trajectory(episode, arm=args.arm, source=source, episode_ref=ref, policy=policy)
        except InvalidExample as exc:
            trajectory_failures.append(dict(task_id=episode['task_id'], episode_ref=ref,
                                            source=source, reason=str(exc)))
            return None
    for receipt in replay_collection['verified']:
        episode, ref = load_teacher_episode(root, receipt, inputs)
        example = trajectory(episode, ref, 'k20_replay')
        if example is not None:
            replay.append(example)
    examples, excluded = [], Counter()
    if args.arm == 'CONTINUE':
        examples = list(replay)
    elif args.arm == 'OFFLINE':
        frozen = splits['sets']['V2_pool']['verified_trajectories']
        if len(frozen) != len(policy.pool) or {v['task_id'] for v in frozen} != policy.pool:
            raise InvalidExample('incomplete_offline_receipts')
        for selected in frozen:
            receipt = receipts.get(selected['receipt_sha256'])
            if (receipt is None or receipt['task_id'] != selected['task_id'] or
                    receipt['episode_hashes']['episode.json'] != selected['episode_sha256']):
                raise InvalidExample('offline_receipt_mismatch')
            episode, ref = load_teacher_episode(root, receipt, inputs)
            if episode['episode_id'] != selected['trajectory_id']:
                raise InvalidExample('offline_trajectory_identity')
            example = trajectory(episode, ref, 'offline_teacher')
            if example is not None:
                examples.append(example)
    if trajectory_failures:
        from sed.provenance import utc_now
        write_json(output / 'validation_failures.json', dict(schema='v2-supervision-validation/1',
            arm=args.arm, sealed=False, failures=trajectory_failures, created=utc_now(),
            code_revision=subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=ROOT, check=True,
                                         capture_output=True, text=True).stdout.strip(),
            code_sha256={name: file_hash(ROOT / name) for name in ('repair/supervision.py', 'tools/build_v2_dataset.py')},
            dependency_versions=versions(), theta0=theta0, dry_run=args.dry_run,
            effective_config=dict(student=config['models']['student'], environment=config['environment'],
                                  teacher_alias=config['models']['teacher']['provider_model_id']),
            excluded=[dict(reason=k, count=v) for k, v in sorted(Counter(f['reason'] for f in trajectory_failures).items())],
            inputs=inputs, new_teacher_calls=0, new_environment_steps=0), exclusive=True)
        raise InvalidExample('mandatory_trajectory_validation_failed')
    budget, label_ledger = None, JsonlLedger(output / 'labels.jsonl')
    if reuse is None and args.arm in ARMS[:2] and (args.rationale_teacher or args.dry_run):
        ledger_path = args.budget_ledger or str(output / 'synthetic_budget.jsonl')
        budget = SharedTeacherBudget(ledger_path,
            input_tokens=args.max_teacher_input_tokens if args.max_teacher_input_tokens is not None else 10**9,
            output_tokens=args.max_teacher_output_tokens if args.max_teacher_output_tokens is not None else 10**8)
    def build_record(ref, record):
        labeler = None
        # Lazy factory: void/wrong-mode episodes cannot start a teacher client.
        def label(prompt):
            nonlocal labeler
            if labeler is None:
                directory = output / 'labels' / digest(ref)
                labeler = (ReusedLabeler(reuse, directory) if reuse is not None else
                           RationaleLabeler(config, record['task_id'], directory, budget, dry_run=args.dry_run))
            return labeler(prompt)
        return build_decisions(record, arm=args.arm, episode_ref=ref, policy=policy,
            labeler=label if reuse is not None or budget is not None else None,
            tokenizer=tokenizer, audit=label_ledger.append)
    for built, reasons in label_results(records, build_record, args.label_workers):
        examples.extend(built)
        excluded.update(reasons)
    # Include source/code/content identities, dependencies and all L receipts.
    code_paths = [ROOT / 'repair/supervision.py', ROOT / 'tools/build_v2_dataset.py',
                  ROOT / 'repair/facts.py', ROOT / 'repair/labels.py', ROOT / 'repair/prompts.py',
                  ROOT / 'repair/schemas.py', ROOT / 'src/sed/react.py', ROOT / 'src/sed/visibility.py',
                  ROOT / 'train/dataset.py', ROOT / 'repair/runtime.py', ROOT / 'repair/takeover.py',
                  ROOT / 'teachers/capped.py', ROOT / 'teachers/openai_teacher.py']
    code_hashes = {str(p.relative_to(ROOT)): file_hash(p) for p in code_paths}
    revision = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=ROOT, check=True, capture_output=True, text=True).stdout.strip()
    from repair.takeover import usage_summary
    attempts = [dict(c, usage_ledger=str(p.relative_to(output))) for p in sorted((output / 'labels').glob('*/usage.jsonl'))
                for c in usage_summary(p)['calls']]
    audit_hashes = {str(p.relative_to(output)): file_hash(p) for p in output.rglob('*')
                    if p.is_file() and not p.name.endswith('.lock')}
    metadata = dict(arm=args.arm, inputs=inputs, splits_sha256=file_hash(args.splits), theta0=theta0,
        tokenizer=tokenizer, tokenizer_info=tokenizer_info, code_revision=revision, code_sha256=code_hashes,
        dependency_versions=versions(), dry_run=args.dry_run, training_eligible=not args.dry_run,
        new_environment_steps=0, actual_training_exposures=0, teacher_attempts=attempts,
        new_teacher_calls=0 if args.dry_run else len(attempts),
        synthetic_label_calls=len(attempts) if args.dry_run else 0,
        teacher_alias=config['models']['teacher']['provider_model_id'],
        global_teacher_budget=budget.summary() if budget else None, label_artifact_sha256=audit_hashes,
        effective_config=dict(student=config['models']['student'], environment=config['environment'],
            rationale_teacher=args.rationale_teacher, dry_run=args.dry_run, label_workers=args.label_workers,
            **LABEL_CONFIG,
            budget_ledger=str(budget.path.resolve()) if budget else None,
            label_token_counter='local_tokenizer' if tokenizer else 'conservative_utf8_byte_bound',
            teacher={k: config['models']['teacher'][k] for k in ('requested_name', 'provider_model_id', 'sampling')}))
    if reuse is not None:
        metadata['label_reuse'] = reuse.summary()
        metadata['effective_config']['label_reuse_from'] = str(reuse.directory)
    (output / 'replay').mkdir()
    # Each directory records diagnostics for its own trajectory examples,
    # including zero counts; replay diagnostics never exclude primary decisions.
    replay_manifest = seal_dataset(output / 'replay', replay, policy=policy, excluded=Counter(), **metadata)
    manifest = seal_dataset(output, examples, policy=policy, excluded=excluded,
        replay=dict(path='replay', examples=20, supervised_turns=replay_manifest['supervised_turns'],
                    manifest_sha256=file_hash(output / 'replay/manifest.json'),
                    example_ids_sha256=replay_manifest['example_ids_sha256']), **metadata)
    print(json.dumps(dict(output=str(output), arm=args.arm, examples=len(examples), replay_examples=20,
                          supervised_turns=manifest['supervised_turns'], dry_run=args.dry_run)))
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    p = sub.add_parser('build')
    p.add_argument('--arm', choices=ARMS, required=True)
    p.add_argument('--splits', default='data/manifest/method_v1_splits.json')
    p.add_argument('--inputs', nargs='*', default=[])
    p.add_argument('--output', required=True)
    p.add_argument('--rationale-teacher', action='store_true')
    p.add_argument('--reuse-labels-from', metavar='DIR', help='Reuse completed L labels from a compatible sealed dataset')
    p.add_argument('--label-workers', type=int, default=1, help='Concurrent record labellers (>= 1; default: 1)')
    p.add_argument('--budget-ledger')
    p.add_argument('--max-teacher-input-tokens', type=int)
    p.add_argument('--max-teacher-output-tokens', type=int)
    p.add_argument('--dry-run', action='store_true', help='Synthetic labels through fake transport; no API/model/network')
    p.add_argument('--config', default='configs/base_e4b.yaml')
    p.add_argument('--verified-root', help='Relocated frozen collection; identities must match')
    p.add_argument('--theta0-training-dir', help='Training output containing provenance.json and checkpoint/')
    p.add_argument('--snapshot', help='Local pinned student snapshot, tokenizer only')
    return build(parser.parse_args(argv))


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except Exception as exc:
        # In particular, never render raw SDK errors or an entire process env.
        detail = ': ' + str(exc) if isinstance(exc, LabelReuseError) else ''
        print('E0.4 build failed: ' + type(exc).__name__ + detail + '; no root manifest was sealed.', file=sys.stderr)
        raise SystemExit(1) from None
