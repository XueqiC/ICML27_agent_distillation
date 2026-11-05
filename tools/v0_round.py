"""One preregistered K10/K20/K30 seed-0 V0 round. No training or server launch here."""
import argparse
from collections import Counter
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from copy import deepcopy
from functools import partial
import json
import multiprocessing
from pathlib import Path
import sys
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
if __package__ in (None, ''):
    sys.path[:0] = [str(ROOT), str(ROOT / 'src')]

from repair.v0_common import (CAPS, DECODINGS, SyntheticManifest, artifacts, bind, completed,
    check_theta0, finish, phase, read, round_identity, safe_effective, seed_manifest_k, THETA0_CELLS)
from repair.v0_labeling import LabelReuseSource, LabelSetupError, label_tokenizer
from sed.config import load_config
from sed.storage import canonical_bytes, digest, file_hash, locked, write_json


def round_root(args):
    root = Path(args.output).resolve()
    if args.seed != 0:
        raise ValueError('V0 first round is preregistered at seed 0')
    # A fresh round root; later subcommands add their own fresh phase directory.
    seeds = getattr(args, 'seeds', None)
    if seeds:
        k = seed_manifest_k(seeds)
    else:
        k = manifest_for(args, root).k
    identity = round_identity(k)
    if not root.exists() or (root.is_dir() and not any(root.iterdir())):
        root.mkdir(parents=True, exist_ok=True)
        write_json(root / 'round.json', identity, exclusive=True)
    elif not (root / 'round.json').is_file() or read(root / 'round.json') != identity:
        raise ValueError('V0 output must be a fresh directory or this round root')
    return root


def config_for(args):
    config = load_config(args.config)
    if config['environment']['max_episode_steps'] != 40 or config['environment']['seed'] != 0:
        raise ValueError('V0 requires the unchanged H40, environment-seed-0 harness')
    return config


def manifest_for(args, root):
    path = Path(args.synthetic_manifest) if args.synthetic_manifest else root / 'synth/manifest.json'
    manifest = SyntheticManifest(path)
    if (root / 'round.json').exists() and read(root / 'round.json') != round_identity(manifest.k):
        raise ValueError('V0 synthetic manifest K differs from the round')
    return manifest


def serving(args, config):
    if args.model_name != THETA0_CELLS[args.K]:
        raise ValueError(f'V0 K{args.K} requires its configured seed-0 theta0 adapter')
    from repair.adapter_eval import serving_spec
    spec, adapter, inputs = serving_spec(SimpleNamespace(**{**vars(args), 'dry_run': False}), config)
    from students.vllm_student import context_options
    spec.update(context_options(config))  # Server provenance has now been loaded, as in v2_task_eval.
    check_theta0(args.K, adapter['name'], adapter['path'])
    expected_seeds = getattr(args, 'seed_manifest_sha256', None)
    if expected_seeds and config['lora_equivalence']['identity']['verified_manifest_sha256'] != expected_seeds:
        raise ValueError('V0 serving gate used a different verified seed manifest')
    weights = Path(adapter['path']).resolve() / 'adapter_model.safetensors'
    if file_hash(weights) != adapter['adapter_weights_sha256']:
        raise ValueError('V0 served adapter weights changed')
    adapter = dict(adapter, path=str(weights.parent))
    inputs.append(weights)
    return spec, adapter, inputs


def sealed_records(directory, name):
    directory = Path(directory)
    if not completed(directory):
        raise ValueError('V0 dependent phase requires completed sealed inputs')
    return read(directory / name)


def source_inputs(*directories):
    return {str(Path(d).resolve() / name): sha for d in directories
            for name, sha in artifacts(d).items()} | {
        str(Path(d).resolve() / 'completion.json'): file_hash(Path(d) / 'completion.json') for d in directories}


def student_factory(spec, task_id, decoding, seed, log_dir):
    from tools.cg_probe import _student
    return partial(_student, log_dir, spec=spec, decoding=decoding, seed=seed, task_id=task_id)


def worker_count(args, name='workers'):
    value = getattr(args, name, 1)
    if type(value) is not int or value < 1:
        raise ValueError('--' + name.replace('_', '-') + ' must be >= 1')
    return value


def _init_worker(stop):
    global _worker_stop
    _worker_stop = stop


def _process_unit(worker, job):
    # Bound submissions in the parent and stop queued units even when a failure
    # races with the initial fill. Active units settle their durable evidence.
    if _worker_stop.is_set():
        return None
    try:
        return worker(**job)
    except BaseException:
        _worker_stop.set()
        raise


def process_results(jobs, worker, workers):
    """Spawn isolated environments/clients; merge in input order, fail fast."""
    if workers == 1:
        return [worker(**job) for job in jobs]
    # Same isolation as eval.run_suite (used by v2_task_eval/repair_coverage).
    context = multiprocessing.get_context('spawn')
    stop = context.Event()
    results, pending = [None] * len(jobs), {}
    remaining = iter(enumerate(jobs))
    exhausted = False
    with ProcessPoolExecutor(max_workers=workers, mp_context=context,
                             initializer=_init_worker, initargs=(stop,)) as pool:
        try:
            while pending or not exhausted:
                while not exhausted and not stop.is_set() and len(pending) < workers:
                    item = next(remaining, None)
                    if item is None:
                        exhausted = True
                    else:
                        index, job = item
                        pending[pool.submit(_process_unit, worker, job)] = index
                if pending:
                    done, _ = wait(pending, return_when=FIRST_COMPLETED)
                    for future in done:
                        results[pending.pop(future)] = future.result()
                elif stop.is_set():
                    raise ValueError('V0 worker stopped; evidence retained')
        finally:
            stop.set()
            for future in pending:
                future.cancel()
    return results


def _attempt_unit(*, manifest, spec, task_id, decoding, seed, directory, probes):
    from repair.v0_execution import attempt
    try:
        row = attempt(manifest, task_id, decoding=decoding, sampling_seed=seed,
            env_factory=partial(manifest.environment, task_id, probes=probes),
            student_factory=student_factory(spec, task_id, decoding, seed, directory / 'student'),
            directory=directory, probes=probes)
        print(json.dumps(dict(stage='attempt', id=row['attempt_id'], success=row['success'])), flush=True)
        if row['execution']['termination_reason'] == 'ENV_ERROR':
            raise ValueError('V0 student/environment error retained; collection halted')
        return row
    except Exception as exc:
        # Never serialize an SDK exception/body (it may contain credentials).
        write_json(directory / 'worker_error.json', dict(error_type=type(exc).__name__, task_id=task_id))
        raise


def run_attempts(args, root, manifest, *, directory=None, probes=False):
    from repair.v0_execution import attempt_id, probe_report
    workers = worker_count(args)
    directory = Path(directory or root / ('probe_attempt' if probes else 'attempt'))
    ids = args.task_ids if args.task_ids is not None else [
        t['task_id'] for t in manifest.tasks.values() if t['heldout_probe'] == probes]
    if len(ids) != len(set(ids)):
        raise ValueError('V0 duplicate requested IDs')
    tasks = [manifest.authorize(tid, probes=probes) for tid in ids]
    expected_ids = {t['task_id'] for t in manifest.tasks.values() if t['heldout_probe'] == probes and
                    (probes or not manifest.value.get('expansion_variant') or t.get('expansion_unit') is not None)}
    config = config_for(args)
    # Serving validation may create its student log directory. Keep it inside
    # this phase too, including when the round itself is a read-only input.
    spec, adapter, inputs = serving(SimpleNamespace(**{**vars(args), 'output': str(directory),
        'K': manifest.k, 'seed_manifest_sha256': manifest.value.get('source_seed_manifest_sha256')}), config)
    inputs += [args.config, manifest.path, config['paths']['manifest'], config['paths']['split_freeze']]
    phase(directory, dict(command='probe' if probes else 'attempt', inputs=bind(inputs),
        config=safe_effective(config), adapter=adapter, serving=spec, task_ids=ids,
        synthetic_manifest_sha256=manifest.sha256, decodings=DECODINGS, seed=0,
        complete_task_coverage=set(ids) == expected_ids, H=40, temperature=.7, top_p=.95), resume=args.resume)
    if completed(directory):
        return read(directory / 'records.json')
    # Scheduling knobs do not enter the scientific identity/seal. A resume may
    # change concurrency; attempt() reuses each deterministic record verbatim.
    jobs = [dict(manifest=manifest, spec=spec, task_id=task['task_id'], decoding=decoding,
        seed=seed, directory=directory / 'episodes' / attempt_id(manifest, task['task_id'], decoding, seed),
        probes=probes) for task in tasks for decoding, seed in DECODINGS]
    records = process_results(jobs, _attempt_unit, workers)
    write_json(directory / 'records.json', records)
    report = probe_report(tasks, records) if probes else dict(attempts=len(records),
        successes=sum(r['success'] is True for r in records), failures=sum(r['success'] is False for r in records),
        unknown=sum(type(r['success']) is not bool for r in records), teacher_calls=0, training_exposure=0)
    if probes:
        report['adapter'] = adapter
    write_json(directory / 'summary.json', report)
    finish(directory, **report)
    return records


def budget_for(root):
    from repair.takeover import SharedTeacherBudget
    budget = SharedTeacherBudget(root / 'teacher_budget.jsonl', **CAPS)
    budget.check()
    if budget.summary()['unsettled_attempts']:
        budget.halt('v0_unsettled_teacher_attempt')
        budget.check()
    return budget


def review_dry_run(args, root, manifest, attempts_dir):
    from repair.records import TrajectoryRecord
    from repair.v0_execution import selected_failures
    from repair.v0_review import (OUTPUT_CAP, RETRY_OUTPUT_CAP, RATIONALE_OUTPUT_CAP,
                                 RATIONALE_RETRY_OUTPUT_CAP, review_prompt)
    attempts = sealed_records(attempts_dir, 'records.json')
    if any(r['heldout_probe'] for r in attempts):
        raise ValueError('V0 probes cannot be sent to trajectory review')
    failures = selected_failures(attempts)
    directory = root / 'review_dry_run'
    phase(directory, dict(command='review --dry-run', inputs={**source_inputs(attempts_dir), **bind([manifest.path])}),
          resume=args.resume)
    if completed(directory):
        return read(directory / 'estimate.json')
    requests = []
    for row in failures:
        manifest.authorize(row['task_id'])
        prompt = review_prompt(TrajectoryRecord.from_json(row['execution']), row['prefix_length'])
        raw_bytes = len(canonical_bytes(prompt.messages()))
        requests.append(dict(attempt_id=row['attempt_id'], request_bytes=raw_bytes,
            input_reservation_bound=raw_bytes + 16 * (len(prompt.messages()) + 1), request=prompt.to_json()))
    estimate = dict(failures=len(failures), teacher_calls=0, student_calls=0,
        first_round_input_byte_bound=sum(r['input_reservation_bound'] for r in requests),
        projected_review_input_at_current_request_sizes=12 * sum(r['input_reservation_bound'] for r in requests),
        review_output_cap=OUTPUT_CAP, review_retry_output_cap=RETRY_OUTPUT_CAP,
        rationale_output_cap=RATIONALE_OUTPUT_CAP, rationale_retry_output_cap=RATIONALE_RETRY_OUTPUT_CAP,
        maximum_review_output_tokens=len(failures) * 3 * 2 * (OUTPUT_CAP + RETRY_OUTPUT_CAP),
        maximum_label_output_tokens=len(failures) * 40 * (RATIONALE_OUTPUT_CAP + RATIONALE_RETRY_OUTPUT_CAP), caps=CAPS,
        note='Input projection is an estimate: later public execution feedback changes request sizes. '
             'Output projections include one truncation retry per call, excluding HTTP retries. '
             'The live shared ledger reserves each actual request before transport.')
    write_json(directory / 'requests.json', requests)
    write_json(directory / 'estimate.json', estimate)
    finish(directory, **estimate)
    for row in requests:
        print(json.dumps({k: v for k, v in row.items() if k != 'request'}))
    return estimate


def _review_unit(*, manifest, spec, config, attempt, directory, budget):
    from repair.joint import usage_summary
    from repair.ledger import BudgetLedger
    from repair.v0_review import WholeReviewer, WholeTransport, repair_failure
    path = directory / 'record.json'
    try:
        budget.check()
        if path.exists():
            record = read(path)
        else:
            transport = WholeTransport(config, attempt['task_id'], directory / 'teacher', budget)
            reviewer = WholeReviewer(transport, directory / 'requests')
            ledger = BudgetLedger(directory / 'environment.jsonl')
            record = repair_failure(attempt, reviewer=reviewer,
                env_factory=partial(manifest.environment, attempt['task_id']),
                student_factory=student_factory(spec, attempt['task_id'], attempt['decoding'],
                                                attempt['sampling_seed'], directory / 'student'),
                ledger=ledger, directory=directory)
            record.update(teacher_usage=usage_summary(directory / 'teacher/usage.jsonl'),
                          synthetic_manifest_sha256=manifest.sha256)
            record['teacher_tokens'] = record['teacher_usage']['totals']
            write_json(path, record, exclusive=True)
        print(json.dumps(dict(stage='review', attempt=attempt['attempt_id'], success=record['success'],
                              rounds=len(record['rounds']))), flush=True)
        if record['execution']['termination_reason'] == 'ENV_ERROR':
            raise ValueError('V0 repair execution failed; evidence retained')
        budget.check()
        return record
    except Exception as exc:
        write_json(directory / 'worker_error.json', dict(error_type=type(exc).__name__,
                                                        attempt_id=attempt['attempt_id']))
        raise


def run_reviews(args, root, manifest, attempts_dir, *, directory=None):
    from repair.v0_execution import selected_failures
    workers = worker_count(args)
    attempts = sealed_records(attempts_dir, 'records.json')
    if any(r['heldout_probe'] for r in attempts):
        raise ValueError('V0 probes cannot be sent to trajectory review')
    directory = Path(directory or root / 'review')
    config = config_for(args)
    spec, adapter, inputs = serving(SimpleNamespace(**{**vars(args), 'K': manifest.k,
        'seed_manifest_sha256': manifest.value.get('source_seed_manifest_sha256')}), config)
    old_adapter = read(Path(attempts_dir) / 'plan.json')['identity']['adapter']
    if any(adapter[k] != old_adapter[k] for k in ('adapter_weights_sha256', 'adapter_config_sha256', 'revision')):
        raise ValueError('V0 handback adapter differs from the failed student')
    inputs += [args.config, manifest.path]
    phase(directory, dict(command='review', inputs={**source_inputs(attempts_dir), **bind(inputs)},
        adapter=adapter, serving=spec, config=safe_effective(config), caps=CAPS, max_rounds=3,
        max_failures_per_item=2, max_patch_blocks=2), resume=args.resume)
    if completed(directory):
        return read(directory / 'records.json')
    budget = budget_for(root)
    failures = selected_failures(attempts)
    for attempt in failures:
        manifest.authorize(attempt['task_id'])
        if attempt['synthetic_manifest_sha256'] != manifest.sha256:
            raise ValueError('V0 review manifest differs from collection')
    # SharedTeacherBudget is a path/caps handle: its accounting and the common
    # WholeTransport rate limiter use the same file locks in every process.
    records = process_results([dict(manifest=manifest, spec=spec, config=config, attempt=attempt,
        directory=directory / 'failures' / attempt['attempt_id'], budget=budget)
        for attempt in failures], _review_unit, workers)
    budget.check()
    write_json(directory / 'records.json', records)
    write_json(directory / 'budget_snapshot.json', budget.summary())
    finish(directory, failures=len(records), repaired=sum(r['verified'] for r in records), budget=budget.summary())
    return records


class DurableLabels:
    """A settled label request can be reused across a process interruption."""
    def __init__(self, make_labeler, directory, reuse=None):
        self.make_labeler, self.directory, self.labeler = make_labeler, Path(directory), None
        self.reuse = reuse

    def __call__(self, prompt):
        path = self.directory / (digest(prompt.to_json()) + '.json')
        if path.exists():
            row = read(path)
            if row['request'] != prompt.to_json():
                raise ValueError('V0 resumed label request changed')
            return row['raw_output']
        transport_log = self.directory.parent / 'teacher/requests.jsonl'
        old = [json.loads(line) for line in transport_log.read_text().splitlines()] if transport_log.exists() else []
        settled = [r for r in old if r.get('request') == prompt.to_json() and r['status'] == 'completed']
        if settled:
            raw = settled[-1]['raw_output']
            write_json(path, dict(request=prompt.to_json(), raw_output=raw, recovered_settled_transport=True), exclusive=True)
            return raw
        reused = self.reuse.find(prompt.to_json()) if self.reuse is not None else None
        if reused is not None:
            write_json(path, dict(request=prompt.to_json(), **reused), exclusive=True)
            return reused['raw_output']
        if self.labeler is None:
            self.labeler = self.make_labeler()
        raw = self.labeler(prompt)
        write_json(path, dict(request=prompt.to_json(), raw_output=raw), exclusive=True)
        return raw


def run_labels(args, root, manifest, attempts_dir, reviews_dir, *, directory=None, expanded=False):
    from repair.v0_data import choose_episodes, decision_rows
    from sed.storage import JsonlLedger
    from tools.build_v2_dataset import LockedTokenizer, label_results
    workers = worker_count(args, 'label_workers')
    attempts, repairs = sealed_records(attempts_dir, 'records.json'), sealed_records(reviews_dir, 'records.json')
    directory = Path(directory or root / 'label')
    config = config_for(args)
    label_inputs = {**source_inputs(attempts_dir, reviews_dir), **bind([manifest.path])}
    config_sha256 = file_hash(args.config)
    inputs = {**label_inputs, **bind([args.config])}
    tokenizer, tokenizer_info = label_tokenizer(args, config, inputs)
    reuse_path = getattr(args, 'reuse_labels_from', None)
    if reuse_path and args.action_only:
        raise LabelSetupError('--reuse-labels-from cannot be combined with --action-only')
    reuse = LabelReuseSource(reuse_path, label_inputs, config_sha256) if reuse_path else None
    if reuse is not None:
        if directory.resolve() == reuse.directory:
            raise LabelSetupError('V0 label reuse requires a new output phase')
        inputs.update({str(reuse.directory / name): sha for name, sha in artifacts(reuse.directory).items()})
        if (reuse.directory / 'completion.json').is_file():
            inputs.update(bind([reuse.directory / 'completion.json']))
    phase(directory, dict(command='label', inputs=inputs, label_source_inputs=label_inputs,
        label_config_sha256=config_sha256,
        tokenizer=tokenizer_info, reuse_labels_from=str(reuse.directory) if reuse else None,
        action_only=args.action_only, expanded=expanded, config=safe_effective(config), caps=CAPS,
        complete_task_coverage=read(Path(attempts_dir) / 'plan.json')['identity']['complete_task_coverage'],
        theta0=read(Path(attempts_dir) / 'plan.json')['identity']['adapter']), resume=args.resume)
    if completed(directory):
        return read(directory / 'rows.json')
    budget = None if args.action_only else budget_for(root)
    if workers > 1:
        tokenizer = LockedTokenizer(tokenizer)
    rows, counts = [], Counter()
    def build_episode(bucket, episode):
        dest = directory / 'episodes' / episode['episode_id']
        if (dest / 'rows.json').exists():
            built, exclusions = read(dest / 'rows.json'), read(dest / 'excluded.json')
        else:
            dest.mkdir(parents=True, exist_ok=True)
            labeler = None
            if budget is not None:
                from repair.v0_review import make_rationale_labeler
                labeler = DurableLabels(partial(make_rationale_labeler, config, episode['task_id'], dest / 'teacher', budget),
                                        dest / 'requests', reuse=reuse)
            built, exclusions = decision_rows(episode, manifest, bucket=bucket,
                episode_ref=str((dest / 'execution.json').resolve()), labeler=labeler,
                audit=JsonlLedger(dest / 'labels.jsonl').append, tokenizer=tokenizer)
            write_json(dest / 'execution.json', episode)
            write_json(dest / 'excluded.json', exclusions)
            write_json(dest / 'rows.json', built)
        if budget:
            budget.check()
        return built, exclusions
    def build_record(bucket, episode):
        try:
            return build_episode(bucket, episode)
        except Exception as exc:
            write_json(directory / 'episodes' / episode['episode_id'] / 'worker_error.json',
                       dict(error_type=type(exc).__name__, episode_id=episode['episode_id']))
            raise
    for built, exclusions in label_results(choose_episodes(attempts, repairs, expanded=expanded),
                                           build_record, workers):
        rows.extend(built)
        counts.update(exclusions)
    write_json(directory / 'rows.json', rows)
    write_json(directory / 'excluded.json', dict(counts))
    write_json(directory / 'budget_snapshot.json', budget.summary() if budget else dict(teacher_calls=0, action_only=True))
    finish(directory, rows=len(rows), excluded=dict(counts))
    return rows


def build_dataset(args, root, manifest, labels_dir, *, directory=None, inherited=None, extra_inputs=None):
    from repair.v0_data import publish_dataset
    rows = sealed_records(labels_dir, 'rows.json')
    if read(Path(labels_dir) / 'plan.json')['identity']['complete_task_coverage'] is not True:
        raise ValueError('V0 dataset cannot seal an incomplete task-coverage smoke run')
    directory = Path(directory or root / 'dataset')
    build_dir = directory.parent / (directory.name + '_build')
    inputs = {**source_inputs(labels_dir), **bind([manifest.path, args.config])}
    # Include private generation/validation, rejected candidates and failed attempts
    # in seals, never in messages. Immutable completed phases only.
    inputs.update(extra_inputs or {})
    config = config_for(args)
    inputs.update(bind([config['paths']['manifest'], config['paths']['split_freeze'], args.splits]))
    if args.replay:
        inputs.update(bind([Path(args.replay) / name for name in ('manifest.json', 'examples.jsonl', 'examples.sha256')]))
    plan = phase(build_dir, dict(command='build', inputs=inputs, replay=args.replay,
        verified_root=args.verified_root, theta0_training_dir=args.theta0_training_dir), resume=args.resume)
    if completed(build_dir):
        from repair.v0_data import load_dataset
        return load_dataset(directory)[0]
    if args.replay:
        replay = Path(args.replay)
    else:
        replay = build_dir / 'v2_continue'
        if not (replay / 'manifest.json').exists():
            if manifest.k == 20:
                from tools.build_v2_dataset import build
                build(SimpleNamespace(arm='CONTINUE', splits=args.splits, inputs=[], output=str(replay),
                    config=args.config, verified_root=args.verified_root, theta0_training_dir=args.theta0_training_dir,
                    snapshot=args.snapshot, label_workers=worker_count(args, 'label_workers'), rationale_teacher=False, dry_run=False,
                    budget_ledger=None, max_teacher_input_tokens=None, max_teacher_output_tokens=None))
            else:
                from repair.v0_data import build_seed_replay
                build_seed_replay(replay, manifest, args, config,
                    read(Path(labels_dir) / 'plan.json')['identity']['theta0'])
    if (directory / 'manifest.json').exists():
        from repair.v0_data import load_dataset
        result, _ = load_dataset(directory)
    else:
        result = publish_dataset(directory, rows, manifest=manifest, replay=replay, inputs=inputs,
            metadata=dict(code_revision=plan['code_revision'], code_sha256=plan['identity']['code_sha256'],
                dependency_versions=plan['identity']['dependencies'], effective_config=safe_effective(config),
                theta0=read(Path(labels_dir) / 'plan.json')['identity']['theta0'],
                round_teacher_budget=budget_for(root).summary() if (root / 'teacher_budget.jsonl').exists() else None),
            inherited=inherited)
    finish(build_dir, dataset_manifest_sha256=file_hash(directory / 'manifest.json'), counts=result['counts'])
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    for command in ('synth', 'attempt', 'probe', 'review', 'label', 'build', 'expand'):
        p = sub.add_parser(command)
        p.add_argument('--output', required=True, help='Round root; created fresh by the first subcommand')
        p.add_argument('--seed', type=int, default=0)
        p.add_argument('--resume', action='store_true')
        p.add_argument('--config', default='configs/base_e4b.yaml')
        p.add_argument('--synthetic-manifest')
        if command in ('synth', 'expand'):
            p.add_argument('--seeds', required=True)
            p.add_argument('--data-root', required=True)
            p.add_argument('--main-checkout', required=True)
        if command == 'synth':
            p.add_argument('--workers', type=int, default=4)
            p.add_argument('--split-freeze', help='Reuse original freeze paths after checking identical exclusions and source hashes')
        if command in ('attempt', 'probe', 'review', 'expand'):
            p.add_argument('--workers', type=int, default=1, help='Concurrent isolated episodes/failures (default: 1)')
            for field in ('endpoint', 'model-name', 'server-provenance', 'lora-check'):
                p.add_argument('--' + field)
            p.add_argument('--task-ids', nargs='+')
        if command == 'attempt':
            p.add_argument('--probes', action='store_true')
        if command == 'probe':
            p.add_argument('--report-only', action='store_true')
            p.add_argument('--attempt-dir', type=Path,
                           help='Fresh exclusive probe phase directory; round is read-only')
        if command in ('review', 'label', 'expand'):
            p.add_argument('--attempts')
        if command == 'review':
            p.add_argument('--dry-run', action='store_true', help='Only public request sizes and budget estimate; no clients')
        if command in ('label', 'expand'):
            p.add_argument('--reviews')
            p.add_argument('--action-only', action='store_true', help='Explicit V2 action-only fallback, no rationale teacher')
            p.add_argument('--byte-token-cap', action='store_true', help='TEST ONLY: count UTF-8 bytes instead of student tokens')
        if command == 'label':
            p.add_argument('--reuse-labels-from', help='Reuse completed raw outputs for identical requests in a previous label phase')
        if command in ('label', 'build', 'expand'):
            p.add_argument('--label-workers', type=int, default=1, help='Concurrent record labellers (default: 1)')
            p.add_argument('--snapshot', help='Local pinned student snapshot containing tokenizer.json; otherwise discover from config')
        if command == 'build':
            p.add_argument('--labels')
        if command in ('build', 'expand'):
            p.add_argument('--replay', help='Sealed V2 CONTINUE bundle; otherwise invoke the unchanged V2 builder')
            p.add_argument('--splits', default='data/manifest/method_v1_splits.json')
            p.add_argument('--verified-root')
            p.add_argument('--theta0-training-dir')
        if command == 'expand':
            p.add_argument('--dataset', required=True)
            p.add_argument('--variant', choices=('D-data', 'D-param'), required=True)
            p.add_argument('--groups')
            p.add_argument('--weights')
            p.add_argument('--strict-fill', action='store_true',
                           help='Fail on unfilled units after same-group fallback (default: continue)')
            p.add_argument('--dry-run', action='store_true',
                           help='Generate and validate seed edits on CPU; no student attempts, teacher calls or dataset build')
            p.add_argument('--synth-only', action='store_true',
                           help='Complete CPU expansion synthesis; leave expand open for a normal --resume')
    args = parser.parse_args(argv)
    if args.command == 'expand':
        from repair.v0_expand import expansion_synth_timeout
        try:
            expansion_synth_timeout()
        except ValueError as exc:
            parser.error(str(exc))
    worker_count(args)
    worker_count(args, 'label_workers')
    if args.command == 'probe' and args.attempt_dir is not None:
        if args.resume or args.report_only:
            parser.error('--attempt-dir requires a fresh run; cannot use --resume or --report-only')
        directory = args.attempt_dir.resolve()
        if directory.exists() or args.attempt_dir.is_symlink():
            raise FileExistsError('V0 attempt directory must not exist')
        root = Path(args.output).resolve()
        manifest = manifest_for(args, root)
        if args.seed != 0 or read(root / 'round.json') != round_identity(manifest.k):
            raise ValueError('V0 probe requires an existing seed-0 round')
        # No round_root() or round.lock: isolated probes only read round inputs.
        run_attempts(args, root, manifest, directory=directory, probes=True)
        if not completed(directory):
            raise ValueError('V0 probe report requires completed attempts')
        print(json.dumps(read(directory / 'summary.json'), sort_keys=True))
        return 0
    root = round_root(args)
    # One invocation mutates a round at a time, including the global teacher ledger.
    with locked(root / 'round.lock'):
        if args.command == 'synth':
            from repair.v0_synth import synthesize
            result = synthesize(seeds_path=args.seeds, data_root=args.data_root, output=root,
                main_checkout=args.main_checkout, seed=args.seed, workers=args.workers, resume=args.resume,
                split_freeze=args.split_freeze)
        else:
            manifest = manifest_for(args, root)
            if args.command in ('attempt', 'probe'):
                probes = args.command == 'probe' or args.probes
                if args.command != 'probe' or not args.report_only:
                    run_attempts(args, root, manifest, probes=probes)
                directory = root / ('probe_attempt' if probes else 'attempt')
                if not completed(directory):
                    raise ValueError('V0 probe report requires completed attempts')
                if read(directory / 'plan.json')['identity']['synthetic_manifest_sha256'] != manifest.sha256:
                    raise ValueError('V0 probe report manifest differs from the requested manifest')
                result = read(directory / 'summary.json')
            elif args.command == 'review':
                attempts = Path(args.attempts or root / 'attempt')
                result = review_dry_run(args, root, manifest, attempts) if args.dry_run else dict(
                    reviewed=len(run_reviews(args, root, manifest, attempts)))
            elif args.command == 'label':
                result = dict(rows=len(run_labels(args, root, manifest,
                    Path(args.attempts or root / 'attempt'), Path(args.reviews or root / 'review'))))
            elif args.command == 'build':
                result = build_dataset(args, root, manifest, Path(args.labels or root / 'label'),
                    extra_inputs=source_inputs(root / 'synth', root / 'attempt', root / 'review'))
                result = dict(counts=result['counts'], dataset=str(root / 'dataset'))
            else:
                from repair.v0_expand import expand
                result = expand(args, root, manifest)
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except Exception as exc:
        # SDK exceptions can contain credentials and private provider bodies.
        print(json.dumps(dict(status='error', error_type=type(exc).__name__,
                              detail=str(exc) if isinstance(exc, LabelSetupError) else
                              'V0 halted; completed evidence and token reservations are retained.')), file=sys.stderr)
        raise SystemExit(1) from None
