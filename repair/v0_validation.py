"""Offline, killable native-planner validation with durable transition receipts.

Timeout means unknown feasibility, never proof of unsolvability. The process
boundary is needed because a Python alarm cannot interrupt native C++ search.
"""
from functools import partial
import json
import multiprocessing
from pathlib import Path

from repair.synth_validation import Execution, offline
from sed.storage import JsonlLedger, file_hash, write_json

VALIDATION_SECONDS = 120


class JournalExecution(Execution):
    def __init__(self, *args, journal, **kwargs):
        super().__init__(*args, **kwargs)
        self.journal = JsonlLedger(journal)

    def reset(self, root, task_id, purpose):
        self.journal.append(dict(event='reset_started', task_id=task_id, purpose=purpose))
        result = super().reset(root, task_id, purpose)
        self.journal.append(dict(event='reset_completed', task_id=task_id, purpose=purpose, entry=self.current['initial']))
        return result

    def step(self, action):
        self.journal.append(dict(event='step_started', action=action, environment_steps=1))
        try:
            result = super().step(action)
        except BaseException as exc:
            self.journal.append(dict(event='step_failed', error_type=type(exc).__name__, outcome='unknown'))
            raise
        self.journal.append(dict(event='step_completed', entry=self.current['steps'][-1]))
        return result

    def plan(self):
        self.journal.append(dict(event='planner_started', never_for_training=True))
        result = super().plan()
        self.journal.append(dict(event='planner_completed', actions=result, never_for_training=True))
        return result


def _child(connection, row, parent, directory, source_root, seed, parent_cache, journal):
    offline()
    try:
        from repair.v0_synth import validate_candidate
        result = validate_candidate(row, parent, Path(directory), source_root, seed, parent_cache,
                                    execution_factory=partial(JournalExecution, journal=journal))
        connection.send((result, parent_cache))
    except BaseException as exc:
        JsonlLedger(journal).append(dict(event='worker_failed', error_type=type(exc).__name__))
    finally:
        connection.close()


def isolated_validation(row, parent, directory, source_root, seed, parent_cache, *,
                        timeout=VALIDATION_SECONDS, worker=_child):
    directory = Path(directory)
    journals = directory / 'validation_attempts' / row['id']
    journals.mkdir(parents=True, exist_ok=True)
    journal = journals / f'{len(list(journals.glob("*.jsonl"))):03d}.jsonl'
    JsonlLedger(journal).append(dict(event='validation_started', timeout_seconds=timeout,
        candidate_id=row['id'], parent_seed_task_id=parent.task_id, teacher_calls=0, student_calls=0))
    context = multiprocessing.get_context('spawn')
    receive, send = context.Pipe(duplex=False)
    process = context.Process(target=worker,
        args=(send, row, parent, str(directory), str(source_root), seed, parent_cache, str(journal)))
    process.start()
    send.close()
    received, reason = None, None
    try:
        if receive.poll(timeout):
            try:
                received = receive.recv()
            except EOFError:
                reason = 'validation_worker_failed_unknown_feasibility'
        else:
            reason = 'validation_timeout_unknown_feasibility'
    finally:
        if received is None and process.is_alive():
            process.terminate()
        process.join(timeout=5)
        if process.is_alive():
            process.kill()
            process.join()
        receive.close()
    if received is not None:
        result, cached = received
        parent_cache.update(cached)
    else:
        events = [json.loads(line) for line in journal.read_text().splitlines()]
        charged = sum(r['event'] == 'step_started' for r in events)
        confirmed = sum(r['event'] == 'step_completed' for r in events)
        path = directory / 'witnesses' / (row['id'] + '.json')
        validation = dict(environment_transitions=charged, environment_transitions_confirmed=confirmed,
            environment_transitions_unknown=charged - confirmed, failure_reason=reason, timeout_seconds=timeout)
        write_json(path, dict(never_for_training=True, episodes=[], validation=validation, rejection=reason,
                             partial_execution_journal=str(journal.relative_to(directory))))
        result = dict(row, status='rejected', rejection=reason, validation=validation,
                      witness_path=str(path.relative_to(directory)), witness_sha256=file_hash(path))
    JsonlLedger(journal).append(dict(event='validation_finished', status=result['status'], reason=result.get('rejection')))
    # Include interrupted attempts from a previous process invocation in cost
    # accounting too. Their exact partial transitions remain append-only evidence.
    all_journals = sorted(journals.glob('*.jsonl'))
    events = [json.loads(line) for p in all_journals for line in p.read_text().splitlines()]
    result['validation']['environment_transitions'] = sum(r['event'] == 'step_started' for r in events)
    result['validation']['environment_transitions_confirmed'] = sum(r['event'] == 'step_completed' for r in events)
    result['validation']['environment_transitions_unknown'] = (
        result['validation']['environment_transitions'] - result['validation']['environment_transitions_confirmed'])
    result['validation_journals'] = {str(p.relative_to(directory)): file_hash(p) for p in all_journals}
    return result
