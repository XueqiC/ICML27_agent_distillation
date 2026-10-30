"""Evaluation-only collection-teacher continuation and durable shared accounting."""
from dataclasses import replace
import json
from pathlib import Path
import re
from types import SimpleNamespace
import uuid

from repair.artifacts import read_json
from repair.budgets import GlobalTeacherBudget
from repair.execute import execute_actions, environment_version, task_fingerprint
from repair.facts import FactTable
from repair.guidance import derive_metrics, DryEnvironment
from repair.ledger import BudgetLedger
from repair.runtime import CaseStopped
from sed.provenance import utc_now
from sed.react import build_messages, PROMPT_TEMPLATE, PROMPT_VERSION
from sed.storage import canonical_bytes, digest, JsonlLedger, locked, write_json
from teachers.openai_teacher import (Teacher, TeacherCallError, TeacherQuotaError,
                                    has_known_usage, rate_limit_options, _is_rate_limit_429)
from teachers.stage import require_completed


class SharedTeacherBudget:
    """Process-safe adapter around GlobalTeacherBudget's accounting semantics.

    Each atomic ledger event is also a full state checkpoint. Reservations are
    request-id bound; replaying this ledger never refunds an unknown attempt.
    The lock covers accounting only, never transport or rate-limit sleeps.
    """
    def __init__(self, path, *, input_tokens, output_tokens):
        self.path = Path(path)
        self.caps = {'input_tokens': input_tokens, 'output_tokens': output_tokens}
        if any(type(v) is not int or v < 0 for v in self.caps.values()):
            raise ValueError('Token caps must be nonnegative integers')
        with locked(str(self.path) + '.accounting.lock'):
            budget, _ = self._load()
            if not self.path.exists():
                budget.ledger.append({'event': 'initialized', **budget.summary()})

    def _load(self):
        rows = []
        if self.path.exists():
            content = self.path.read_bytes()
            if not content.endswith(b'\n'):
                raise ValueError('Damaged global budget ledger')
            rows = [json.loads(line) for line in content.splitlines()]
        if not rows:
            return GlobalTeacherBudget(self.path, **self.caps), rows
        state = rows[-1]
        if state['caps'] != self.caps:
            raise ValueError('Global caps changed on resume')
        # Restore the existing accounting implementation, without its deliberate
        # fresh-run-only constructor. Every restore is under the process lock.
        budget = object.__new__(GlobalTeacherBudget)
        budget.ledger, budget.caps = JsonlLedger(self.path), dict(self.caps)
        budget.used = dict(state['tokens_or_unsettled_bounds'])
        budget.reported_usage = dict(state['reported_usage'])
        budget.unknown_quantities = dict(state['unknown_quantities'])
        budget.calls, budget.stop_reason = state['reserved_attempts'], state['stop_reason']
        budget.context = {}
        return budget, rows

    def reserve(self, request_id, input_bound, output_cap, **context):
        with locked(str(self.path) + '.accounting.lock'):
            budget, rows = self._load()
            if any(r.get('request_id') == request_id for r in rows):
                raise ValueError('Duplicate transport reservation')
            budget.context = dict(context, request_id=request_id, timestamp=utc_now())
            budget.reserve(input_bound, output_cap)

    def settle(self, request_id, usage):
        if not has_known_usage(usage):
            return  # Unknown quantities retain the entire conservative bound.
        with locked(str(self.path) + '.accounting.lock'):
            budget, rows = self._load()
            reservation = next((r for r in rows if r['event'] == 'reservation' and r.get('request_id') == request_id), None)
            if reservation is None:
                raise ValueError('Settlement without reservation')
            if any(r['event'] == 'settlement' and r.get('request_id') == request_id for r in rows):
                return  # Crash recovery can safely replay durable usage receipts.
            budget.context = {'request_id': request_id, 'timestamp': utc_now()}
            try:
                budget.settle(reservation['amounts']['input_tokens'], reservation['amounts']['output_tokens'], usage)
            except CaseStopped:
                # The upstream helper sets this reason after its settlement row.
                budget.ledger.append({'event': 'stopped', **budget.summary()})
                raise

    def halt(self, reason='teacher_halted'):
        with locked(str(self.path) + '.accounting.lock'):
            budget, _ = self._load()
            # Fatal transport failures take precedence over a concurrent cap stop.
            budget.stop_reason = reason
            budget.ledger.append({'event': 'halted', 'timestamp': utc_now(), **budget.summary()})

    def summary(self):
        with locked(str(self.path) + '.accounting.lock'):
            budget, rows = self._load()
            reservations = {r['request_id'] for r in rows if r['event'] == 'reservation'}
            settled = {r['request_id'] for r in rows if r['event'] == 'settlement'}
            return {**budget.summary(), 'unsettled_attempts': len(reservations - settled)}

    def check(self):
        reason = self.summary()['stop_reason']
        if reason:
            raise CaseStopped(reason)


class _Reservations:
    def __init__(self, budget, output_cap):
        self.budget, self.output_cap, self.input_bound = budget, output_cap, None

    def reserve(self, *, request_id, logical_id, task_id, usage_ledger, input_tokens):
        if self.input_bound is None:
            raise ValueError('Public prompt must be bounded before transport')
        self.budget.reserve(request_id, self.input_bound, self.output_cap,
            logical_id=logical_id, task_id=task_id, usage_ledger=str(usage_ledger))


class _UsageLedger:
    def __init__(self, ledger, budget):
        self.ledger, self.path, self.budget = ledger, ledger.path, budget
        self.retry = None

    def append(self, row, **kwargs):
        if self.retry is not None:
            row = self.retry.annotate(row)
        self.ledger.append(row, **kwargs)
        if row['status'] != 'started' and row['status'] != 'cache_hit':
            self.budget.settle(row['call_id'], row.get('usage'))


_TRANSIENT_HTTP = {408, 409, 429}
_CONNECTION_ERRORS = {
    'APIConnectionError', 'APITimeoutError', 'ConnectionError', 'TimeoutError',
    'ConnectionResetError', 'ConnectionAbortedError', 'ConnectionRefusedError',
    'BrokenPipeError', 'TimeoutException', 'ReadTimeout', 'ConnectTimeout',
    'WriteTimeout', 'PoolTimeout', 'NetworkError', 'ConnectError', 'ReadError',
    'WriteError', 'CloseError', 'ProxyError', 'RemoteProtocolError',
}
_RETRY_DELAYS = (5, 10, 20, 40, 80, 160)


def _teacher_failure(exc=None, row=None):
    """Classify sanitized receipts; inspect quota bodies only in memory."""
    row = row or {}
    if isinstance(exc, TeacherQuotaError) or row.get('error_type') == 'TeacherQuotaError':
        return 'QUOTA'
    status = getattr(exc, 'status_code', None)
    if status == 429:
        return 'TRANSIENT' if _is_rate_limit_429(exc) else 'QUOTA'
    # Completed/invalid responses must not inherit an earlier transport failure.
    if row.get('status') not in ('failed', 'incomplete'):
        row = {}
    status = status if status is not None else row.get('http_status')
    if status is not None:
        return 'TRANSIENT' if (status in _TRANSIENT_HTTP
                               or isinstance(status, int) and 500 <= status < 600) else 'OTHER'
    if (isinstance(exc, (ConnectionError, TimeoutError))
            or type(exc).__name__ in _CONNECTION_ERRORS
            or row.get('error_type') in _CONNECTION_ERRORS
            or type(exc).__name__ == 'TeacherRateLimitError'):
        return 'TRANSIENT'
    return 'OTHER'


class _TeacherRetry:
    """Opt-in repair retries, separate from the unchanged transport attempts."""
    def __init__(self, teacher, budget):
        self.teacher, self.budget = teacher, budget
        self.logical_id = str(uuid.uuid4())
        self.last = None

    def begin(self, index):
        self.index, self.request_id, self.last = index, str(uuid.uuid4()), None
        self.teacher.ledger.retry = self

    def failure(self, exc):
        kind = _teacher_failure(exc, self.last)
        retry = (kind == 'TRANSIENT' and self.index < len(_RETRY_DELAYS)
                 and not self.budget.summary()['stop_reason'])
        wait = 0
        if retry:
            if self.last and self.last.get('repair_will_retry'):
                wait = self.last['repair_retry_wait_seconds']
            else:
                base = _RETRY_DELAYS[self.index]
                # Positive jitter bounded to 10%: all six waits total <=346.5s.
                wait = base + self.teacher.jitter(0, base * .1)
        return dict(failure_class=kind, will_retry=retry, retry_wait_seconds=wait)

    def annotate(self, row):
        row = dict(row, repair_request_id=self.request_id,
                   repair_logical_request_id=self.logical_id,
                   repair_attempt_index=self.index, failure_class=None)
        if row['status'] in ('failed', 'incomplete'):
            self.last = row
            info = self.failure(None)
            transport_retry = bool(row.get('will_retry'))
            row.update(failure_class=info['failure_class'],
                       transport_will_retry=transport_retry,
                       repair_will_retry=info['will_retry'] and not transport_retry,
                       repair_retry_wait_seconds=info['retry_wait_seconds'] if not transport_retry else 0,
                       # Resume reconciliation must recognize a scheduled repair
                       # retry; retain the original transport decision separately.
                       will_retry=transport_retry or info['will_retry'])
        elif row['status'] == 'cache_hit' or row.get('truncated'):
            row['failure_class'] = 'OTHER'
        if row['status'] != 'state_link':
            self.last = row
        return row

    def record_failure(self, exc, info):
        # Policy-level rejections (e.g. empty completed output) need a receipt
        # too. This bookkeeping row carries no second usage/settlement.
        self.teacher.ledger.ledger.append(dict(
            run_id=self.teacher.run_id, timestamp=utc_now(), status='repair_outcome',
            call_id=(self.last or {}).get('call_id', self.request_id), usage=None,
            repair_request_id=self.request_id, repair_logical_request_id=self.logical_id,
            repair_attempt_index=self.index, error_type=type(exc).__name__, **info))

    def wait(self, seconds):
        # Short sleeps also observe another worker's stop during long backoff.
        self.budget.check()
        while seconds > 0:
            interval = min(seconds, 1.0)
            self.teacher.sleep(interval)
            seconds -= interval
            self.budget.check()


class _QuotaGuard:
    """Keep collection transport/backoff, but fail quota 429 without retrying.

    The default Teacher treats all 429s as retryable. Only quota classification
    is intercepted; ordinary responses and rate-limit exceptions pass through.
    """
    def __init__(self, client, budget):
        self.client, self.budget = client, budget
        api = getattr(client.responses, 'with_raw_response', None)
        self.api = api or client.responses
        self.responses = SimpleNamespace(with_raw_response=self) if api else self

    def create(self, **kwargs):
        self.budget.check()
        try:
            return self.api.create(**kwargs)
        except Exception as exc:
            if getattr(exc, 'status_code', None) == 429 and not _is_rate_limit_429(exc):
                self.budget.halt('teacher_quota')
                raise TeacherQuotaError('Teacher quota exhausted') from None
            raise


class FakeClient:
    """No SDK, network or model. Used only by explicitly synthetic CLI runs."""
    def __init__(self):
        self.responses = self

    def create(self, **kwargs):
        return SimpleNamespace(output_text='Thought: Synthetic takeover.\nAction: look',
            model='FAKE_NO_MODEL', id='synthetic', status='completed',
            usage={'input_tokens': 100, 'output_tokens': 20,
                   'output_tokens_details': {'reasoning_tokens': 0}})


class TeacherPolicy:
    def __init__(self, teacher, task, budget, *, reservation):
        self.teacher, self.task, self.budget, self.reservation = teacher, task, budget, reservation
        self.failure = None

    def __call__(self, history, visible):
        retry = _TeacherRetry(self.teacher, self.budget)
        try:
            messages = build_messages(history, self.task, visible)
            # UTF-8 bytes plus framing bound, NOT the limiter's char/4 estimate.
            self.reservation.input_bound = len(canonical_bytes(messages)) + 16 * (len(messages) + 1)
            for index in range(len(_RETRY_DELAYS) + 1):
                self.budget.check()
                retry.begin(index)
                try:
                    result = self.teacher.generate(history, self.task, visible)
                    require_completed(result, allow_truncated=False)
                    if result.metadata.get('cache_hit') or not result.raw_text.strip():
                        raise TeacherCallError('Empty reply or unexpected cache hit')
                    return result.raw_text  # Nonempty parser errors retain shared parser semantics.
                except CaseStopped:
                    raise
                except Exception as exc:
                    info = retry.failure(exc)
                    retry.record_failure(exc, info)
                    if not info['will_retry']:
                        raise
                    retry.wait(info['retry_wait_seconds'])
        except CaseStopped:
            self.failure = self.budget.summary()['stop_reason'] or 'budget_stop'
            raise
        except Exception as exc:
            self.failure = self.budget.summary()['stop_reason']
            if not self.failure:
                self.failure = 'teacher_quota' if _teacher_failure(exc, retry.last) == 'QUOTA' else 'teacher_halted'
                self.budget.halt(self.failure)
            raise TeacherCallError('Teacher halted; inspect sanitized call evidence') from None
        finally:
            self.teacher.ledger.retry = None


def make_teacher_policy(config, task, directory, output, budget, *, dry_run=False, client=None):
    """Same Teacher/options as collect_seeds.default_teacher; isolated run cache."""
    spec, directory = config['models']['teacher'], Path(directory)
    reservation = _Reservations(budget, spec['sampling'].get('max_output_tokens', 512))
    teacher = Teacher(cache_dir=directory / 'cache', usage_ledger=directory / 'usage.jsonl',
        run_id=directory.name, provider_model_id=spec['provider_model_id'], sampling=spec['sampling'],
        max_attempts=spec['max_attempts'], timeout_seconds=spec['timeout_seconds'],
        **rate_limit_options(spec, Path(output) / 'data/teacher_rate_limit'),
        secrets_file=config['paths']['secrets_file'], reservation_budget=reservation,
        client=FakeClient() if dry_run else client)
    teacher.ledger = _UsageLedger(teacher.ledger, budget)
    native_client = teacher._client
    # Lazy native creation preserves Teacher's secret handling and SDK options.
    teacher._client = lambda: _QuotaGuard(native_client(), budget)
    return TeacherPolicy(teacher, task, budget, reservation=reservation)


class RecordedEnvironment(DryEnvironment):
    def __init__(self, original, branch_index):
        super().__init__(original)
        self.branch_index = branch_index

    def step(self, action):
        self.index += 1
        if self.index <= self.branch_index:
            return self._result(self.original.steps[self.index - 1])
        return ({'observation': 'Synthetic continuation: no target observed.', 'admissible_commands': ['look']},
                0.0, False, self.index >= self.max_episode_steps, {'won': False})


def run_key(case_id, mode, run):
    return digest([case_id, mode, run])


def usage_summary(path=None):
    rows = [json.loads(line) for line in Path(path).read_text().splitlines()] if path and Path(path).exists() else []
    starts = {r['call_id'] for r in rows if r['status'] == 'started'}
    ends = [r for r in rows if r['status'] not in ('started', 'cache_hit', 'repair_outcome')]
    complete = {r['call_id'] for r in ends}
    return {'totals': {k: sum((r.get('usage') or {}).get(k, 0) or 0 for r in ends)
                       for k in ('input_tokens', 'output_tokens', 'reasoning_tokens')},
            'unknown_usage_attempts': sum(not has_known_usage(r.get('usage')) for r in ends) + len(starts - complete),
            'calls': [{k: r.get(k) for k in ('call_id', 'logical_call_id', 'timestamp', 'status', 'usage',
                       'returned_model_id', 'attempt', 'error_type', 'http_status', 'will_retry',
                       'repair_request_id', 'repair_logical_request_id', 'repair_attempt_index',
                       'failure_class', 'transport_will_retry', 'repair_will_retry',
                       'repair_retry_wait_seconds')} for r in ends],
            'interrupted_requests': len(starts - complete)}


def takeover_metrics(execution, target, branch_index):
    metrics = derive_metrics(execution, target, branch='T', branch_index=branch_index)
    facts = FactTable.from_record(execution.initial, execution.steps)
    pattern = re.compile(r'^(?:heat|cool|clean) ' + re.escape(target) + r' \d+ with .+$', re.I)
    transformed = any(s.get('action_outcome') == 'legal' and pattern.fullmatch(s.get('action_parsed') or '')
        and 'nothing happens.' not in (s.get('visible') or {}).get('observation', '').lower() for s in execution.steps)
    placed = facts.conditions()['placed_count'].get('value')
    suffix_length = max(0, len(execution.steps) - branch_index)
    return {**metrics, 'transform_done': transformed, 'placed': placed > 0 if placed is not None else None,
            'placed_count': placed, 'steps_used_after_branch': suffix_length,
            'steps_to_success': suffix_length if execution.success is True else None,
            'steps_remaining_when_found': execution.max_steps - branch_index - metrics['steps_to_find']
                if metrics['steps_to_find'] is not None else None}


def execute_takeover(case, original, *, mode, run, env_factory, policy_factory, output, run_sha256, budget):
    output = Path(output)
    key = run_key(case['case_id'], mode, run)
    path = output / 'records' / (key + '.json')
    with locked(output / 'locks' / (key + '.lock')):
        if path.exists():
            row = read_json(path)
            if row['run_sha256'] != run_sha256 or row['run_key'] != key:
                raise ValueError('Resume record identity mismatch')
            return row
        branch_index = 0 if mode == 'from_start' else case['branch_index']
        row = dict(schema_version=1, run_key=key, run_sha256=run_sha256, case_id=case['case_id'],
            task_id=case['task_id'], group=case['group'], source=case['source'], mode=mode, run=run,
            branch_index=branch_index, prefix_verified=False, void=True, void_reason=None,
            execution=None, success='unknown', metrics=None, teacher_usage=usage_summary(),
            training_exposure=0)
        stop = budget.summary()['stop_reason']
        if stop:
            row.update(status='not_run_budget' if stop.startswith('global_') else 'not_run_halted', void_reason=stop)
        else:
            directory = output / 'calls' / key / str(uuid.uuid4())
            directory.mkdir(parents=True, exist_ok=False)
            row['call_log_dir'] = str(directory.relative_to(output))
            ledger = BudgetLedger(directory / 'budget.jsonl')
            policy, reached, execution = None, False, None
            try:
                env = env_factory()
                try:
                    version = environment_version(env)
                finally:
                    env.close()
                reference = replace(original, environment_version=version,
                    fingerprint=task_fingerprint(version, original.game_sha256, original.initial['visible'], original.env_seed))
                policy = policy_factory(directory)
                def continuation(history, visible):
                    nonlocal reached
                    reached = True
                    return policy(history, visible)
                execution = execute_actions(env_factory, original.task_id,
                    [s['action_raw'] for s in original.steps[:branch_index]], original.max_steps,
                    prefix_reference=reference, continuation=continuation,
                    ledger=ledger, interaction_kind='evaluation_steps')
                reason = None if reached else execution.error_type or 'prefix_not_completed'
                if execution.termination_reason == 'ENV_ERROR':
                    reason = (policy.failure if policy else None) or execution.error_type or 'execution_error'
                row.update(status='completed' if reason is None else 'void', prefix_verified=reached,
                    void=reason is not None, void_reason=reason, execution=execution.to_json(),
                    success=execution.success if reason is None else 'unknown',
                    metrics=takeover_metrics(execution, case['group'].split('-')[1].lower(), branch_index)
                        if reason is None else None)
            except Exception as exc:
                row.update(status='void', void_reason='setup:' + type(exc).__name__)
            row['teacher_usage'] = usage_summary(directory / 'usage.jsonl')
            row['budget'] = ledger.summary()
            row['teacher_stop_reason'] = budget.summary()['stop_reason']
        write_json(path, row, exclusive=True)
        return row
