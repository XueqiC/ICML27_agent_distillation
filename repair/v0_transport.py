"""V0 structured calls with audited connection and output-truncation retries.

Adapted from repair.supervision.RationaleLabeler's transport wiring. Each cap
has an immutable capped-client identity; all attempts share one usage ledger
and the run-wide reservations. Exhausted timeouts are retained local failures;
quota, authentication, caps and unknown response outcomes remain fatal.
"""
from dataclasses import dataclass
from functools import partial
import json
from pathlib import Path

from repair.takeover import _Reservations, _UsageLedger, _QuotaGuard, _CONNECTION_ERRORS
from sed.storage import JsonlLedger, canonical_bytes
from teachers.baseline_transport import BaselineTransport
from teachers.capped import from_config
from teachers.openai_teacher import TeacherCallError, TeacherTimeoutError, rate_limit_options
from teachers.rate_limit import RateLimiter
from teachers.stage import require_completed


class TeacherOutputTruncated(Exception):
    """Both output caps exhausted; receipts contain raw output and known usage."""
    def __init__(self, attempts):
        super().__init__('Teacher output truncated at both configured caps')
        self.attempts = attempts


class TeacherTimeout(Exception):
    """Timeout retries exhausted; no review/label was obtained."""
    def __init__(self, attempts):
        super().__init__('V0 teacher timeout retries exhausted')
        self.attempts = attempts


@dataclass(frozen=True)
class V0TransportPolicy(BaselineTransport):
    method: str = 'v0_structured'
    upstream: str = 'repair/v0_transport.py; incident-2 transport policy'
    version: str = 'v0-structured-transport-v1'
    connection_retries_only: bool = True
    timeout_is_recoverable: bool = True
    connection_error_types: tuple = tuple(sorted(_CONNECTION_ERRORS))

    def disclosure(self):
        # Durable counter identity must round-trip through JSON unchanged.
        return dict(super().disclosure(), connection_error_types=list(self.connection_error_types))


class V0TeacherMixin:
    """Opt in only for V0 review/label clients; SDK internal retries stay off."""
    def __init__(self, **kwargs):
        policy = V0TransportPolicy()
        super().__init__(**dict(kwargs, timeout_seconds=policy.timeout_seconds,
            max_attempts=policy.http_attempts, retry_rate_limits_only=False))
        self.transport_policy = policy

    def parse_response(self, raw):
        try:
            self.response_validator(raw)
        except ValueError as exc:
            return None, type(exc).__name__
        return 'v0_structured_response', None


class _V0UsageLedger(_UsageLedger):
    def append(self, row, **kwargs):
        if (row['status'] == 'failed' and row.get('usage') is None
                and row.get('http_status') is None and row.get('error_type') in _CONNECTION_ERRORS):
            # Zero *reported* usage is not proof of zero provider billing. Keep
            # usage=None, the failed receipt and the original reservation.
            row = dict(row, reported_usage=dict(input_tokens=0, output_tokens=0, reasoning_tokens=0),
                       usage_unreported=True)
        super().append(row, **kwargs)
        if row.get('usage_unreported'):
            self.budget.settle(row['call_id'], row['reported_usage'])


class StructuredTransport:
    def __init__(self, config, task_id, directory, budget, *, teacher_class,
                 output_cap, retry_output_cap, phase, validator, client=None):
        spec = config['models']['teacher']
        if spec['requested_name'] != 'gpt-6-luna' or spec['provider_model_id'] != 'gpt-6-luna':
            raise ValueError('V0 requires configured gpt-6-luna')
        self.directory, self.budget = Path(directory), budget
        self.task = dict(task_id=task_id, split='train', sealed=False)
        self.output_cap, self.retry_output_cap, self.phase = output_cap, retry_output_cap, phase
        self.config, self.teacher_class, self.client = config, teacher_class, client
        self.validator = validator
        self.transports = {}
        self.teacher, self.reservation = self._transport(output_cap)

    def _transport(self, cap):
        if cap in self.transports:
            return self.transports[cap]
        spec = self.config['models']['teacher']
        suffix = '' if cap == self.output_cap else f'output-cap-{cap}'
        directory = self.directory / suffix
        teacher = from_config(self.config, output=directory, cache_dir=directory / 'cache',
            usage_ledger=self.directory / 'usage.jsonl', run_id=self.directory.name + ('-' + suffix if suffix else ''),
            cap=80, teacher_class=self.teacher_class,
            sampling={**spec['sampling'], 'max_output_tokens': cap})
        reservation = _Reservations(self.budget, cap)
        native = teacher.teacher
        native.reservation_budget = reservation
        native.ledger = _V0UsageLedger(native.ledger, self.budget)
        if self.client is not None:
            native.client.client = self.client
        original = native._client
        native._client = lambda: _QuotaGuard(original(), self.budget)
        options = rate_limit_options(spec, self.budget.path.parent / 'joint_teacher_rate_limit')
        native.limiter = RateLimiter(options.pop('rate_limit_dir'), spec['provider_model_id'], **options)
        self.transports[cap] = teacher, reservation
        return teacher, reservation

    def __call__(self, prompt):
        from sed.provenance import utc_now
        path = self.directory / 'requests.jsonl'
        rows = [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []
        matching = [r for r in rows if r.get('request') == prompt.to_json()]
        self.budget.check()
        # Do not replay settled requests on resume, including rejected outputs.
        if matching and matching[-1]['status'] == 'completed':
            return matching[-1]['raw_output']
        if matching and matching[-1]['status'] == 'teacher_timeout':
            raise TeacherTimeout(matching[-1]['teacher_attempts'])
        if matching and matching[-1]['status'] != 'teacher_output_truncated':
            self.budget.halt(f'v0_interrupted_{self.phase}_request')
            raise TeacherCallError('V0 previous request lacks a known outcome')
        attempts = [r for r in matching if r['status'] == 'teacher_output_truncated']
        ledger = JsonlLedger(path)
        for cap in (self.output_cap, self.retry_output_cap):
            if any(r.get('output_cap') == cap for r in attempts):
                continue
            row = dict(timestamp=utc_now(), request=prompt.to_json(), status='started', output_cap=cap)
            ledger.append(row)
            try:
                self.budget.check()
                teacher, reservation = self._transport(cap)
                teacher.teacher.response_validator = partial(self.validator, prompt=prompt)
                reservation.input_bound = len(canonical_bytes(prompt.messages())) + 16 * (len(prompt.messages()) + 1)
                result = teacher.generate([], self.task,
                    {'observation': canonical_bytes(prompt.to_json()).decode(), 'admissible_commands': []})
                require_completed(result)
                if result.metadata.get('cache_hit') or result.metadata['usage']['output_tokens'] > cap:
                    raise TeacherCallError('Invalid V0 completion')
                if (result.truncated and result.metadata['response_status'] == 'incomplete'
                        and result.metadata.get('incomplete_reason') == 'max_output_tokens'):
                    receipt = dict(row, status='teacher_output_truncated', error_type='TruncatedOutput',
                                   raw_output=result.raw_text, metadata=result.metadata)
                    ledger.append(receipt)
                    attempts.append(receipt)
                    # A concurrent hard stop must also stop this worker.
                    self.budget.check()
                    continue
                require_completed(result, allow_truncated=False)
                ledger.append(dict(row, status='completed', raw_output=result.raw_text, metadata=result.metadata))
                return result.raw_text
            except TeacherTimeoutError:
                # A concurrent quota/cap stop still wins over this local failure.
                self.budget.check()
                usage = [json.loads(line) for line in (self.directory / 'usage.jsonl').read_text().splitlines()]
                terminal = usage[-1]
                attempts = [r for r in usage if r.get('logical_call_id') == terminal['logical_call_id']
                            and r['status'] != 'started']
                ledger.append(dict(row, status='teacher_timeout', raw_output=None,
                                   teacher_attempts=attempts))
                raise TeacherTimeout(attempts) from None
            except Exception as exc:
                reason = self.budget.summary()['stop_reason'] or f'{self.phase}_transport_failed'
                self.budget.halt(reason)
                ledger.append(dict(row, status='failed', error_type=type(exc).__name__, reason=reason))
                raise TeacherCallError(f'V0 {self.phase} halted; see sanitized request ledgers') from None
        raise TeacherOutputTruncated(attempts)
