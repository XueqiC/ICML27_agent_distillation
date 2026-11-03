"""V0 structured calls with one bounded retry for known output truncation.

Adapted from repair.supervision.RationaleLabeler's transport wiring. Each cap
has an immutable capped-client identity; all attempts share one usage ledger
and the run-wide reservations. Unknown usage and transport errors stay fatal.
"""
import json
from pathlib import Path

from repair.takeover import _Reservations, _UsageLedger, _QuotaGuard
from sed.storage import JsonlLedger, canonical_bytes
from teachers.capped import from_config
from teachers.openai_teacher import TeacherCallError, rate_limit_options
from teachers.rate_limit import RateLimiter
from teachers.stage import require_completed


class TeacherOutputTruncated(Exception):
    """Both output caps exhausted; receipts contain raw output and known usage."""
    def __init__(self, attempts):
        super().__init__('Teacher output truncated at both configured caps')
        self.attempts = attempts


class StructuredTransport:
    def __init__(self, config, task_id, directory, budget, *, teacher_class,
                 output_cap, retry_output_cap, phase, client=None):
        spec = config['models']['teacher']
        if spec['requested_name'] != 'gpt-6-luna' or spec['provider_model_id'] != 'gpt-6-luna':
            raise ValueError('V0 requires configured gpt-6-luna')
        self.directory, self.budget = Path(directory), budget
        self.task = dict(task_id=task_id, split='train', sealed=False)
        self.output_cap, self.retry_output_cap, self.phase = output_cap, retry_output_cap, phase
        self.config, self.teacher_class, self.client = config, teacher_class, client
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
        native.ledger = _UsageLedger(native.ledger, self.budget)
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
                if not result.raw_text.strip():
                    raise TeacherCallError('Empty V0 completion')
                ledger.append(dict(row, status='completed', raw_output=result.raw_text, metadata=result.metadata))
                return result.raw_text
            except Exception as exc:
                reason = self.budget.summary()['stop_reason'] or f'{self.phase}_transport_failed'
                self.budget.halt(reason)
                ledger.append(dict(row, status='failed', error_type=type(exc).__name__, reason=reason))
                raise TeacherCallError(f'V0 {self.phase} halted; see sanitized request ledgers') from None
        raise TeacherOutputTruncated(attempts)
