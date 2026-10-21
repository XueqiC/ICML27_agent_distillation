"""Durable logical-call caps and a shared, non-refundable reservation ledger.

By default only rate-limit HTTP 429s retry (up to 20 attempts in one slot).
Baselines opt into their audited upstream HTTP policy in that same logical slot.
Every attempt reserves tokens; exhausted transport and quota failures halt.
Cap-truncated outputs consume their slot as rejected outputs without halting.
An interrupted reservation also halts the run, including during 429 backoff.
Use one counter/run ID for a collector; never reset it to resume a failed run.
Explicit recovery acknowledges failures without refunding any reservation.
"""
import json
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

from sed.manifest import authorize_task
from sed.provenance import utc_now
from sed.react import PROMPT_VERSION
from sed.storage import digest, locked, write_json
from sed.visibility import visible_history, visible_observation, visible_task
from teachers.openai_teacher import Teacher, TeacherCallError, TeacherResult, read_api_key


class BudgetExceeded(TeacherCallError):
    pass


class CollectorHalted(TeacherCallError):
    pass


class CollectionBudgetExceeded(BudgetExceeded):
    incomplete = True


class SharedReservations:
    """Concurrent logical-call/input reservations for retrying teachers.

    SQLite commits each reservation before transport; unknown/failed requests are
    never refunded. A logical call may have several HTTP attempts, each reserving
    its input estimate. Only short transactions hold the writer lock, never
    network calls or limiter waits.
    """
    def __init__(self, path, *, max_calls=160_000, max_input_tokens=700_000_000):
        if any(type(v) is not int or v < 0 for v in (max_calls, max_input_tokens)):
            raise ValueError("Nonnegative integer collection caps required")
        self.path = Path(path)
        self.limits = {"max_calls": max_calls, "max_input_tokens": max_input_tokens}
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._transaction() as db:
            db.execute("CREATE TABLE IF NOT EXISTS budget (id INTEGER PRIMARY KEY, value TEXT NOT NULL)")
            db.execute("CREATE TABLE IF NOT EXISTS calls (id TEXT PRIMARY KEY)")
            db.execute("CREATE TABLE IF NOT EXISTS requests (id TEXT PRIMARY KEY, logical_id TEXT, "
                       "input_tokens INTEGER, task_id TEXT, usage_ledger TEXT, timestamp TEXT)")
            state = self._state(db)
            if state["limits"] != self.limits:
                raise ValueError("Collection reservation caps changed on resume")
            self._save(db, state)

    @contextmanager
    def _transaction(self):
        db = sqlite3.connect(self.path, timeout=60)
        try:
            db.execute("PRAGMA synchronous=FULL")
            db.execute("BEGIN IMMEDIATE")
            yield db
            db.commit()
        except BaseException:
            db.rollback()
            raise
        finally:
            db.close()

    def _state(self, db):
        row = db.execute("SELECT value FROM budget WHERE id=1").fetchone()
        return json.loads(row[0]) if row else dict(limits=self.limits, logical_calls=0,
            reserved_input_tokens=0, transport_reservations=0, stopped=None)

    @staticmethod
    def _save(db, state):
        db.execute("INSERT OR REPLACE INTO budget VALUES (1, ?)", (json.dumps(state),))

    def snapshot(self):
        with self._transaction() as db:
            return self._state(db)

    def reserve(self, *, logical_id, request_id, input_tokens, task_id, usage_ledger):
        if type(input_tokens) is not int or input_tokens <= 0:
            raise ValueError("Positive input token reservation required")
        with self._transaction() as db:
            state = self._state(db)
            new_call = not db.execute("SELECT 1 FROM calls WHERE id=?", (logical_id,)).fetchone()
            if not state["stopped"]:
                if state["logical_calls"] + new_call > self.limits["max_calls"]:
                    state["stopped"] = "logical_call_cap"
                elif state["reserved_input_tokens"] + input_tokens > self.limits["max_input_tokens"]:
                    state["stopped"] = "reserved_input_token_cap"
            if not state["stopped"]:
                db.execute("INSERT OR IGNORE INTO calls VALUES (?)", (logical_id,))
                db.execute("INSERT INTO requests VALUES (?, ?, ?, ?, ?, ?)",
                           (request_id, logical_id, input_tokens, task_id, str(usage_ledger), utc_now()))
                state["logical_calls"] += new_call
                state["reserved_input_tokens"] += input_tokens
                state["transport_reservations"] += 1
            self._save(db, state)
        if state["stopped"]:
            raise CollectionBudgetExceeded("Collection cap reached; no request made")


def state_hash(history, task, visible_context):
    return digest({"prompt_version": PROMPT_VERSION, "task_id": task["task_id"],
                   "task": visible_task(task), "history": visible_history(history),
                   "visible_context": visible_observation(visible_context)})


class _HTTPFailure(Exception):
    def __init__(self, raw):
        self.status_code = raw.status_code
        self.response = getattr(raw, "http_response", raw)
        self.body = None
        if self.status_code == 429:
            try:
                self.body = (self.response.json() if hasattr(self.response, "json") else raw.parse())
            except Exception:
                pass
        self.request_id = getattr(self.response, "headers", {}).get("x-request-id")
        super().__init__("Teacher transport did not return HTTP 200")


class _Transport:
    def __init__(self, owner, client):
        self.owner, self.client = owner, client
        self.responses = SimpleNamespace(with_raw_response=self)

    def create(self, **kwargs):
        owner = self.owner
        counter = owner._counter
        # The pending slot spans retries of this logical call. Each transport
        # attempt still has its own durable ledger/token reservation in Teacher.
        if not counter["pending"]:
            if counter["attempts"] >= owner.cap:
                raise BudgetExceeded("Teacher attempt cap reached")
            counter["attempts"] += 1
        counter["pending"] = True
        write_json(owner.counter_path, counter)
        if self.client is None:
            from openai import OpenAI
            secret = read_api_key(owner.teacher.secrets_file)
            owner.teacher._secret = secret
            self.client = OpenAI(api_key=secret, max_retries=0, timeout=owner.teacher.client_timeout())
        # Require the HTTP response interface so non-200 (including 201) cannot pass.
        raw = self.client.responses.with_raw_response.create(**kwargs)
        if raw.status_code != 200:
            raise _HTTPFailure(raw)
        return raw


class CappedTeacher:
    def __init__(self, *, counter_path, cap, cache_dir, usage_ledger, run_id,
                 dry_run=False, client=None, teacher_class=Teacher, evaluation_only=False, **teacher_options):
        if type(cap) is not int or cap < 0 or not run_id:
            raise ValueError("Nonnegative integer cap and stable collector run_id required")
        if {"max_attempts", "purpose", "retry_rate_limits_only"} & teacher_options.keys():
            raise ValueError("Capped collectors fix 20 rate-limit-only attempts and purpose")
        if client is not None and getattr(client, "max_retries", 0) != 0:
            raise ValueError("Injected SDK client must disable retries")
        self.counter_path, self.cap = Path(counter_path), cap
        self.evaluation_only = evaluation_only
        if type(evaluation_only) is not bool:
            raise ValueError("evaluation_only must be an explicit boolean")
        self.dry_run, self.run_id = dry_run, run_id
        self._counter, self._planned = None, set()
        self.teacher = teacher_class(cache_dir=cache_dir, usage_ledger=usage_ledger,
            run_id=run_id, max_attempts=20, retry_rate_limits_only=True,
            purpose="evaluation" if evaluation_only else "generation", **teacher_options)
        self.teacher.client = _Transport(self, client)
        self.identity = {"schema_version": 1, "run_id": run_id, "cap": cap,
                         "usage_ledger": str(Path(usage_ledger).resolve()),
                         "provider_model_id": self.teacher.provider_model_id,
                         "sampling": self.teacher.sampling,
                         "prompt_template_sha256": digest(self.teacher.prompt_template)}
        if evaluation_only:
            self.identity["purpose"] = "dev_evaluation_only"
        if self.teacher.transport_policy:
            self.identity["baseline_transport"] = self.teacher.transport_policy.disclosure()

    def _read_counter(self):
        value = (json.loads(self.counter_path.read_text()) if self.counter_path.exists() else
                 {"identity": self.identity, "attempts": 0, "pending": False, "halted": False})
        ledger = self.teacher.ledger.path
        try:
            rows = [json.loads(line) for line in ledger.read_text().splitlines()] if ledger.exists() else []
        except ValueError:
            raise TeacherCallError("Teacher usage ledger is incomplete/unknown; inspect retained evidence") from None
        from teachers.chain_recovery import inherited_counter
        value, rows, inherited, self.cap = inherited_counter(self.counter_path, value, self.identity, rows)
        if value["identity"] != self.identity:
            raise ValueError("Collector identity/cap changed")
        rows = [r for r in rows if r.get("run_id") == self.run_id]
        all_rows = rows
        recovery = value.get("recovery")
        if recovery:
            boundary = recovery["through_rows"]
            if len(rows) < boundary or digest(rows[:boundary]) != recovery["ledger_prefix_sha256"]:
                raise TeacherCallError("Recovered teacher ledger changed; refusing to reset accounting")
            rows = rows[boundary:]
        starts = {r["call_id"] for r in rows if r["status"] == "started"}
        ends = {r["call_id"] for r in rows if r["status"] in {"success", "rejected", "failed", "incomplete"}}
        logical_starts = {r.get("logical_call_id", r["call_id"]) for r in all_rows if r["status"] == "started"}
        completed = {r["logical_call_id"] for r in rows
                     if r["status"] in {"success", "rejected"} and "logical_call_id" in r}
        failed = any(r["status"] in {"failed", "incomplete"} and not (
            (r.get("http_status") == 429 or self.teacher.transport_policy is not None)
            and r.get("retryable") and r.get("will_retry")
            and r.get("logical_call_id") in completed) for r in rows)
        # Never lower a reservation count. Ledger recovers a missing/stale counter.
        value["attempts"] = max(value["attempts"], inherited + len(logical_starts))
        if value["pending"] or starts - ends or failed:
            value["halted"] = True
        return value

    def check_health(self):
        if self._read_counter()["halted"]:
            raise CollectorHalted("Teacher transport halted; inspect retained evidence and explicitly recover or rerun")

    def _cached(self, key):
        path = self.teacher.cache_dir / (key + ".json")
        if not path.exists():
            return False
        cached = json.loads(path.read_text())
        if cached["cache_key"] != key or digest(cached["result"]) != cached["result_sha256"]:
            raise TeacherCallError("Teacher cache integrity check failed")
        from teachers.stage import require_completed
        result = cached["result"]
        require_completed(TeacherResult(result["raw_text"], result["parsed_action"], result["metadata"]),
                          allow_truncated=False)
        return True

    def generate(self, history, task, visible_context):
        if self.evaluation_only:
            authorize_task(task, "evaluation")
            if task["split"] != "dev" or task.get("sealed"):
                raise ValueError("Capped evaluation teacher requires unsealed DEV tasks")
        else:
            authorize_task(task, "generation")
        key = self.teacher.cache_key(history, task, visible_context)
        sid = state_hash(history, task, visible_context)
        if self.dry_run:
            counter = self._read_counter()
            hit = self._cached(key) or key in self._planned
            if not hit:
                self._planned.add(key)
            return TeacherResult("", None, {"dry_run": True, "cache_hit": hit,
                "cache_key": key, "state_hash": sid, "would_call": not hit,
                "would_call_total": len(self._planned), "attempts_used": counter["attempts"],
                "remaining_cap": max(0, self.cap - counter["attempts"]),
                "within_cap": len(self._planned) <= self.cap - counter["attempts"],
                "halted": counter["halted"]})
        with locked(str(self.counter_path) + ".lock"):
            self._counter = self._read_counter()
            if self._counter["halted"]:
                raise CollectorHalted("Collector halted; inspect sanitized counter/usage ledger")
            if not self._cached(key) and self._counter["attempts"] >= self.cap:
                raise BudgetExceeded("Teacher attempt cap reached; no request made")
            write_json(self.counter_path, self._counter)
            try:
                result = self.teacher.generate(history, task, visible_context)
                from teachers.stage import require_completed
                require_completed(result)
            except BaseException:
                self._counter.update(halted=True, pending=False)
                write_json(self.counter_path, self._counter)
                raise
            self._counter["pending"] = False
            write_json(self.counter_path, self._counter)
            self.teacher.ledger.append({"run_id": self.run_id, "timestamp": utc_now(),
                "status": "state_link", "call_id": result.metadata.get("call_id"),
                "state_hash": sid, "cache_key": key, "cache_hit": result.metadata["cache_hit"],
                "attempts_reserved": self._counter["attempts"]})
            return result._replace(metadata={**result.metadata, "state_hash": sid})


def from_config(config, *, output, cache_dir, usage_ledger, run_id, cap, dry_run=False,
                teacher_class=Teacher, sampling=None, baseline_method=None):
    from teachers.openai_teacher import rate_limit_options
    spec = config["models"]["teacher"]
    return CappedTeacher(counter_path=Path(output) / "counter.json", cap=cap,
        cache_dir=cache_dir, usage_ledger=usage_ledger, run_id=run_id, dry_run=dry_run,
        teacher_class=teacher_class, provider_model_id=spec["provider_model_id"],
        baseline_method=baseline_method,
        sampling=sampling or spec["sampling"], timeout_seconds=spec["timeout_seconds"],
        secrets_file=config["paths"]["secrets_file"],
        **rate_limit_options(spec, Path(usage_ledger).parent / "r03_rate_limit"))
