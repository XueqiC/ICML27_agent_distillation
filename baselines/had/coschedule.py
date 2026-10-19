"""Filesystem-only orchestration of the unchanged HAD stage transactions.

Exit 75 releases an idle GPU allocation at a resumable boundary. The launcher
submits its replacement. Scheduling logs are separate from scientific artifacts.
"""
from contextlib import contextmanager
import fcntl
import json
import math
import os
from pathlib import Path
import time

from baselines.common.pipeline import stage_run
from baselines.had.data import require_cpu_teacher
from sed.storage import JsonlLedger, digest


RESUBMIT = 75


class WaitExpired(Exception):
    """No runnable GPU stage became available within the idle budget."""


class PeerStageHalted(RuntimeError):
    """An existing capped-teacher failure cannot be fixed by GPU resubmission."""


class Waiter:
    def __init__(self, output, role, *, limit=1200, initial=1, maximum=30,
                 clock=time.monotonic, sleep=time.sleep):
        if (not all(math.isfinite(v) for v in (limit, initial, maximum))
                or not 0 < limit <= 1200 or not 0 < initial <= maximum <= 60):
            raise ValueError("Require 0 < GPU wait <= 1200 and 0 < poll <= max poll <= 60 seconds")
        self.role, self.limit = role, limit
        self.initial, self.maximum = initial, maximum
        self.clock, self.sleep = clock, sleep
        self.idle_since = None
        self.identity = None
        self.ledger = JsonlLedger(Path(output) / "scheduling" / (role + ".jsonl"))

    def log(self, event, **details):
        self.ledger.append({"event": event, "role": self.role, "timestamp": time.time(),
            "pid": os.getpid(), "job_id": os.environ.get("SLURM_JOB_ID"),
            "identity_sha256": self.identity, **details})

    def progress(self):
        self.idle_since = None

    def until(self, ready, reason):
        """Log before every sleep so SIGKILL cannot erase evidence of a wait."""
        started, delay = None, self.initial
        outcome = "interrupted"
        try:
            while not ready():
                now = self.clock()
                if started is None:
                    started = now
                    if self.idle_since is None:
                        self.idle_since = now
                    self.log("wait_started", reason=reason, limit_seconds=self.limit if self.role == "gpu" else None)
                elapsed = now - self.idle_since
                if self.role == "gpu" and elapsed >= self.limit:
                    outcome = "resubmit"
                    raise WaitExpired()
                duration = min(delay, self.limit - elapsed) if self.role == "gpu" else delay
                self.log("wait_poll", reason=reason, elapsed_seconds=now - started,
                         idle_seconds=elapsed, sleep_seconds=duration)
                self.sleep(duration)
                delay = min(self.maximum, delay * 2)
            outcome = "ready"
        finally:
            if started is not None:
                self.log("wait_finished", reason=reason, outcome=outcome,
                         elapsed_seconds=self.clock() - started)

    @contextmanager
    def lock(self, path, *, shared=False):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fd = os.open(path, os.O_CREAT | os.O_RDWR, 0o600)
        def acquire():
            try:
                fcntl.flock(fd, (fcntl.LOCK_SH if shared else fcntl.LOCK_EX) | fcntl.LOCK_NB)
            except BlockingIOError:
                return False
            return True
        try:
            self.until(acquire, "lock:" + path.name)
            yield
        finally:
            os.close(fd)


def worker_pipeline(args, identity, config, waiter):
    # Import here to keep the ordinary per-stage entry point independent.
    from baselines.had.pipeline import train_pipeline
    if args.worker == "teacher":
        require_cpu_teacher()
    waiter.identity = digest(identity)

    def execute(root, name, key, action, *, selected):
        own = (name.startswith("teacher-") == (args.worker == "teacher"))
        directory = Path(root) / "stages" / name
        complete = directory / "complete.json"
        if not own:
            def ready():
                if complete.exists():
                    return True
                if name.startswith("teacher-"):
                    attempts = sorted(directory.glob("attempt-*"))
                    failure = attempts[-1] / "failure.json" if attempts else None
                    if failure is not None and failure.exists():
                        error = json.loads(failure.read_text())["error_type"]
                        if error in {"BudgetExceeded", "CollectorHalted", "TeacherCallError"}:
                            raise PeerStageHalted("HAD teacher halted; inspect retained stage/counter evidence")
                return False
            waiter.until(ready, "stage:" + name)
        with waiter.lock(directory / ".stage.lock"):
            if not own and not complete.exists():
                raise ValueError("Completed HAD prerequisite disappeared")
            def run(attempt):
                if not own:
                    raise ValueError("Worker cannot execute a peer's stage")
                waiter.progress()
                waiter.log("stage_started", stage=name, stage_identity=key)
                action(attempt)
                waiter.progress()
            # Always validate identity AND hashes, including reused stages. A
            # completion's existence alone never authorizes downstream work.
            reused = complete.exists()
            result = stage_run(root, name, key, run)
            waiter.log("stage_reused" if reused else "stage_complete", stage=name, stage_identity=key)
        return result

    from argparse import Namespace
    all_rounds = Namespace(**{**vars(args), "through": "train-249"})
    train_pipeline(all_rounds, identity, config, execute_stage=execute,
                   publication_lock=waiter.lock)


def run_worker(args, config):
    from baselines.had import RECIPE
    from baselines.had.pipeline import prepare, update_disclosure
    if args.worker == "teacher":
        require_cpu_teacher()
    waiter = Waiter(args.output, args.worker, limit=args.gpu_wait_seconds,
                    initial=args.poll_seconds, maximum=args.poll_max_seconds)
    output = Path(args.output)
    identity = None
    waiter.log("worker_started")
    try:
        # Shared locks allow the pair to coexist, while excluding legacy
        # per-stage/evaluation writers. Same-role duplicates cannot race attempts.
        with waiter.lock(output / ".pipeline.lock", shared=True), \
                waiter.lock(output / ("." + args.worker + ".worker.lock")):
            with waiter.lock(output / ".prepare.lock"):
                identity = prepare(args, config, method="had", recipe=RECIPE, registered_k=None)
            waiter.identity = digest(identity)
            try:
                worker_pipeline(args, identity, config, waiter)
            finally:
                with waiter.lock(output / ".disclosure.lock"):
                    update_disclosure(output, identity)
        waiter.log("worker_complete")
        return 0
    except WaitExpired:
        waiter.log("worker_resubmit", reason="gpu_idle_limit")
        return RESUBMIT
    except BaseException as exc:
        waiter.log("worker_stopped", error_type=type(exc).__name__)
        raise
