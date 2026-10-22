"""Small, durable startup sidecar; never part of a frozen training identity."""
from contextlib import contextmanager
from contextvars import ContextVar
from functools import wraps
import os
from pathlib import Path
import time
import uuid

from sed.storage import JsonlLedger

_ACTIVE = ContextVar("startup_timing", default=None)


def timing_provenance():
    timer = _ACTIVE.get()
    return timer.disclosure if timer else None


class StartupTiming:
    def __init__(self, path, *, process, mode, snapshot):
        self.started = time.monotonic()
        self.snapshot = Path(snapshot).resolve()
        self.disclosure = {"path": str(Path(path).resolve()), "invocation_id": str(uuid.uuid4()),
                           "process": process, "mode": mode,
                           "parent_log": os.environ.get("SED_STARTUP_PARENT_LOG"),
                           "units": "seconds", "nested_phases": "inclusive; do not sum"}
        self.ledger = JsonlLedger(path)

    def record(self, phase, started, *, status="complete"):
        now = time.monotonic()
        self.ledger.append({**self.disclosure, "phase": phase, "status": status,
                            "seconds": now - started, "elapsed_seconds": now - self.started,
                            "timestamp_unix": time.time()})


@contextmanager
def startup_timing(path, *, process, mode, snapshot):
    timer = StartupTiming(path, process=process, mode=mode, snapshot=snapshot)
    token = _ACTIVE.set(timer)
    parent = os.environ.get("SED_STARTUP_PARENT_LOG")
    os.environ["SED_STARTUP_PARENT_LOG"] = timer.disclosure["path"]
    try:
        timer.record("startup", timer.started, status="started")
        yield timer
    finally:
        _ACTIVE.reset(token)
        if parent is None:
            os.environ.pop("SED_STARTUP_PARENT_LOG", None)
        else:
            os.environ["SED_STARTUP_PARENT_LOG"] = parent


@contextmanager
def phase(name):
    timer = _ACTIVE.get()
    started = time.monotonic()
    try:
        yield
    except BaseException:
        if timer:
            timer.record(name, started, status="failed")
        raise
    else:
        if timer:
            timer.record(name, started)


def timed(name):
    def decorate(function):
        @wraps(function)
        def run(*args, **kwargs):
            with phase(name):
                return function(*args, **kwargs)
        return run
    return decorate


@contextmanager
def snapshot_hashing(directory):
    timer = _ACTIVE.get()
    if timer and Path(directory).resolve() == timer.snapshot:
        with phase("snapshot_hashing"):
            yield
    else:
        yield


def first_update(started):
    timer = _ACTIVE.get()
    if timer:
        timer.record("first_update", started)
