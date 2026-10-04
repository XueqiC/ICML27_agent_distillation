"""Run-local POSIX limiter: strict rolling-minute request/token reservations.

Every HTTP attempt reserves prompt UTF-8 bytes (plus framing allowance) and the
output cap. This deliberately overestimates text tokens; it is not billed usage.
Reservations, including failed/unknown calls, are never refunded. All workers
must use the same directory on a filesystem supporting POSIX advisory locks.
"""

import json
from pathlib import Path
import time

from sed.storage import canonical_bytes, digest, locked, write_json


def reserved_tokens(prompt, max_output_tokens):
    return len(canonical_bytes(prompt)) + 256 * (len(prompt) + 1) + max_output_tokens


class RateLimiter:
    def __init__(self, directory, model, *, requests_per_minute=300,
                 tokens_per_minute=400_000, clock=time.time, sleep=time.sleep):
        if any(type(v) is not int or v <= 0 for v in (requests_per_minute, tokens_per_minute)):
            raise ValueError("Teacher rate limits must be positive integers")
        self.limits = {"requests_per_minute": requests_per_minute,
                       "tokens_per_minute": tokens_per_minute}
        self.path = Path(directory) / (digest(model) + ".json")
        self.clock, self.sleep = clock, sleep

    def _read(self):
        state = json.loads(self.path.read_text()) if self.path.exists() else {
            "schema_version": 1, "limits": self.limits, "reservations": [], "blocked_until": 0}
        if state["limits"] != self.limits:
            raise ValueError("Shared teacher limiter configuration changed")
        return state

    def acquire(self, tokens):
        if type(tokens) is not int or not 0 < tokens <= self.limits["tokens_per_minute"]:
            raise ValueError("Request token reservation exceeds teacher TPM limit")
        waited = 0.0
        while True:
            with locked(str(self.path) + ".lock"):
                state = self._read()
                now = self.clock()
                rows = [r for r in state["reservations"] if r[0] + 60 > now]
                state["reservations"] = rows
                wait = max(0.0, state["blocked_until"] - now)
                if (len(rows) >= self.limits["requests_per_minute"]
                        or sum(r[1] for r in rows) + tokens > self.limits["tokens_per_minute"]):
                    wait = max(wait, min(r[0] + 60 - now for r in rows))
                if wait <= 0:
                    rows.append([now, tokens])
                    write_json(self.path, state)
                    return {"limiter_wait_seconds": waited, "reserved_tokens": tokens,
                            "reservation_timestamp": now}
            # Never sleep while holding the lock. Recheck both budgets after waking.
            self.sleep(wait)
            waited += wait

    def defer_until(self, deadline):
        """Share provider cooldowns with every worker, including after final failure."""
        with locked(str(self.path) + ".lock"):
            state = self._read()
            state["blocked_until"] = max(state["blocked_until"], deadline)
            write_json(self.path, state)
