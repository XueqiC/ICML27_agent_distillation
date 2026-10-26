"""Run-wide conservative teacher reservation, including transport retries."""
from repair.runtime import CaseStopped
from repair.ledger import teacher_token_usage
from sed.storage import JsonlLedger
from pathlib import Path


class GlobalTeacherBudget:
    def __init__(self, path, *, input_tokens=5_000_000, output_tokens=500_000, context=None):
        if any(type(v) is not int or v < 0 for v in (input_tokens, output_tokens)):
            raise ValueError("Global token caps must be nonnegative integers")
        if Path(path).exists():
            raise ValueError('Global budget already exists; interrupted runs require explicit reconciliation')
        self.ledger = JsonlLedger(path)
        self.caps = {"input_tokens": input_tokens, "output_tokens": output_tokens}
        self.used = {"input_tokens": 0, "output_tokens": 0}
        self.reported_usage = {k: 0 for k in ("input_tokens", "output_tokens", "reasoning_tokens", "visible_tokens")}
        self.unknown_quantities = {}
        self.calls, self.stop_reason = 0, None
        self.context = dict(context or {})

    def reserve(self, input_bound, output_cap):
        if self.stop_reason:
            raise CaseStopped(self.stop_reason)
        amounts = dict(input_tokens=input_bound, output_tokens=output_cap)
        for key, amount in amounts.items():
            if self.used[key] + amount > self.caps[key]:
                self.stop_reason = "global_" + key + "_cap"
                self.ledger.append({"event": "stopped", "reason": self.stop_reason,
                                    "requested": amounts, **self.summary()})
                raise CaseStopped(self.stop_reason)
        for key, amount in amounts.items():
            self.used[key] += amount
        self.calls += 1
        self.ledger.append({"event": "reservation", "amounts": amounts, **self.summary()})

    def settle(self, input_bound, output_cap, usage):
        # Release only one known completed attempt; unknown retries stay reserved.
        counts = teacher_token_usage(usage)
        for key, reserved in (("input_tokens", input_bound), ("output_tokens", output_cap)):
            self.used[key] += usage[key] - reserved
        for key, amount in counts.items():
            if amount is None:
                self.unknown_quantities[key] = self.unknown_quantities.get(key, 0) + 1
            else:
                self.reported_usage[key] += amount
        self.ledger.append({"event": "settlement", "usage": {**usage, **counts}, **self.summary()})
        if any(self.used[k] > self.caps[k] for k in self.caps):
            self.stop_reason = "global_reported_usage_exceeded_reservation"
            raise CaseStopped(self.stop_reason)

    def summary(self):
        return {**self.context, "caps": dict(self.caps), "tokens_or_unsettled_bounds": dict(self.used),
                "reported_usage": dict(self.reported_usage), "unknown_quantities": dict(self.unknown_quantities),
                "reserved_attempts": self.calls, "stop_reason": self.stop_reason}
