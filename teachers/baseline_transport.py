"""Opt-in baseline transport; never read by trusted seed collection.

SDK retries are implemented at the audited Teacher boundary so every HTTP
attempt has a durable start/end and token reservation. See TRANSPORT_POLICY.md.
"""
from dataclasses import asdict, dataclass


SOURCES = {
    "leap": "third_party/leap_llm/src/leap_llm/utils/openai.py:53-58",
    "steca": "third_party/STeCa/data_collection/data_cali_alfworld.py:345-349",
    "gad": "third_party/LMOps/gad/deepscaler/utils.py:45-59",
    "coa": "SDK default; docs/baselines/NEW_METHOD_SPECS_1019.md (Chain-of-Agents)",
    "agentdistill": "SDK default; docs/baselines/NEW_METHOD_SPECS_1019.md (AgentDistill)",
    "had": "SDK default; docs/baselines/NEW_METHOD_SPECS_1019.md (HAD)",
    "adftp": "third_party/agent-distillation/src/smolagents/models.py:1079-1100",
}


@dataclass(frozen=True)
class BaselineTransport:
    method: str
    upstream: str
    timeout_seconds: float = 600
    connect_timeout_seconds: float = 5
    http_attempts: int = 3
    method_level_tries: int = 1
    rate_limit_attempts: int = 20
    version: str = "baseline-transport-v1"

    def disclosure(self):
        return {**asdict(self), "sdk_max_retries": self.http_attempts - 1,
            "sdk_internal_retries": 0,
            "retry_owner": "audited Teacher loop; SDK-equivalent HTTP retries",
            "backoff": "SDK exponential 0.5s, cap 8s, downward jitter; retry-after-ms/retry-after",
            "429_policy": "retry429: separate 20-attempt budget; retry-after/reset or full jitter 5s..300s; "
                          "no logical slot or method try consumed; quota/exhaustion halt"}


def baseline_transport(method):
    # No fallback for unknown methods: new teacher paths must be audited.
    return BaselineTransport(method, SOURCES[method], method_level_tries=3 if method == "leap" else 1)


def sdk_retry_wait(headers, retry_index, now, jitter):
    """OpenAI BaseClient backoff (Apache-2.0), with sanitized timing only."""
    from teachers.openai_teacher import _seconds
    try:
        delay = float(headers["retry-after-ms"]) / 1000
    except (KeyError, ValueError, TypeError):
        delay = _seconds(headers.get("retry-after", ""), now)
    if 0 < delay <= 60:
        return delay
    return max(0, min(0.5 * 2 ** min(retry_index, 1000), 8) * (1 - .25 * jitter(0, 1)))
