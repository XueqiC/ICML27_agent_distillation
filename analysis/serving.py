"""Recorded serving-noise reference from MECHANISM_PLAN Amendment 2."""

SERVING_NOISE_REFERENCE = {
    "source": "docs/MECHANISM_PLAN.md Amendment 2 (2026-10-04)",
    "comparison": "cross-server base-vs-base, rai dev140, greedy",
    "batch_invariant": False,
    "discordant": 12,
    "n": 140,
    "first_only": 6,
    "second_only": 6,
    "rate": 12 / 140,
    "scope": "Historical non-batch-invariant reference; not a noise estimate for batch-invariant runs.",
}

SERVING_NOISE_TEXT = (
    "Historical cross-server base-vs-base serving noise: 12/140 discordant outcomes "
    "(6 each way; rai dev140, batch_invariant=0; MECHANISM_PLAN Amendment 2). "
    "This is not a noise estimate for batch-invariant runs."
)
