# Shared teacher engineering fixes

2026-10-19: transport failures propagate through seed collection, full TRAIN,
shared episode execution, evaluation, rewrites/corrections and diagnostic
branches. Official output rejections remain separate. This changes failure
handling and accounting, not a research algorithm. No real API calls or training
were used to validate this change.

See [transport failure contract](../docs/TEACHER_TRANSPORT_FAILURES.md).
